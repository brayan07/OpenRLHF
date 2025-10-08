import torch
import types

from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker, Experience


class DummyArgs:
    def __init__(self,
                 use_dynamic_batch=False,
                 micro_rollout_batch_size=16,
                 actor_num_nodes=1,
                 actor_num_gpus_per_node=8,
                 ring_attn_size=1,
                 ds_tensor_parallel_size=1,
                 rollout_max_tokens_per_gpu=4096,
                 ):
        self.use_dynamic_batch = use_dynamic_batch
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.actor_num_nodes = actor_num_nodes
        self.actor_num_gpus_per_node = actor_num_gpus_per_node
        self.ring_attn_size = ring_attn_size
        self.ds_tensor_parallel_size = ds_tensor_parallel_size
        self.rollout_max_tokens_per_gpu = rollout_max_tokens_per_gpu

        # required by other methods but unused in these tests
        self.n_samples_per_prompt = 4
        self.advantage_estimator = "reinforce"
        self.remote_rm_url = None


class DummyStrategy:
    def __init__(self, args):
        self.args = args


class DummyTokenizer:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id


def make_sample(total_length: int, seq_len: int = None) -> Experience:
    # create a single-sample Experience with tensors of given length
    if seq_len is None:
        seq_len = total_length
    sequences = torch.ones(seq_len, dtype=torch.long)
    attention_mask = torch.ones(seq_len, dtype=torch.long)
    # action mask length can differ; keep same for simplicity
    action_mask = torch.ones(seq_len - 1 if seq_len > 0 else 0, dtype=torch.bool)
    info = {
        "total_length": torch.tensor([total_length], dtype=torch.long)
    }
    return Experience(
        sequences=sequences.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        action_mask=action_mask.unsqueeze(0) if action_mask.numel() > 0 else torch.zeros((1, 0), dtype=torch.bool),
        info=info,
    )
 
def test_split_rollout_samples_static_batches():
    # Build maker locally
    args = DummyArgs(use_dynamic_batch=False, micro_rollout_batch_size=2)
    strategy = DummyStrategy(args)
    tokenizer = DummyTokenizer(pad_token_id=0)
    maker_static = RemoteExperienceMaker(
        actor_model_group=None,
        critic_model_group=None,
        reward_model_group=None,
        initial_model_group=None,
        kl_controller=types.SimpleNamespace(value=0.0),
        strategy=strategy,
        tokenizer=tokenizer,
    )
    # 5 samples, batch size 2 -> 3 concatenated batches
    rollout = [make_sample(total_length=i + 1) for i in range(5)]

    batches = maker_static.split_rollout_samples(rollout)

    assert len(batches) == 3

    # verify indices preserved per batch
    expected_indices = [
        [0, 1],
        [2, 3],
        [4],
    ]
    for batch, exp_idx in zip(batches, expected_indices):
        # index was set before concat; concat sums lists
        assert batch.index == exp_idx
        # sequences batch dimension equals number of items in this batch
        assert batch.sequences.shape[0] == len(exp_idx)
        # info contains total_length stacked for each item
        assert "total_length" in batch.info
        assert batch.info["total_length"].shape[0] == len(exp_idx)


def test_split_rollout_samples_dynamic_partitions(monkeypatch):
    # Build maker and args locally
    args = DummyArgs(
        use_dynamic_batch=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        # Make effective_actor_num small to allow num_batch we want via monkeypatch
        ring_attn_size=8,
        ds_tensor_parallel_size=1,
    )
    strategy = DummyStrategy(args)
    tokenizer = DummyTokenizer(pad_token_id=0)

    maker_dynamic = RemoteExperienceMaker(
        actor_model_group=None,
        critic_model_group=None,
        reward_model_group=None,
        initial_model_group=None,
        kl_controller=types.SimpleNamespace(value=0.0),
        strategy=strategy,
        tokenizer=tokenizer,
    )

    # Monkeypatch helpers deterministically
    import openrlhf.trainer.ppo_utils.experience_maker as em
    # Force minimum batch num to 2 so num_batch becomes 2 (effective_actor_num=1 via ring_attn_size=8)
    monkeypatch.setattr(em, "get_minimum_num_micro_batch_size", lambda total_lengths, *_: 2)

    def fake_partitions(total_lengths, num_batch, _):
        assert len(total_lengths) == 5
        assert num_batch == 2  # we forced minimum to 2 and effective to 1
        return [[0, 2, 4], [1, 3]]

    monkeypatch.setattr(em, "get_seqlen_balanced_partitions", fake_partitions)

    # 5 samples, partitions enforced to [[0,2,4],[1,3]]
    rollout = [make_sample(total_length=i + 1) for i in range(5)]

    batches = maker_dynamic.split_rollout_samples(rollout)

    # Should match number of partitions
    assert len(batches) == 2

    # Check that order within each micro-batch matches our fake partitions
    assert batches[0].index == [0, 2, 4]
    assert batches[1].index == [1, 3]

    # Sanity check shapes align
    assert batches[0].sequences.shape[0] == 3
    assert batches[1].sequences.shape[0] == 2


def test_rewards_preserved_static_batches():
    # Build maker locally
    args = DummyArgs(use_dynamic_batch=False, micro_rollout_batch_size=2)
    strategy = DummyStrategy(args)
    tokenizer = DummyTokenizer(pad_token_id=0)
    maker = RemoteExperienceMaker(
        actor_model_group=None,
        critic_model_group=None,
        reward_model_group=None,
        initial_model_group=None,
        kl_controller=types.SimpleNamespace(value=0.0),
        strategy=strategy,
        tokenizer=tokenizer,
    )

    # Create samples with per-sample rewards equal to their original index
    rollout = []
    for i in range(5):
        s = make_sample(total_length=i + 1)
        s.rewards = torch.tensor([[float(i)]])  # shape (1, 1)
        rollout.append(s)

    batches = maker.split_rollout_samples(rollout)

    # Verify that rewards (first column) match the batch indices in order
    expected_indices = [[0, 1], [2, 3], [4]]
    for batch, exp_idx in zip(batches, expected_indices):
        rewards_vals = batch.rewards[:, 0].tolist()
        assert rewards_vals == [float(x) for x in exp_idx]


def test_rewards_preserved_dynamic_partitions(monkeypatch):
    # Dynamic config with effective_actor_num=1 to simplify
    args = DummyArgs(
        use_dynamic_batch=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        ring_attn_size=8,  # effective_actor_num = 1
        ds_tensor_parallel_size=1,
    )
    strategy = DummyStrategy(args)
    tokenizer = DummyTokenizer(pad_token_id=0)
    maker = RemoteExperienceMaker(
        actor_model_group=None,
        critic_model_group=None,
        reward_model_group=None,
        initial_model_group=None,
        kl_controller=types.SimpleNamespace(value=0.0),
        strategy=strategy,
        tokenizer=tokenizer,
    )

    # Partition into two micro-batches deterministically
    import openrlhf.trainer.ppo_utils.experience_maker as em
    monkeypatch.setattr(em, "get_minimum_num_micro_batch_size", lambda total_lengths, *_: 2)

    def fake_partitions(total_lengths, num_batch, _):
        assert len(total_lengths) == 5
        assert num_batch == 2
        return [[0, 2, 4], [1, 3]]

    monkeypatch.setattr(em, "get_seqlen_balanced_partitions", fake_partitions)

    # Create samples with rewards equal to index
    rollout = []
    for i in range(5):
        s = make_sample(total_length=i + 1)
        s.rewards = torch.tensor([[float(i)]])
        rollout.append(s)

    batches = maker.split_rollout_samples(rollout)

    assert batches[0].index == [0, 2, 4]
    assert batches[1].index == [1, 3]

    # Check rewards preserved per batch
    rewards0 = batches[0].rewards[:, 0].tolist()
    rewards1 = batches[1].rewards[:, 0].tolist()
    assert rewards0 == [0.0, 2.0, 4.0]
    assert rewards1 == [1.0, 3.0]


def test_static_full_microbatch_16_items_rewards_and_meta_preserved():
    # Config per request
    args = DummyArgs(
        use_dynamic_batch=False,
        micro_rollout_batch_size=16,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        ring_attn_size=1,
        ds_tensor_parallel_size=1,
        rollout_max_tokens_per_gpu=4096,
    )
    # n_samples_per_prompt affects downstream grouping; set to 4 as requested
    args.n_samples_per_prompt = 4

    strategy = DummyStrategy(args)
    tokenizer = DummyTokenizer(pad_token_id=0)
    maker = RemoteExperienceMaker(
        actor_model_group=None,
        critic_model_group=None,
        reward_model_group=None,
        initial_model_group=None,
        kl_controller=types.SimpleNamespace(value=0.0),
        strategy=strategy,
        tokenizer=tokenizer,
    )

    # Build 16 experiences with rewards and ONLY 4 distinct prompts (4 samples per prompt)
    # Use a deterministic, non-sorted interleaved order to avoid grouping by prompt
    prompt_ids = [0, 2, 1, 3] * 4  # length 16, not grouped by prompt
    prompts_seq = [f"p{pid}" for pid in prompt_ids]

    rollout = []
    for i, p in enumerate(prompts_seq):
        s = make_sample(total_length=5)  # constant length to simplify
        s.rewards = torch.tensor([[float(i)]])  # reward equals global position
        s.prompts = [p]
        s.labels = [f"l{i}"]
        rollout.append(s)

    # Assert only 4 unique prompts are used and input order is not grouped by prompt
    in_prompts = [s.prompts[0] for s in rollout]
    assert len(set(in_prompts)) == 4
    # Not grouped: there should be many transitions between adjacent prompts
    transitions = sum(1 for a, b in zip(in_prompts, in_prompts[1:]) if a != b)
    assert transitions >= 8

    batches = maker.split_rollout_samples(rollout)

    # With batch size 16 and 16 items, expect a single concatenated batch
    assert len(batches) == 1
    batch = batches[0]

    # Indices should be 0..15
    assert batch.index == list(range(16))

    # Rewards preserved in order
    assert batch.rewards[:, 0].tolist() == [float(i) for i in range(16)]

    # Prompts and labels concatenated and ordered exactly as input (not grouped)
    assert batch.prompts == prompts_seq
    assert batch.labels == [f"l{i}" for i in range(16)]

    # Info total_length stacked per item
    assert "total_length" in batch.info
    assert batch.info["total_length"].shape[0] == 16

    # For each resulting label l{i}, ensure reward equals i
    for idx, label in enumerate(batch.labels):
        assert label.startswith("l")
        i = int(label[1:])
        assert float(batch.rewards[idx, 0].item()) == float(i)
