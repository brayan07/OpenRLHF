import os
import json
import torch
import pytest

import ray

@pytest.fixture(scope="module")
def ray_cluster():
    if ray is None:
        pytest.skip("ray not available")
    if not ray.is_initialized():
        ray.init(local_mode=True, runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "false"}})
    yield
    if ray.is_initialized():
        ray.shutdown()


def make_sample_experience(seq_len: int, prompt: str = "p", label: str = "l"):
    from openrlhf.trainer.ppo_utils.experience_maker import Experience

    # Build minimal Experience with variable sequence length
    sequences = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(sequences)
    action_mask = torch.zeros_like(sequences)
    # mark last half as actions
    action_mask[:, seq_len // 2 :] = 1
    info = {
        "response_length": torch.tensor([seq_len // 2], dtype=torch.long),
        "total_length": torch.tensor([seq_len], dtype=torch.long),
    }
    return Experience(
        index=[0],
        sequences=sequences,
        attention_mask=attention_mask,
        action_mask=action_mask,
        prompts=[prompt],
        labels=[label],
        info=info,
    )


def test_experience_cpu_detach_and_serializable_dict():
    from openrlhf.trainer.ppo_utils.experience_maker import Experience

    e = make_sample_experience(7)
    # add a tensor with grad to ensure detach clears grad_fn
    t = torch.randn(3, requires_grad=True)
    e.values = t + 1  # create grad_fn

    e.to_cpu_detached()
    d = e.to_serializable_dict()

    # Ensure tensors are on CPU and detached
    assert isinstance(d["sequences"], torch.Tensor)
    assert d["sequences"].device.type == "cpu"
    assert d["values"].grad_fn is None
    assert all(isinstance(v, torch.Tensor) and v.device.type == "cpu" for v in d["info"].values())


def test_disk_logger_writes_lists_without_rebatching(tmp_path, ray_cluster):
    # Import inside test to follow TDD structure; the module will be created to satisfy this test
    from openrlhf.utils.experience_logger import ExperienceDiskLogger

    # Build inputs: rollout ~5 items with varying lengths, experiences ~3 items
    rollout_samples = [make_sample_experience(l, prompt=f"p{i}", label=f"l{i}") for i, l in enumerate([5, 7, 3, 9, 6])]
    experiences = [make_sample_experience(l, prompt=f"ep{i}", label=f"el{i}") for i, l in enumerate([11, 4, 8])]

    # Convert to serializable per-item dicts without batching
    for s in rollout_samples:
        s.to_cpu_detached()
    for e in experiences:
        e.to_cpu_detached()

    rollout_records = [s.to_serializable_dict() for s in rollout_samples]
    experience_records = [e.to_serializable_dict() for e in experiences]

    logger = ExperienceDiskLogger.remote(str(tmp_path), write_jsonl=False)
    ray.get(logger.log_rollouts.remote(123, rollout_records))
    ray.get(logger.log_experiences.remote(123, experience_records))

    # Verify files exist
    r_path = tmp_path / "rollouts_step000123.pt"
    e_path = tmp_path / "experiences_step000123.pt"
    assert r_path.exists()
    assert e_path.exists()

    # Load back and ensure lengths and ordering preserved
    r_loaded = torch.load(r_path, map_location="cpu")
    e_loaded = torch.load(e_path, map_location="cpu")
    assert isinstance(r_loaded, list) and isinstance(e_loaded, list)
    assert len(r_loaded) == len(rollout_records)
    assert len(e_loaded) == len(experience_records)

    # Check a few shape properties per item to ensure no rebatching occurred
    for src, rec in zip(rollout_records, r_loaded):
        assert rec["sequences"].shape == src["sequences"].shape
        assert rec["attention_mask"].shape == src["attention_mask"].shape
        assert rec["action_mask"].shape == src["action_mask"].shape

    # Order preservation
    assert r_loaded[0]["prompts"][0] == "p0"
    assert r_loaded[1]["prompts"][0] == "p1"


def test_disk_logger_optional_jsonl(tmp_path, ray_cluster):
    from openrlhf.utils.experience_logger import ExperienceDiskLogger

    samples = [make_sample_experience(l, prompt=f"p{i}", label=f"l{i}") for i, l in enumerate([4, 6, 5])]
    for s in samples:
        s.to_cpu_detached()
    records = [s.to_serializable_dict() for s in samples]

    # Provide lightweight JSONL records manually (implementation may use Experience.to_jsonl_record later)
    jsonl_records = [
        {"prompt": r["prompts"][0], "label": r["labels"][0], "response_length": int(r["info"]["response_length"][0])}
        for r in records
    ]

    logger = ExperienceDiskLogger.remote(str(tmp_path), write_jsonl=True)
    ray.get(logger.log_rollouts.remote(5, records, jsonl_records))

    # Verify JSONL exists and has same number of lines
    jsonl_path = tmp_path / "rollouts.jsonl"
    assert jsonl_path.exists()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) >= len(jsonl_records)  # allow multiple appends across runs
    # Check the last N entries match current batch
    tail = lines[-len(jsonl_records) :]
    assert [t["prompt"] for t in tail] == [r["prompt"] for r in jsonl_records]


@pytest.mark.usefixtures("ray_cluster")
def test_load_experience_file_roundtrip(tmp_path):
    from openrlhf.utils.experience_logger import ExperienceDiskLogger, load_experience_file

    # prepare and write
    samples = [make_sample_experience(l, prompt=f"p{i}", label=f"l{i}") for i, l in enumerate([4, 6, 5])]
    for s in samples:
        s.to_cpu_detached()
    records = [s.to_serializable_dict() for s in samples]

    logger = ExperienceDiskLogger.remote(str(tmp_path), write_jsonl=False)
    ray.get(logger.log_experiences.remote(42, records))

    # load back as Experience objects
    path = tmp_path / "experiences_step000042.pt"
    loaded = load_experience_file(str(path))
    assert len(loaded) == len(records)
    # spot check: type and a few fields
    from openrlhf.trainer.ppo_utils.experience_maker import Experience
    assert isinstance(loaded[0], Experience)
    assert loaded[0].sequences.shape == records[0]["sequences"].shape
    assert loaded[1].prompts[0] == "p1"


@pytest.mark.usefixtures("ray_cluster")
def test_iter_jsonl_reads_back(tmp_path):
    from openrlhf.utils.experience_logger import ExperienceDiskLogger, iter_jsonl

    samples = [make_sample_experience(l, prompt=f"pp{i}", label=f"ll{i}") for i, l in enumerate([3, 7])]
    for s in samples:
        s.to_cpu_detached()
    records = [s.to_serializable_dict() for s in samples]

    jsonl_records = [
        {"prompt": r["prompts"][0], "label": r["labels"][0], "response_length": int(r["info"]["response_length"][0])}
        for r in records
    ]

    logger = ExperienceDiskLogger.remote(str(tmp_path), write_jsonl=True)
    ray.get(logger.log_rollouts.remote(7, records, jsonl_records))

    # read via utility and compare tail
    it = list(iter_jsonl(str(tmp_path / "rollouts.jsonl")))
    assert len(it) >= len(jsonl_records)
    tail = it[-len(jsonl_records):]
    assert [t["prompt"] for t in tail] == [r["prompt"] for r in jsonl_records]


def test_cli_has_experience_logging_flags(monkeypatch):
    """
    We expect a get_parser() function exposing CLI flags:
      --log_experience_dir, --log_experience_every, --log_experience_jsonl
    """
    import importlib
    mod = importlib.import_module("openrlhf.cli.train_ppo_ray")

    assert hasattr(mod, "get_parser"), "train_ppo_ray.py should expose get_parser() for testable parsing"

    parser = mod.get_parser()
    args = parser.parse_args([
        "--log_experience_dir", "/tmp/out",
        "--log_experience_every", "3",
        "--log_experience_jsonl",
    ])
    assert args.log_experience_dir == "/tmp/out"
    assert args.log_experience_every == 3
    assert args.log_experience_jsonl is True

    # Defaults should parse without flags
    args2 = parser.parse_args([])
    assert hasattr(args2, "log_experience_dir")
    assert hasattr(args2, "log_experience_every")
    assert hasattr(args2, "log_experience_jsonl")
