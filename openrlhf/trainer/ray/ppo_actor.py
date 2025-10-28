import math
import os
import socket
from abc import ABC
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import deepspeed
import ray
import torch
import torch.distributed
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.models import Actor, PolicyLoss
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_util import stateless_init_process_group, torch_dist_barrier_and_cuda_sync
from openrlhf.utils.logging_utils import init_logger

from peft import get_peft_model_state_dict

from ..ppo_utils import NaiveReplayBuffer

logger = init_logger(__name__)

from .launcher import BaseModelActor
from .utils import get_physical_gpu_id
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call


class ActorPPOTrainer(ABC):
    def __init__(
        self,
        strategy,
        actor: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        ema_beta: float = 0.992,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        tokenizer=None,
        dataloader_pin_memory: bool = True,
        vllm_engines: List = None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.generate_kwargs = kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.micro_train_batch_size = micro_train_batch_size
        self.ema_beta = ema_beta

        self.actor = actor
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engines = vllm_engines
        self.max_epochs = self.args.max_epochs

        # Dynamic LoRA flags/state
        self._use_dynamic_lora = bool(getattr(self.args, "vllm_dynamic_lora", False) and getattr(self.args, "lora_rank", 0) > 0)
        self._last_lora_adapter_id: Optional[int] = None
        self._lora_swap_seq: int = 0

        self.actor_loss_fn = PolicyLoss(
            clip_eps_low=self.args.eps_clip_low_high[0],
            clip_eps_high=self.args.eps_clip_low_high[1],
            dual_clip=self.args.dual_clip,
            policy_loss_type=self.args.policy_loss_type,
            enable_vllm_is_correction=self.args.enable_vllm_is_correction,
            vllm_is_truncated_threshold=(
                self.args.vllm_is_truncated_threshold if self.args.enable_vllm_is_correction else None
            ),
        )

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args, "packing_samples", False),
            self.args.use_dynamic_batch,
        )

        # Init torch group for weights sync
        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.args.colocate_all_models and not self.args.async_train:
            self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and each of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = stateless_init_process_group(
                    master_address, master_port, 0, world_size, torch.cuda.current_device()
                )

            ray.get(refs)

        torch_dist_barrier_and_cuda_sync()

    def ppo_train(self, kl_ctl: float):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.use_dynamic_batch:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        not_shuffle = (
            self.strategy.ring_attn_group is not None
            or self.args.ds_tensor_parallel_size > 1
            or self.args.use_dynamic_batch
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=not not_shuffle,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for step, experience in enumerate(pbar):

                experience.to_device(device)
                status = self.training_step(experience, kl_ctl, step)
                status["kl"] *= status["response_length"]
                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]

                short_status = {
                    "act_loss": status["policy_loss"],
                    "reward": status["reward"],
                    "return": status["return"],
                    "gen_len": status["response_length"],
                    "tot_len": status["total_length"],
                    "kl": status["kl"],
                    "act_lr": status["actor_lr"],
                }

                if "entropy_loss" in status:
                    short_status["ent_loss"] = status["entropy_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience, kl_ctl: float, step: int) -> Dict[str, float]:
        self.actor.train()

        sequences = experience.sequences
        action_mask = experience.action_mask
        attention_mask = experience.attention_mask
        packed_seq_lens = None
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs

        # actor loss
        action_log_probs, output = self.actor(
            sequences,
            action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
            return_entropy=self.args.entropy_loss_coef is not None,
        )

        # loss function
        actor_loss, clip_ratio, ppo_kl, vllm_kl = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            rollout_log_probs=experience.rollout_log_probs,
        )
        experience.info["ppo_clip_ratio"] = clip_ratio.detach()
        experience.info["ppo_kl"] = ppo_kl.detach()
        if vllm_kl is not None:
            experience.info["vllm_kl"] = vllm_kl.detach()

        if self.args.use_kl_loss:
            if self.args.init_kl_coef > 0:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)
            kl_loss = masked_mean(kl, experience.action_mask)
            experience.info["kl"] = kl_loss.detach()
        else:
            kl_loss = 0

        loss = actor_loss + kl_loss * kl_ctl
        # mixtral
        if self.aux_loss:
            loss += output.aux_loss * self.args.aux_loss_coef
        # entropy loss
        if self.args.entropy_loss_coef is not None:
            entropy_loss = masked_mean(output.entropy[:, -experience.action_mask.shape[1] :], experience.action_mask)
            if self.args.entropy_loss_coef != 0:
                loss -= entropy_loss * self.args.entropy_loss_coef

        if self.args.use_dynamic_batch:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]

        self.strategy.backward(loss, self.actor, self.actor_optim)
        if self.args.use_dynamic_batch:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        else:
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            if self.args.use_dynamic_batch:
                if self.replay_buffer.dynamic_optimizer_step[step]:
                    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
            else:
                self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        # status
        status = {"policy_loss": actor_loss.detach().item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        if self.args.entropy_loss_coef is not None:
            status["entropy_loss"] = entropy_loss.detach().item()

        # merge logs from info field
        for k, v in experience.info.items():
            if isinstance(v, list):
                status[k] = torch.tensor(v, dtype=torch.float).mean().item()
            elif isinstance(v, torch.Tensor):
                status[k] = v.float().mean().item()
        return status

    def _build_lora_payload(self, adapter_name: str) -> Dict[str, Any]:
        """Build LoRA payload from the local actor model on CPU.
        
        Under ZeRO-3, all ranks must call this to participate in gather collectives,
        but only rank 0 will export tensors and return a payload.
        """
        if getattr(self.strategy.args, "lora_rank", 0) <= 0:
            raise RuntimeError("LoRA is not enabled on the policy actor; cannot export adapter state.")
        torch.cuda.empty_cache()
        base_model = self.actor.model.module
        if not hasattr(base_model, "peft_config") or not base_model.peft_config:
            raise RuntimeError("Actor model does not carry PEFT configuration; expected LoRA-enabled PEFT model.")

        # Use the first adapter's config as template
        first_name = list(base_model.peft_config.keys())[0]
        adapter_config = base_model.peft_config[first_name].to_dict()
        bias_mode = getattr(base_model.peft_config[first_name], "bias", "none")
        if bias_mode == "all":
            raise RuntimeError("LoRA bias='all' is not supported for dynamic adapter export under ZeRO-3.")

        # Stream export: gather each required tensor individually and put to object store
        zero_stage = getattr(self.strategy.args, "zero_stage", 0)
        name_to_param = dict(base_model.named_parameters())
        # LoRA params for the selected adapter
        lora_param_names = [
            n for n in name_to_param.keys() if "lora_" in n and first_name in n
        ]
        # Optional bias for lora_only
        bias_names: list[str] = []
        if bias_mode == "lora_only":
            prefixes = {n.split("lora_")[0] for n in lora_param_names}
            bias_names = [p + "bias" for p in prefixes if (p + "bias") in name_to_param]

        param_names = list(dict.fromkeys(lora_param_names + bias_names))  # preserve order, de-dup
        if not param_names:
            raise RuntimeError("No LoRA parameters found to export.")

        is_rank0 = torch.distributed.get_rank() == 0
        tensor_refs: Dict[str, ray.ObjectRef] = {}
        total_params = len(param_names)
        pbar = tqdm(param_names, desc="Exporting LoRA tensors", total=total_params, disable=not is_rank0)
        for pname in pbar:
            p = name_to_param[pname]
            # All ranks participate in gather collective under ZeRO-3
            if zero_stage == 3:
                with deepspeed.zero.GatheredParameters([p], enabled=True, modifier_rank=0):
                    # Only rank 0 exports to CPU and object store
                    if is_rank0:
                        t_cpu = p.detach().to("cpu")
                        # Strip adapter name from key for vLLM compatibility
                        # e.g., "lora_A.default.weight" -> "lora_A.weight"
                        vllm_key = pname.replace(f".{first_name}.", ".")
                        tensor_refs[vllm_key] = ray.put(t_cpu)
            else:
                # No gather needed; only rank 0 exports
                if is_rank0:
                    t_cpu = p.detach().to("cpu")
                    # Strip adapter name from key for vLLM compatibility
                    vllm_key = pname.replace(f".{first_name}.", ".")
                    tensor_refs[vllm_key] = ray.put(t_cpu)

        # Only rank 0 returns the payload; other ranks return None
        if is_rank0:
            return {"adapter_name": adapter_name, "config": adapter_config, "tensor_refs": tensor_refs}
        else:
            return None

    def _broadcast_to_vllm(self):
        # Dynamic LoRA path: push adapter payload instead of dense weights
        if self._use_dynamic_lora:
            t0 = time.time()
            # All ranks must call _build_lora_payload to participate in ZeRO-3 gather collectives
            self._lora_swap_seq += 1
            step_adapter_name = f"step-{self._lora_swap_seq}"
            payload = self._build_lora_payload(step_adapter_name)
            
            # Only rank 0 orchestrates vLLM adapter swap
            if torch.distributed.get_rank() == 0:
                print("Putting lora pyload in object store")
                payload_ref = ray.put(payload)

                # Optional: reset prefix cache prior to swap (async)
                reset_refs = []
                if getattr(self.strategy.args, "enable_prefix_caching", False):
                    for engine in self.vllm_engines:
                        reset_refs.append(engine.reset_prefix_cache.remote())

                # Load new adapter across all engines
                print("Loading lora payload in vLLM")
                adapter_ids = batch_vllm_engine_call(self.vllm_engines, "load_lora_from_payload", payload_ref)
                # Use first adapter_id as canonical (all engines derive same id from name)
                print("Adapter id: ", adapter_ids)
                new_adapter_id = int(adapter_ids[0]) if isinstance(adapter_ids, list) and adapter_ids else int(adapter_ids)

                # Aggressively unload previous adapter after successful load
                if self._last_lora_adapter_id is not None:
                    print("Unloading previous adapter")
                    batch_vllm_engine_call(self.vllm_engines, "unload_lora_adapter", self._last_lora_adapter_id)

                self._last_lora_adapter_id = new_adapter_id
                # Ensure cache reset completion if scheduled
                print("Waiting for cache reset")
                if getattr(self.strategy.args, "enable_prefix_caching", False):
                    ray.get(reset_refs)
                logger.info(
                    f"Dynamic LoRA: loaded adapter='{step_adapter_name}' id={new_adapter_id} across {len(self.vllm_engines)} engines in {time.time()-t0:.3f}s"
                )

            # sync ranks
            print("Syncing ranks")
            torch_dist_barrier_and_cuda_sync()
            return

        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        count, num_params = 0, len(list(model.named_parameters()))

        def _broadcast_param(param, count, num_params):
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(param.data, 0, group_name=self._model_update_group)
                else:
                    self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
                ray.get(refs)

        def _handle_cuda_ipc(param, count, num_params):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_cuda_sync()

        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # broadcast
            if not self.use_cuda_ipc:
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _broadcast_param(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _broadcast_param(param, count, num_params)
            # CUDA IPC
            else:
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _handle_cuda_ipc(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                        _handle_cuda_ipc(param, count, num_params)

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()

    def get_current_lora_adapter_id(self) -> Optional[int]:
        """Return the currently loaded LoRA adapter ID for vLLM generation."""
        return self._last_lora_adapter_id if self._use_dynamic_lora else None


@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                attn_implementation=strategy.args.attn_implementation,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # Verify only LoRA params are trainable and report counts
        is_dist0 = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        try:
            base_model = self.strategy._unwrap_model(self.actor)
        except Exception:
            base_model = None
        if is_dist0 and base_model is not None:
            # Print PEFT summary if available
            if hasattr(base_model, "print_trainable_parameters"):
                base_model.print_trainable_parameters()

            total_params = 0
            trainable_params = 0
            offenders: list[str] = []

            # Allow-list
            modules_to_save = set(getattr(self.strategy.args, "modules_to_save", []) or [])
            adapter_names = list(getattr(base_model, "peft_config", {}).keys()) if hasattr(base_model, "peft_config") else []
            bias_mode = None
            if adapter_names:
                bias_mode = getattr(base_model.peft_config[adapter_names[0]], "bias", "none")

            def _numel(param: torch.nn.Parameter) -> int:
                # Under ZeRO-3, param.numel() can be 0 for partitioned placeholders; prefer ds_numel if present
                return int(getattr(param, "ds_numel", param.numel()))

            for name, p in base_model.named_parameters():
                n = _numel(p)
                total_params += n
                if p.requires_grad:
                    trainable_params += n
                    is_lora = ("lora_" in name) or ("lora_magnitude_vector" in name)  # DoRA vector
                    is_saved_mod = any(m in name for m in modules_to_save)
                    is_allowed_bias = (bias_mode == "lora_only" and name.endswith(".bias"))
                    if not (is_lora or is_saved_mod or is_allowed_bias):
                        offenders.append(name)

            pct = (100.0 * trainable_params / max(1, total_params))
            logger.info(
                f"Params: total={total_params:,} trainable={trainable_params:,} ({pct:.4f}%)"
            )
            if offenders:
                # Fail-fast per project guidelines
                sample = offenders[:10]
                more = " ..." if len(offenders) > 10 else ""
                raise RuntimeError(
                    f"Found non-LoRA trainable parameters: {sample}{more}. "
                    f"If intended, add them to modules_to_save or adjust LoRA config."
                )

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states["global_step"] = states["global_step"]
            self.checkpoint_states["episode"] = states["episode"]
            self.checkpoint_states["data_loader_state_dict"] = states["data_loader_state_dict"]
            self.checkpoint_states["controller_state"] = states["controller_state"]

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

        # configure Trainer
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.eps_clip,
            ema_beta=args.ema_beta,
            vllm_engines=self.vllm_engines,
        )

    def fit(self, kl_ctl: float = 0):
        """Train actor model with the replay buffer."""
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.ppo_train(kl_ctl)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        """Generates actor values."""
        device = torch.cuda.current_device()
        self.actor.eval()
        with torch.no_grad():
            action_log_probs = self.actor(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
            )
        self.actor.train()  # reset model state
        return action_log_probs.to("cpu")

    def broadcast_to_vllm(self):
        self.trainer._broadcast_to_vllm()

    def get_current_lora_adapter_id(self) -> Optional[int]:
        """Return the currently loaded LoRA adapter ID for vLLM generation."""
        return self.trainer.get_current_lora_adapter_id()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        self.trainer.replay_buffer.append(experience)

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def save_checkpoint(self, tag, client_states):
        args = self.strategy.args
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        # wait
        torch_dist_barrier_and_cuda_sync()

    def get_lora_state(self):
        """Export the current LoRA adapter weights and config for dynamic vLLM loading."""

        if getattr(self.strategy.args, "lora_rank", 0) <= 0:
            raise RuntimeError("LoRA is not enabled on the policy actor; cannot export adapter state.")

        base_model = self.strategy._unwrap_model(self.actor)

        if not hasattr(base_model, "peft_config") or not base_model.peft_config:
            raise RuntimeError("Actor model does not carry PEFT configuration; expected LoRA-enabled PEFT model.")

        adapter_names = list(base_model.peft_config.keys())
        if not adapter_names:
            raise RuntimeError("No LoRA adapters found on the policy actor model.")
        if len(adapter_names) > 1:
            logger.warning(
                "Multiple LoRA adapters detected on policy actor; exporting only the first adapter '%s'.",
                adapter_names[0],
            )

        adapter_name = adapter_names[0]
        adapter_config = base_model.peft_config[adapter_name].to_dict()
        bias_mode = getattr(base_model.peft_config[adapter_name], "bias", "none")
        if bias_mode == "all":
            raise RuntimeError("LoRA bias='all' is not supported for dynamic adapter export under ZeRO-3.")

        # Stream export: gather each required tensor individually and put to object store
        zero_stage = getattr(self.strategy.args, "zero_stage", 0)
        name_to_param = dict(base_model.named_parameters())
        lora_param_names = [n for n in name_to_param.keys() if "lora_" in n and adapter_name in n]
        bias_names: list[str] = []
        if bias_mode == "lora_only":
            prefixes = {n.split("lora_")[0] for n in lora_param_names}
            bias_names = [p + "bias" for p in prefixes if (p + "bias") in name_to_param]

        param_names = list(dict.fromkeys(lora_param_names + bias_names))
        if not param_names:
            raise RuntimeError("No LoRA parameters found to export.")

        tensor_refs: Dict[str, ray.ObjectRef] = {}
        for pname in param_names:
            p = name_to_param[pname]
            if zero_stage == 3:
                with deepspeed.zero.GatheredParameters([p], enabled=True, modifier_rank=0):
                    t_cpu = p.detach().to("cpu")
            else:
                t_cpu = p.detach().to("cpu")
            # Strip adapter name from key for vLLM compatibility
            # e.g., "lora_A.default.weight" -> "lora_A.weight"
            vllm_key = pname.replace(f".{adapter_name}.", ".")
            tensor_refs[vllm_key] = ray.put(t_cpu)

        payload = {"adapter_name": adapter_name, "config": adapter_config, "tensor_refs": tensor_refs}
        return ray.put(payload)
