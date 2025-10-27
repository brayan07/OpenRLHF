import os
import queue
import json
import tempfile
from pathlib import Path
from typing import Any, List
import hashlib

import torch
from safetensors.torch import save_file as save_safetensors
from vllm.lora.request import LoRARequest
import time

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.utils.logging_utils import init_logger

from .utils import get_bundle_indices, ray_noset_visible_devices

logger = init_logger(__name__)


def adapter_id_from_name(adapter_name: str) -> int:
    """Stable positive 31-bit ID derived from adapter name."""
    digest = hashlib.sha1(adapter_name.encode("utf-8")).hexdigest()
    val = int(digest[:8], 16) & 0x7FFFFFFF
    return val or 1


def load_lora_from_payload_local(llm_engine, payload: dict, registry: dict | None = None) -> int:
    """Core LoRA loader that operates on a provided llm_engine.

    - Writes adapter payload to a temp dir
    - Calls llm_engine.add_lora(LoRARequest)
    - Updates optional registry with metadata
    Returns adapter_id
    """
    if not isinstance(payload, dict):
        raise TypeError("LoRA payload must be a dict with keys: adapter_name, config, tensors")

    adapter_name = payload.get("adapter_name") or "adapter"
    adapter_id = adapter_id_from_name(adapter_name)
    config = payload.get("config")
    tensors = payload.get("tensors")

    if not isinstance(config, dict) or not isinstance(tensors, dict) or not tensors:
        raise RuntimeError("Invalid LoRA payload: missing or malformed 'config'/'tensors'.")

    def _json_safe(o):
        if isinstance(o, set):
            return list(o)
        if isinstance(o, tuple):
            return list(o)
        if isinstance(o, dict):
            return {k: _json_safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [ _json_safe(v) for v in o ]
        try:
            json.dumps(o)
            return o
        except TypeError:
            return str(o)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "adapter_config.json").write_text(json.dumps(_json_safe(config)))

        cpu_tensors = {}
        for name, t in tensors.items():
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(f"LoRA tensor '{name}' is not a torch.Tensor")
            cpu_tensors[name] = t.detach().to("cpu")

        save_safetensors(cpu_tensors, str(tmp_path / "adapter_model.safetensors"))

        req = LoRARequest(lora_name=adapter_name, lora_int_id=adapter_id, lora_path=str(tmp_path))
        ok = llm_engine.add_lora(req)
        if not ok:
            raise RuntimeError(f"vLLM failed to add LoRA adapter '{adapter_name}' (id={adapter_id}).")

    if registry is not None:
        registry[adapter_id] = {"name": adapter_name, "loaded_at_ts": time.time()}

    return adapter_id


def unload_lora_adapter_local(llm_engine, adapter_id: int, registry: dict | None = None) -> bool:
    if not isinstance(adapter_id, int) or adapter_id <= 0:
        raise ValueError("adapter_id must be a positive int")
    ok = llm_engine.remove_lora(adapter_id)
    if ok is not False and registry is not None:
        registry.pop(adapter_id, None)
    return bool(ok) if isinstance(ok, bool) else True


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


class BaseLLMRayActor:
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        kwargs.pop("agent_func_path", None)
        noset_visible_devices = ray_noset_visible_devices()
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.requests = {}
        self.response_queues = queue.Queue()

        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.kwargs = kwargs

        import vllm
        from packaging import version

        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


@ray.remote
class LLMRayActor(BaseLLMRayActor):
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        import vllm

        self.llm = vllm.LLM(*args, **self.kwargs)
        # Track adapter metadata locally for logging/metrics
        self._lora_registry: dict[int, dict] = {}

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def add_requests(self, sampling_params, prompt_token_ids):
        """
        Process requests from rank0 and generate responses.
        Since only rank0 will send requests, we don't need to track actor ranks.
        """
        from vllm.inputs import TokensPrompt

        requests = [TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids]
        responses = self.llm.generate(prompts=requests, sampling_params=sampling_params)
        self.response_queues.put(responses)

    def get_responses(self):
        """
        Return the responses for the actor with the given rank
        """
        return self.response_queues.get()

    # Dynamic LoRA helpers
    def _adapter_id_from_name(self, adapter_name: str) -> int:
        return adapter_id_from_name(adapter_name)

    def load_lora_from_payload(self, payload_ref) -> int:
        """Load a LoRA adapter into vLLM from a Ray payload.

        Payload schema: {"adapter_name": str, "config": dict, "tensors": dict[str, torch.Tensor]}
        Returns: adapter_id (int)
        """
        payload = ray.get(payload_ref) if isinstance(payload_ref, ray.ObjectRef) else payload_ref
        return load_lora_from_payload_local(self.llm.llm_engine, payload, registry=self._lora_registry)

    def unload_lora_adapter(self, adapter_id: int) -> bool:
        return unload_lora_adapter_local(self.llm.llm_engine, adapter_id, registry=self._lora_registry)

    def list_lora_adapters(self) -> list[int]:
        # vLLM returns a set[int]
        return list(self.llm.llm_engine.list_loras())

    def get_lora_registry(self) -> dict[int, dict]:
        # Return a shallow copy to avoid external mutation
        return dict(self._lora_registry)


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    llm_actor_cls=LLMRayActor,
    logprobs_mode=None,
    agent_func_path=None,
    # LoRA-related options
    dtype: str = "bfloat16",
    enable_lora: bool = False,
    max_lora_rank: int | None = None,
    lora_dtype: str | None = None,
    max_loras: int | None = None,
    enable_lora_bias: bool | None = None,
):
    import vllm
    from packaging import version

    assert version.parse(vllm.__version__) > version.parse("0.8.2"), "OpenRLHF only supports vllm > 0.8.2"

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.5

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        additional_kwargs = {}
        if logprobs_mode:
            additional_kwargs["logprobs_mode"] = logprobs_mode
            additional_kwargs["max_logprobs"] = 1
            assert version.parse(vllm.__version__) > version.parse(
                "0.10.0"
            ), "vLLM > 0.10.0 is required for logprobs_mode"

        vllm_engines.append(
            llm_actor_cls.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_extension_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype=dtype,
                trust_remote_code=True,
                full_determinism=full_determinism,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.5 if use_hybrid_engine else 1,
                enable_sleep_mode=vllm_enable_sleep,
                agent_func_path=agent_func_path,
                # LoRA
                enable_lora=enable_lora,
                **({"max_lora_rank": max_lora_rank} if max_lora_rank is not None else {}),
                **({"lora_dtype": lora_dtype} if lora_dtype is not None else {}),
                **({"max_loras": max_loras} if max_loras is not None else {}),
                **({"enable_lora_bias": enable_lora_bias} if enable_lora_bias is not None else {}),
                **additional_kwargs,
            )
        )

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep")

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
