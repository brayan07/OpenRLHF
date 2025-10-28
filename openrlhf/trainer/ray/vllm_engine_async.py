import asyncio
import os
import json
import tempfile
from pathlib import Path
import hashlib
import time

import torch
from safetensors.torch import save_file as save_safetensors
from vllm.lora.request import LoRARequest

import ray

from openrlhf.utils.agent import AgentExecutorBase

from .vllm_engine import BaseLLMRayActor


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        self.agent_func_path = kwargs.pop("agent_func_path")
        # Initialize super class
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        # Initialize result queue for streaming completed results
        self.result_queue = asyncio.Queue()
        self.agent_executor = None

        os.environ["VLLM_USE_V1"] = "1"
        import vllm
        from packaging import version

        assert version.parse(vllm.__version__) > version.parse("0.8.5"), "Asyn VLLM version must be greater than 0.8.5"

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()
        # Track adapter metadata locally for logging/metrics
        self._lora_registry: dict[int, dict] = {}

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    # Dynamic LoRA helpers (async)
    def _adapter_id_from_name(self, adapter_name: str) -> int:
        digest = hashlib.sha1(adapter_name.encode("utf-8")).hexdigest()
        val = int(digest[:8], 16) & 0x7FFFFFFF
        return val or 1

    async def load_lora_from_payload(self, payload_ref) -> int:
        # Resolve Ray object ref if provided
        payload = ray.get(payload_ref) if isinstance(payload_ref, ray.ObjectRef) else payload_ref
        if not isinstance(payload, dict):
            raise TypeError("LoRA payload must be a dict with keys: adapter_name, config, tensors|tensor_refs")

        adapter_name = payload.get("adapter_name") or "adapter"
        adapter_id = self._adapter_id_from_name(adapter_name)
        config = payload.get("config")
        tensors = payload.get("tensors")
        tensor_refs = payload.get("tensor_refs")

        if not isinstance(config, dict):
            raise RuntimeError("Invalid LoRA payload: missing or malformed 'config'.")
        if tensors is None and tensor_refs is None:
            raise RuntimeError("Invalid LoRA payload: expected 'tensors' or 'tensor_refs'.")
        if isinstance(tensors, dict) and len(tensors) == 0:
            raise RuntimeError("Invalid LoRA payload: 'tensors' must be a non-empty dict.")
        if isinstance(tensor_refs, dict) and len(tensor_refs) == 0:
            raise RuntimeError("Invalid LoRA payload: 'tensor_refs' must be a non-empty dict.")

        def _json_safe(o):
            if isinstance(o, set):
                return list(o)
            if isinstance(o, tuple):
                return list(o)
            if isinstance(o, dict):
                return {k: _json_safe(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_json_safe(v) for v in o]
            try:
                json.dumps(o)
                return o
            except TypeError:
                return str(o)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "adapter_config.json").write_text(json.dumps(_json_safe(config)))

            cpu_tensors = {}
            if isinstance(tensors, dict):
                for name, t in tensors.items():
                    if not isinstance(t, torch.Tensor):
                        raise RuntimeError(f"LoRA tensor '{name}' is not a torch.Tensor")
                    cpu_tensors[name] = t.detach().to("cpu")
            elif isinstance(tensor_refs, dict):
                for name, ref in tensor_refs.items():
                    if not isinstance(ref, ray.ObjectRef):
                        raise RuntimeError(
                            f"LoRA tensor ref '{name}' has invalid type {type(ref)}; expected ray.ObjectRef"
                        )
                    try:
                        t = ray.get(ref)
                    except Exception as e:
                        raise RuntimeError(f"Failed to resolve tensor ref '{name}': {e}")
                    if not isinstance(t, torch.Tensor):
                        raise RuntimeError(f"LoRA tensor ref '{name}' did not resolve to torch.Tensor")
                    cpu_tensors[name] = t.detach().to("cpu")
            else:
                raise RuntimeError("Invalid LoRA payload: tensors must be dict or provide tensor_refs dict.")

            if not cpu_tensors:
                raise RuntimeError("Invalid LoRA payload: no tensors to save.")

            save_safetensors(cpu_tensors, str(tmp_path / "adapter_model.safetensors"))

            req = LoRARequest(lora_name=adapter_name, lora_int_id=adapter_id, lora_path=str(tmp_path))
            ok = await self.llm.add_lora(req)
            # v0 returns None (no failure), v1 returns bool
            if ok is False:
                raise RuntimeError(f"vLLM failed to add LoRA adapter '{adapter_name}' (id={adapter_id}).")

        # Record metadata for logging/metrics
        self._lora_registry[adapter_id] = {
            "name": adapter_name,
            "loaded_at_ts": time.time(),
        }

        return adapter_id

    async def unload_lora_adapter(self, adapter_id: int) -> bool:
        if not isinstance(adapter_id, int) or adapter_id <= 0:
            raise ValueError("adapter_id must be a positive int")
        ok = await self.llm.remove_lora(adapter_id)
        if ok is not False:
            self._lora_registry.pop(adapter_id, None)
        return bool(ok) if isinstance(ok, bool) else True

    async def list_lora_adapters(self) -> list[int]:
        adapters = await self.llm.list_loras()
        return list(adapters)

    async def get_lora_registry(self) -> dict[int, dict]:
        return dict(self._lora_registry)

    async def add_requests(self, sampling_params, prompts, labels, max_length, hf_tokenizer=None, max_steps=10000):
        """
        Process requests from rank0 and generate responses with multiple agent interactions.
        Each prompt will go through multiple steps of interaction using the AgentExecutor.
        Results are streamed back as each agent completes its execution.

        Args:
            sampling_params: Parameters for sampling
            prompts: List of prompts to process
            labels: List of labels corresponding to prompts
            max_steps: Maximum number of interaction steps
        """

        # Create AgentExecutor instance
        if self.agent_executor is None:
            assert self.agent_func_path.endswith(".py"), "Agent path must be a Python file"
            import importlib.util

            spec = importlib.util.spec_from_file_location("agent_module", self.agent_func_path)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)

            # Load AgentExecutor class instead of step function
            assert hasattr(agent_module, "AgentExecutor"), "Agent module must contain AgentExecutor class"
            self.agent_executor_cls = agent_module.AgentExecutor
            assert issubclass(
                self.agent_executor_cls, AgentExecutorBase
            ), "AgentExecutor must inherit from AgentExecutorBase"

            self.agent_executor = self.agent_executor_cls(
                max_steps=max_steps,
                max_length=max_length,
                llm_engine=self.llm,
                hf_tokenizer=hf_tokenizer,
                result_queue=self.result_queue,
            )

        # Create and start tasks for all agent executions with controlled concurrency
        import copy

        tasks = []
        for prompt, label in zip(prompts, labels):
            tasks.append(self.agent_executor.execute(prompt, label, copy.deepcopy(sampling_params)))

        # Run the async code using the class's event loop
        await asyncio.gather(*tasks)

    async def get_responses(self):
        """
        Synchronously get all completed agent results from the queue.
        Waits for all tasks to complete before returning results.
        Returns: List of all completed agent results.
        """
        # Get all results from the queue
        results = []
        while not self.result_queue.empty():
            try:
                results.append(await self.result_queue.get())
            except asyncio.QueueEmpty:
                break
        return results
