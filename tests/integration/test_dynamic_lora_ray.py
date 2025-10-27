import json
import os

import pytest
import ray
import torch

from openrlhf.trainer.ray.vllm_engine import (
    load_lora_from_payload_local,
    unload_lora_adapter_local,
)


class FakeLLMEngine:
    def __init__(self):
        self.loaded_ids = set()

    def add_lora(self, lora_request):
        # Validate files exist and are readable while temp dir is alive
        cfg_path = os.path.join(lora_request.lora_path, "adapter_config.json")
        st_path = os.path.join(lora_request.lora_path, "adapter_model.safetensors")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            assert isinstance(cfg, dict)
        # Avoid importing safetensors here; existence is enough for this ray-level test
        assert os.path.exists(st_path)
        self.loaded_ids.add(int(lora_request.lora_int_id))
        return True

    def remove_lora(self, lora_id: int):
        if int(lora_id) in self.loaded_ids:
            self.loaded_ids.remove(int(lora_id))
            return True
        return False

    def list_loras(self):
        return set(self.loaded_ids)


def make_payload(adapter_name: str = "ray-step-1"):
    cfg = {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.0}
    tensors = {
        "layer.q_proj.lora_A.weight": torch.zeros(4, 8),
        "layer.q_proj.lora_B.weight": torch.zeros(8, 4),
    }
    return {"adapter_name": adapter_name, "config": cfg, "tensors": tensors}


@ray.remote
class TestAdapterLoader:
    def __init__(self):
        self.engine = FakeLLMEngine()
        self.registry: dict[int, dict] = {}

    def load_from_ref(self, payload_ref):
        # In local_mode, refs may be resolved to plain dicts
        try:
            from ray import ObjectRef  # type: ignore
            is_ref = isinstance(payload_ref, ObjectRef)
        except Exception:
            is_ref = False
        payload = ray.get(payload_ref) if is_ref else payload_ref
        return load_lora_from_payload_local(self.engine, payload, self.registry)

    def unload(self, adapter_id: int) -> bool:
        return unload_lora_adapter_local(self.engine, adapter_id, self.registry)

    def list_ids(self):
        return list(self.engine.list_loras())

    def registry_snapshot(self):
        return dict(self.registry)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ray_object_ref_path(tmp_path):
    if not ray.is_initialized():
        # Run Ray in local_mode to avoid worker import issues for test module-defined actors
        ray.init(local_mode=True, num_cpus=2, ignore_reinit_error=True)

    try:
        actor = TestAdapterLoader.options(num_cpus=1).remote()
        payload = make_payload("ray-step-1")
        payload_ref = ray.put(payload)

        adapter_id = ray.get(actor.load_from_ref.remote(payload_ref))
        ids = ray.get(actor.list_ids.remote())
        reg = ray.get(actor.registry_snapshot.remote())

        assert adapter_id in ids
        assert adapter_id in reg and reg[adapter_id]["name"] == "ray-step-1"

        ok = ray.get(actor.unload.remote(adapter_id))
        assert ok is True
        ids2 = ray.get(actor.list_ids.remote())
        assert adapter_id not in ids2
    finally:
        if ray.is_initialized():
            ray.shutdown()
