import json

import pytest
import torch
from safetensors.torch import load_file as load_safetensors

from openrlhf.trainer.ray.vllm_engine import (
    load_lora_from_payload_local,
    unload_lora_adapter_local,
    adapter_id_from_name,
)


class FakeLLMEngine:
    def __init__(self):
        self.loaded_ids = set()
        self.last_request = None

    def add_lora(self, lora_request):
        # Validate files exist and are readable while temp dir is alive
        self.last_request = lora_request
        cfg_path = f"{lora_request.lora_path}/adapter_config.json"
        st_path = f"{lora_request.lora_path}/adapter_model.safetensors"
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            assert isinstance(cfg, dict)
        tensors = load_safetensors(st_path)
        assert isinstance(tensors, dict) and len(tensors) > 0
        # Mark as loaded
        self.loaded_ids.add(int(lora_request.lora_int_id))
        return True

    def remove_lora(self, lora_id: int):
        if int(lora_id) in self.loaded_ids:
            self.loaded_ids.remove(int(lora_id))
            return True
        return False

    def list_loras(self):
        return set(self.loaded_ids)


def make_payload(adapter_name: str = "test-adapter"):
    cfg = {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.0}
    tensors = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(4, 8),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(8, 4),
    }
    return {"adapter_name": adapter_name, "config": cfg, "tensors": tensors}

def build_fake_engine_and_registry():
    engine = FakeLLMEngine()
    registry: dict[int, dict] = {}
    return engine, registry


def test_load_and_unload_roundtrip():
    engine, registry = build_fake_engine_and_registry()
    payload = make_payload("step-1")

    adapter_id = load_lora_from_payload_local(engine, payload, registry)
    assert adapter_id in engine.loaded_ids
    assert adapter_id in registry
    assert registry[adapter_id]["name"] == "step-1"
    assert "loaded_at_ts" in registry[adapter_id]

    # list adapters exposes engine state
    ids_list = list(engine.list_loras())
    assert adapter_id in ids_list

    # unload cleans up engine and registry
    ok = unload_lora_adapter_local(engine, adapter_id, registry)
    assert ok is True
    assert adapter_id not in engine.loaded_ids
    assert adapter_id not in registry


def test_adapter_id_determinism():
    id1 = adapter_id_from_name("step-42")
    id2 = adapter_id_from_name("step-42")
    id3 = adapter_id_from_name("step-43")
    assert id1 == id2
    assert id1 != id3
    assert id1 > 0 and id2 > 0 and id3 > 0


@pytest.mark.parametrize(
    "bad_payload",
    [
        {},
        {"adapter_name": "a"},
        {"adapter_name": "a", "config": {}},
        {"adapter_name": "a", "config": {}, "tensors": {}},
        {"adapter_name": "a", "config": {}, "tensors": {"k": "not-a-tensor"}},
    ],
)
def test_invalid_payload_errors(bad_payload):
    engine, registry = build_fake_engine_and_registry()
    with pytest.raises((TypeError, RuntimeError)):
        load_lora_from_payload_local(engine, bad_payload, registry)
