import json

import pytest
import torch
from safetensors.torch import load_file as load_safetensors
import ray

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


def test_load_with_tensor_refs_roundtrip():
    # Ensure Ray is running
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1)
    engine, registry = build_fake_engine_and_registry()

    adapter_name = "step-refs"
    cfg = {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.0}
    # Put tensors individually into Ray object store
    refs = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": ray.put(torch.zeros(4, 8)),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": ray.put(torch.zeros(8, 4)),
    }
    payload = {"adapter_name": adapter_name, "config": cfg, "tensor_refs": refs}

    adapter_id = load_lora_from_payload_local(engine, payload, registry)
    assert adapter_id in engine.loaded_ids
    assert adapter_id in registry and registry[adapter_id]["name"] == adapter_name

    # Unload cleans up engine and registry
    ok = unload_lora_adapter_local(engine, adapter_id, registry)
    assert ok is True
    assert adapter_id not in engine.loaded_ids
    assert adapter_id not in registry
    # Shutdown Ray if we started it
    if ray.is_initialized():
        ray.shutdown()


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
        {"adapter_name": "a", "config": {}, "tensor_refs": {}},
        {"adapter_name": "a", "config": {}, "tensor_refs": {"k": "not-a-ref"}},
    ],
)
def test_invalid_payload_errors(bad_payload):
    engine, registry = build_fake_engine_and_registry()
    with pytest.raises((TypeError, RuntimeError)):
        load_lora_from_payload_local(engine, bad_payload, registry)


def test_sync_engine_auto_lora_injection():
    """Test that LLMRayActor automatically injects active LoRA into generate calls."""
    from unittest.mock import Mock, MagicMock
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    
    # Create a simple class that mimics the actor's add_requests behavior
    class MockActor:
        def __init__(self):
            self.llm = Mock()
            self.llm.generate = Mock(return_value=[])
            self._active_lora_request = None
            self.response_queues = MagicMock()
        
        def add_requests(self, sampling_params, prompt_token_ids):
            """Mimics LLMRayActor.add_requests with auto LoRA injection."""
            from vllm.inputs import TokensPrompt
            requests = [TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids]
            responses = self.llm.generate(
                prompts=requests, 
                sampling_params=sampling_params,
                lora_request=self._active_lora_request
            )
            self.response_queues.put(responses)
    
    actor = MockActor()
    
    # Initially, no LoRA should be injected
    sampling_params = SamplingParams(temperature=1.0)
    actor.add_requests(sampling_params, [[1, 2, 3]])
    
    # Verify generate was called with lora_request=None
    assert actor.llm.generate.called
    call_kwargs = actor.llm.generate.call_args[1]
    assert call_kwargs.get("lora_request") is None
    
    # Simulate loading a LoRA adapter - set active LoRA
    # Note: lora_path cannot be None in newer vLLM versions
    actor._active_lora_request = LoRARequest(
        lora_name="step-1",
        lora_int_id=123456,
        lora_path="/tmp/dummy"  # Dummy path for testing
    )
    
    # Reset mock
    actor.llm.generate.reset_mock()
    
    # Now generate should automatically inject the LoRA
    actor.add_requests(sampling_params, [[4, 5, 6]])
    
    # Verify generate was called with the active LoRA request
    assert actor.llm.generate.called
    call_kwargs = actor.llm.generate.call_args[1]
    assert call_kwargs.get("lora_request") is not None
    assert call_kwargs["lora_request"].lora_int_id == 123456
    assert call_kwargs["lora_request"].lora_name == "step-1"


@pytest.mark.asyncio
async def test_async_engine_wrapper_auto_lora_injection():
    """Test that AsyncEngineLoRAWrapper automatically injects active LoRA."""
    from unittest.mock import AsyncMock, Mock
    from openrlhf.trainer.ray.vllm_engine_async import AsyncEngineLoRAWrapper
    from vllm.lora.request import LoRARequest
    
    # Create a mock async engine
    mock_engine = Mock()
    
    # Mock the generate method as an async generator
    async def mock_generate(prompts, sampling_params, request_id, lora_request=None):
        # Store the lora_request for verification
        mock_engine.last_lora_request = lora_request
        yield Mock(outputs=[Mock(token_ids=[1, 2, 3])])
    
    mock_engine.generate = mock_generate
    
    # Create active LoRA request (lora_path cannot be None)
    active_lora = LoRARequest(lora_name="step-1", lora_int_id=123, lora_path="/tmp/dummy")
    
    # Create wrapper with getter that returns active LoRA
    wrapper = AsyncEngineLoRAWrapper(mock_engine, lambda: active_lora)
    
    # Call generate without explicit lora_request
    outputs = []
    async for output in wrapper.generate(
        prompts=Mock(),
        sampling_params=Mock(),
        request_id="test-123"
    ):
        outputs.append(output)
    
    # Verify the active LoRA was injected
    assert mock_engine.last_lora_request is not None
    assert mock_engine.last_lora_request.lora_int_id == 123
    assert mock_engine.last_lora_request.lora_name == "step-1"
    assert len(outputs) == 1
    
    # Test with explicit lora_request (should override)
    override_lora = LoRARequest(lora_name="override", lora_int_id=456, lora_path="/tmp/override")
    outputs = []
    async for output in wrapper.generate(
        prompts=Mock(),
        sampling_params=Mock(),
        request_id="test-456",
        lora_request=override_lora
    ):
        outputs.append(output)
    
    # Verify the explicit LoRA was used
    assert mock_engine.last_lora_request is not None
    assert mock_engine.last_lora_request.lora_int_id == 456
    assert mock_engine.last_lora_request.lora_name == "override"


@pytest.mark.asyncio
async def test_async_engine_wrapper_forwards_attributes():
    """Test that AsyncEngineLoRAWrapper forwards non-generate methods correctly."""
    from unittest.mock import Mock, AsyncMock
    from openrlhf.trainer.ray.vllm_engine_async import AsyncEngineLoRAWrapper
    
    # Create a mock engine with various methods
    mock_engine = Mock()
    mock_engine.add_lora = AsyncMock(return_value=True)
    mock_engine.remove_lora = AsyncMock(return_value=True)
    mock_engine.list_loras = AsyncMock(return_value={1, 2, 3})
    mock_engine.some_property = "test_value"
    
    # Create wrapper
    wrapper = AsyncEngineLoRAWrapper(mock_engine, lambda: None)
    
    # Test that methods are forwarded
    result = await wrapper.add_lora(Mock())
    assert result is True
    assert mock_engine.add_lora.called
    
    result = await wrapper.remove_lora(1)
    assert result is True
    assert mock_engine.remove_lora.called
    
    result = await wrapper.list_loras()
    assert result == {1, 2, 3}
    
    # Test that properties are forwarded
    assert wrapper.some_property == "test_value"
