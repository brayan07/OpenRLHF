import os
import time

import pytest
import torch
import ray

try:
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    LoraConfig = None
    get_peft_model = None
    get_peft_model_state_dict = None

from openrlhf.trainer.ray.vllm_engine import (
    create_vllm_engines,
    batch_vllm_engine_call,
    adapter_id_from_name,
)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for e2e vLLM tests")
@pytest.mark.skipif(AutoModelForCausalLM is None, reason="transformers/peft are required for e2e tests")
async def test_multi_engine_load_unload_and_swap(tmp_path):
    # Model selection: allow override via env
    pretrain = os.environ.get("OPENRLHF_TEST_MODEL", "facebook/opt-125m")
    print(f"[E2E] Starting test with model={pretrain}", flush=True)
    print(
        f"[E2E] CUDA available={torch.cuda.is_available()} count={torch.cuda.device_count()}",
        flush=True,
    )

    # Build LoRA payload on CPU matching OPT attention modules
    print("[E2E] Loading base HF model...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrain,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    )
    print("[E2E] Base model loaded", flush=True)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none")
    print("[E2E] Wrapping with PEFT LoRA...", flush=True)
    peft_model = get_peft_model(base_model, lora_cfg)
    print("[E2E] PEFT model ready", flush=True)

    print("[E2E] Exporting LoRA state dict...", flush=True)
    state = get_peft_model_state_dict(peft_model)  # dict[str, Tensor]
    print(f"[E2E] Exported {len(state)} tensors", flush=True)
    payload_a = {
        "adapter_name": "step-1",
        "config": lora_cfg.to_dict(),
        "tensors": {k: v.detach().to("cpu") for k, v in state.items()},
    }

    # Slightly modify weights for a second adapter (same shapes)
    payload_b = {
        "adapter_name": "step-2",
        "config": lora_cfg.to_dict(),
        "tensors": {k: (v.detach().to("cpu") + 0) for k, v in state.items()},
        "tensors": {k: (v.detach().to("cpu") + 0) for k, v in state.items()},
    }

    # Initialize Ray and vLLM engines
    if not ray.is_initialized():
        print("[E2E] Initializing Ray...", flush=True)
        gpu_count = torch.cuda.device_count()
        # Ensure enough CPUs for placement group (each bundle requests CPU:1)
        num_cpus = max(2, gpu_count * 2)
        print(f"[E2E] ray.init num_cpus={num_cpus}", flush=True)
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=num_cpus)
        print("[E2E] Ray initialized", flush=True)

    engines = []
    try:
        # Optional GPU info
        try:
            os.system("nvidia-smi || true")
        except Exception:
            pass
        gpu_count = torch.cuda.device_count()
        num_engines = 2 if gpu_count >= 2 else 1
        print(f"[E2E] Creating vLLM engines... (gpu_count={gpu_count}, num_engines={num_engines})", flush=True)
        engines = create_vllm_engines(
            num_engines=num_engines,
            tensor_parallel_size=1,
            pretrain=pretrain,
            seed=1234,
            full_determinism=False,
            enable_prefix_caching=True,
            enforce_eager=False,
            dtype="bfloat16",
            max_model_len=1024,
            shared_pg=None,
            gpu_memory_utilization=0.70,
            vllm_enable_sleep=False,
            # LoRA must be enabled in vLLM to accept dynamic adapters
            enable_lora=True,
            max_lora_rank=lora_cfg.r,
            lora_dtype="bfloat16",
            max_loras=8,
        )
        print("[E2E] vLLM engines ready", flush=True)

        # Load adapter A across engines
        print("[E2E] Loading adapter A (step-1) across engines...", flush=True)
        ref_a = ray.put(payload_a)
        ids_a = batch_vllm_engine_call(engines, "load_lora_from_payload", ref_a)
        print(f"[E2E] Loaded adapter A, ids={ids_a}", flush=True)
        assert len(ids_a) == num_engines
        assert all(x == adapter_id_from_name("step-1") for x in ids_a)

        # Ensure listed on each engine
        print("[E2E] Listing adapters after A load...", flush=True)
        lists = batch_vllm_engine_call(engines, "list_lora_adapters")
        print(f"[E2E] list_lora_adapters results={lists}", flush=True)
        for l in lists:
            assert adapter_id_from_name("step-1") in l

        # Swap to adapter B and ensure A is unloaded (aggressive unload)
        print("[E2E] Swapping to adapter B (step-2)...", flush=True)
        ref_b = ray.put(payload_b)
        ids_b = batch_vllm_engine_call(engines, "load_lora_from_payload", ref_b)
        print(f"[E2E] Loaded adapter B, ids={ids_b}", flush=True)
        assert len(ids_b) == num_engines
        assert all(x == adapter_id_from_name("step-2") for x in ids_b)

        # Unload A explicitly (should be no-op if aggressive unload already removed it)
        print("[E2E] Unloading adapter A (step-1)...", flush=True)
        batch_vllm_engine_call(engines, "unload_lora_adapter", adapter_id_from_name("step-1"))

        print("[E2E] Listing adapters after swap...", flush=True)
        lists2 = batch_vllm_engine_call(engines, "list_lora_adapters")
        print(f"[E2E] list_lora_adapters results after swap={lists2}", flush=True)
        for l in lists2:
            assert adapter_id_from_name("step-2") in l
            assert adapter_id_from_name("step-1") not in l

    finally:
        # Cleanup
        print("[E2E] Cleaning up...", flush=True)
        if engines:
            try:
                batch_vllm_engine_call(engines, "unload_lora_adapter", adapter_id_from_name("step-2"))
            except Exception:
                pass
        if ray.is_initialized():
            ray.shutdown()
        print("[E2E] Done", flush=True)
