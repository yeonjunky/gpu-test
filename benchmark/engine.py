"""Builds a vLLM LLM() instance from a configs/run_matrix.yaml (model_entry,
run_entry) pair. GPU-only -- vllm is not installed/importable on the local
authoring machine, so this module is never exercised locally; it is the
first thing spike_tests/ must validate on the remote H100 box.

Quantization path: vLLM's in-flight bnb quantization driven by
hf_overrides={"quantization_config": bnb_args} does not work -- it crashes
with a weight-shape AssertionError in vLLM's weight_loader, reproduced across
every vLLM version from 0.9.2 to 0.25.1 (see docs/spike_test_error_report.md).
Instead, bnb rungs are pre-quantized offline with transformers +
BitsAndBytesConfig and saved to a local cache dir; vLLM then loads that cache
dir directly with no quantization=/hf_overrides at all, since vLLM correctly
infers quantization from a checkpoint whose config.json and weight tensors
are already genuinely quantized (the officially documented "pre-quantized
checkpoint" pattern). Note this can roughly double a model's on-disk
footprint (full-precision source + quantized cache) -- see docs/RUNBOOK.md's
disk-space guidance.

awq/gptq/fp8 quant_methods (added for configs/ablation_matrix.yaml) need no
pre-quantize-and-cache step at all, unlike bitsandbytes: awq_marlin/gptq_marlin
only *load* a checkpoint whose config.json already declares quant_method
"awq"/"gptq" (vLLM 0.25.1 has no on-the-fly AWQ/GPTQ quantizer), so these use
an already-quantized community checkpoint via hf_repo_override; fp8 is vLLM's
own on-the-fly dynamic W8A8 quantizer applied directly to a full-precision
checkpoint. Important: quantization="fp8" is a *complete alternative*
weight+activation scheme, not a layer stacked on top of bnb/awq/gptq -- there
is no "weight-quant X + activation-quant" combo in this tooling (see
docs/Updates.md). kv_cache_dtype, in contrast, genuinely is an independent
axis (a separate vLLM CacheConfig) that layers on top of any of the above.
"""
import gc
import json
from pathlib import Path
from typing import Any

DEFAULT_QUANTIZED_CACHE_DIR = "/tmp/quantized_models"


def _quantize_and_cache(model_entry: dict, run_entry: dict, cache_dir: str) -> str:
    """Quantizes model_entry's checkpoint with run_entry['bnb_args'] via
    transformers + BitsAndBytesConfig and saves the result locally. Skips
    re-quantizing if the cache dir already has a saved checkpoint. Returns
    the local path to hand to vLLM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dst_dir = Path(cache_dir) / model_entry["id"] / run_entry["quant_level"]
    if (dst_dir / "config.json").exists():
        return str(dst_dir)

    trust_remote_code = model_entry.get("trust_remote_code", False)
    bnb_config = BitsAndBytesConfig(**run_entry["bnb_args"])

    model = AutoModelForCausalLM.from_pretrained(
        model_entry["hf_repo"],
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_entry["hf_repo"], trust_remote_code=trust_remote_code)

    dst_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst_dir)
    tokenizer.save_pretrained(dst_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return str(dst_dir)


def _build_base_kwargs(model_entry: dict, run_entry: dict, defaults: dict) -> dict:
    """Assembles the vLLM LLM() kwargs common to every quant_method (dtype,
    parallelism, context length, memory budget, and any per-model
    vllm_extra_args), before quantization-specific kwargs are added. Pure
    dict-in/dict-out -- no vllm/torch/transformers import required."""
    kwargs: dict[str, Any] = {
        "dtype": run_entry["dtype"],
        "tensor_parallel_size": defaults.get("tensor_parallel_size", 1),
        "max_model_len": model_entry.get("max_model_len", defaults["max_model_len"]),
        "gpu_memory_utilization": model_entry.get("gpu_memory_utilization", defaults["gpu_memory_utilization"]),
        "trust_remote_code": model_entry.get("trust_remote_code", defaults.get("trust_remote_code", False)),
        "seed": defaults.get("seed", 42),
        "kv_cache_dtype": run_entry.get("kv_cache_dtype", "auto"),
    }
    extra = model_entry.get("vllm_extra_args") or {}
    for k, v in extra.items():
        kwargs[k] = json.loads(v) if isinstance(v, str) and v.strip().startswith(("{", "[")) else v
    return kwargs


def build_llm(model_entry: dict, run_entry: dict, defaults: dict, quantized_cache_dir: str = DEFAULT_QUANTIZED_CACHE_DIR):
    from vllm import LLM

    kwargs = _build_base_kwargs(model_entry, run_entry, defaults)
    quant_method = run_entry["quant_method"]

    if quant_method == "bitsandbytes":
        kwargs["model"] = _quantize_and_cache(model_entry, run_entry, quantized_cache_dir)
    elif quant_method == "none":
        kwargs["model"] = run_entry.get("hf_repo_override") or model_entry["hf_repo"]
    elif quant_method in ("awq", "gptq", "fp8"):
        # Community pre-quantized checkpoint (awq/gptq) or the base checkpoint
        # (fp8, vLLM quantizes online) -- no pre-quantize-and-cache step needed
        # for any of these three, unlike bitsandbytes.
        kwargs["model"] = run_entry.get("hf_repo_override") or model_entry["hf_repo"]
        kwargs["quantization"] = run_entry["vllm_quantization"]
    else:
        raise ValueError(f"Unknown quant_method: {quant_method}")

    return LLM(**kwargs)
