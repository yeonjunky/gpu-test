"""Builds a vLLM LLM() instance from a configs/run_matrix.yaml (model_entry,
run_entry) pair. GPU-only -- vllm is not installed/importable on the local
authoring machine, so this module is never exercised locally; it is the
first thing spike_tests/ must validate on the remote H100 box.

Primary quantization path: pass quantization="bitsandbytes" plus
hf_overrides={"quantization_config": bnb_args} so vLLM's
BitsAndBytesConfig.from_config() parses the per-rung bnb settings
(load_in_8bit/4bit, bnb_4bit_quant_type, bnb_4bit_use_double_quant) without
needing a separately pre-quantized checkpoint on disk.

Fallback path (build_llm_with_patched_config): if spike_tests/spike_test_bnb_quant_args.py
shows hf_overrides does NOT propagate bnb_4bit_use_double_quant correctly,
use this instead -- it writes a local copy of the model's config.json with
the quantization_config patched in, and points LLM(model=...) at that local
directory. Switch via the --quant-config-mode flag on run_one_combo.py.
"""
import json
import shutil
from pathlib import Path
from typing import Any


def build_llm(model_entry: dict, run_entry: dict, defaults: dict):
    from vllm import LLM  # imported lazily: only available on the GPU box

    kwargs: dict[str, Any] = {
        "model": model_entry["hf_repo"],
        "dtype": run_entry["dtype"],
        "tensor_parallel_size": defaults.get("tensor_parallel_size", 1),
        "max_model_len": model_entry.get("max_model_len", defaults["max_model_len"]),
        "gpu_memory_utilization": model_entry.get("gpu_memory_utilization", defaults["gpu_memory_utilization"]),
        "trust_remote_code": model_entry.get("trust_remote_code", defaults.get("trust_remote_code", False)),
        "seed": defaults.get("seed", 42),
    }
    extra = model_entry.get("vllm_extra_args") or {}
    for k, v in extra.items():
        kwargs[k] = json.loads(v) if isinstance(v, str) and v.strip().startswith(("{", "[")) else v

    if run_entry["quant_method"] == "bitsandbytes":
        kwargs["quantization"] = "bitsandbytes"
        kwargs["hf_overrides"] = {"quantization_config": run_entry["bnb_args"]}
    elif run_entry["quant_method"] != "none":
        raise ValueError(f"Unknown quant_method: {run_entry['quant_method']}")

    return LLM(**kwargs)


def build_llm_with_patched_config(model_entry: dict, run_entry: dict, defaults: dict, work_dir: str = "/tmp/patched_models"):
    """Fallback for when hf_overrides doesn't propagate bnb_4bit_use_double_quant
    (see module docstring). Downloads/locates the model snapshot, writes a copy
    of config.json with quantization_config patched in, and loads from that
    local path instead of the bare HF repo id."""
    from huggingface_hub import snapshot_download
    from vllm import LLM

    if run_entry["quant_method"] != "bitsandbytes":
        return build_llm(model_entry, run_entry, defaults)

    src_dir = Path(snapshot_download(repo_id=model_entry["hf_repo"]))
    dst_dir = Path(work_dir) / model_entry["id"] / run_entry["quant_level"]
    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        if item.name == "config.json":
            continue
        link_path = dst_dir / item.name
        if not link_path.exists():
            link_path.symlink_to(item)

    config = json.loads((src_dir / "config.json").read_text())
    config["quantization_config"] = run_entry["bnb_args"]
    (dst_dir / "config.json").write_text(json.dumps(config, indent=2))

    kwargs = {
        "model": str(dst_dir),
        "dtype": run_entry["dtype"],
        "tensor_parallel_size": defaults.get("tensor_parallel_size", 1),
        "max_model_len": model_entry.get("max_model_len", defaults["max_model_len"]),
        "gpu_memory_utilization": model_entry.get("gpu_memory_utilization", defaults["gpu_memory_utilization"]),
        "trust_remote_code": model_entry.get("trust_remote_code", defaults.get("trust_remote_code", False)),
        "seed": defaults.get("seed", 42),
        "quantization": "bitsandbytes",
    }
    return LLM(**kwargs)
