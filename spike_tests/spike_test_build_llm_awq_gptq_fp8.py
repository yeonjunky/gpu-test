"""Spike test: do the 3 new quant_method branches added to
benchmark/engine.py::build_llm for the ablation study (awq, gptq, fp8) each
actually load and generate against the real qwen2.5-32b rows in
configs/ablation_matrix.yaml? Run before trusting the ablation orchestrator
run.

awq/gptq use official pre-quantized checkpoints (Qwen/Qwen2.5-32B-Instruct-AWQ,
Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4) loaded via vLLM's quantization=
"awq_marlin"/"gptq_marlin" -- no pre-quantize-and-cache step, unlike
bitsandbytes. fp8 is vLLM's own on-the-fly dynamic W8A8 quantizer applied
directly to the base bf16 checkpoint. All three are genuinely new code paths
in build_llm() (see benchmark/engine.py's module docstring for why none of
them need a local quantized-checkpoint cache the way bitsandbytes does).

Also exercises kv_cache_dtype="auto" (the default for every row in this
pilot's Phase 1+2) through _build_base_kwargs, and -- incidentally, since it
calls build_llm() the same way run_one_combo.py does -- the same
find_entries() lookup used by the real ablation orchestrator run, so PASS/FAIL
here is representative of what the full pilot will do for these 3 combos.

Each quant_level loads in its own subprocess (clean CUDA teardown between
32B-model loads), sequentially -- never run these concurrently, each wants a
large fraction of the H100's VRAM.

Run on the remote H100 box before the ablation pilot orchestrator run:
    python spike_tests/spike_test_build_llm_awq_gptq_fp8.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `benchmark` is importable regardless of cwd

import yaml

MODEL_ID = "qwen2.5-32b"
QUANT_LEVELS = ["awq_kv_auto", "gptq_kv_auto", "fp8_online_kv_auto"]
ABLATION_MATRIX_PATH = REPO_ROOT / "configs" / "ablation_matrix.yaml"


def _worker(quant_level: str) -> None:
    from vllm import SamplingParams

    from benchmark.engine import build_llm
    from benchmark.run_one_combo import find_entries

    with open(ABLATION_MATRIX_PATH) as f:
        run_matrix = yaml.safe_load(f)
    model_entry, run_entry = find_entries(run_matrix, MODEL_ID, quant_level)
    defaults = run_matrix["defaults"]

    print(f"Loading {model_entry['hf_repo']} ({MODEL_ID}/{quant_level}, quant_method={run_entry['quant_method']}) via build_llm ...")
    llm = build_llm(model_entry, run_entry, defaults)

    outputs = llm.generate(
        ["What is the capital of France? Answer in one word."],
        SamplingParams(max_tokens=16, temperature=0),
    )
    text = outputs[0].outputs[0].text.strip()
    assert text, "generation produced empty output"
    print(f"Generated: {text!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        _worker(args._worker)
        return

    failures = []
    for quant_level in QUANT_LEVELS:
        print(f"\n=== {MODEL_ID}/{quant_level} ===")
        proc = subprocess.run(
            [sys.executable, __file__, "--_worker", quant_level],
            timeout=3600,
        )
        if proc.returncode != 0:
            print(f"FAIL: {quant_level} crashed (returncode={proc.returncode})", file=sys.stderr)
            failures.append(quant_level)
        else:
            print(f"PASS: {quant_level} loads and generates.")

    print()
    if failures:
        print(f"FAIL: {len(failures)}/{len(QUANT_LEVELS)} quant_level(s) failed: {', '.join(failures)}")
        print(
            "Check the printed traceback above for the specific failure. If it looks like a "
            "missing checkpoint (awq/gptq hf_repo_override), verify the repo id still exists on "
            "HF Hub. If it's a vLLM quantization= argument error, this vLLM version's supported "
            "quantization methods may have changed -- check vllm.model_executor.layers."
            "quantization.QuantizationMethods."
        )
        sys.exit(1)

    print(f"PASS: all {len(QUANT_LEVELS)} quant_level(s) load and generate in this vLLM version.")


if __name__ == "__main__":
    main()
