"""Spike test: does benchmark.engine.build_llm's pre-quantize-and-cache bnb
path (transformers + BitsAndBytesConfig -> save_pretrained -> vLLM loads the
local quantized checkpoint) actually work, and does it honor
bnb_4bit_use_double_quant?

Supersedes spike_test_bnb_quant_args.py, which checked whether vLLM's
hf_overrides={"quantization_config": {...}} propagated bnb_4bit_use_double_quant.
That mechanism was confirmed broken (weight-shape AssertionError in vLLM's
loader, reproduced across every vLLM version from 0.9.2 to 0.25.1 -- see
spike_test_error_report.md) and benchmark/engine.py::build_llm no longer uses
it. This test instead calls build_llm(...) directly -- the same function
run_one_combo.py uses -- so it actually exercises the current production
code path instead of a deprecated one.

Loads a SMALL model (not one of the 4 target models, to keep this fast)
twice through build_llm() -- once with double_quant=False, once True --
each in its own subprocess (clean VRAM teardown between loads), then
compares peak VRAM. Double-quant should show a measurably smaller footprint
(~0.4 bits/param less). If the two are statistically indistinguishable,
build_llm's quantize-and-cache path is not honoring the setting.

Run on the remote H100 box as part of Phase 2:
    python spike_tests/spike_test_build_llm_bnb.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `benchmark` is importable regardless of cwd

TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # small, fast, representative dense model
CACHE_DIR = "/tmp/spike_test_build_llm_bnb_cache"


def _worker(double_quant: bool) -> None:
    import torch

    from benchmark.engine import build_llm
    from vllm import SamplingParams

    model_entry = {"id": "spike-test", "hf_repo": TEST_MODEL}
    run_entry = {
        "quant_level": f"nf4_doublequant_{double_quant}",
        "quant_method": "bitsandbytes",
        "dtype": "bfloat16",
        "bnb_args": {
            "load_in_8bit": False,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": double_quant,
        },
    }
    defaults = {
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.5,
        "seed": 42,
    }

    llm = build_llm(model_entry, run_entry, defaults, quantized_cache_dir=CACHE_DIR)
    outputs = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=8))
    assert outputs[0].outputs[0].text, "generation produced empty output"
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(json.dumps({"peak_mb": peak_mb}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--double-quant", choices=["true", "false"], help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        _worker(args.double_quant == "true")
        return

    shutil.rmtree(CACHE_DIR, ignore_errors=True)

    results = {}
    for dq in ("false", "true"):
        print(f"=== build_llm({TEST_MODEL}, bnb_4bit_use_double_quant={dq}) ===")
        proc = subprocess.run(
            [sys.executable, __file__, "--_worker", "--double-quant", dq],
            capture_output=True, text=True, timeout=1600,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            print(f"FAIL: worker crashed for double_quant={dq}")
            sys.exit(1)
        last_line = proc.stdout.strip().splitlines()[-1]
        results[dq] = json.loads(last_line)["peak_mb"]
        print(f"peak VRAM (torch allocator): {results[dq]:.1f} MB")

    delta = results["false"] - results["true"]
    pct = 100 * delta / results["false"] if results["false"] else 0
    print(f"\nDelta: {delta:.1f} MB ({pct:.1f}% reduction with double-quant)")

    shutil.rmtree(CACHE_DIR, ignore_errors=True)

    if delta > 5:
        print("PASS: bnb_4bit_use_double_quant is honored by build_llm's quantize-and-cache path.")
    else:
        print(
            "FAIL: no measurable memory difference -- build_llm is not correctly "
            "applying bnb_4bit_use_double_quant. Check _quantize_and_cache in "
            "benchmark/engine.py."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
