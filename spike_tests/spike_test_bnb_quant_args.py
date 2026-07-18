"""Spike test: is bnb_4bit_use_double_quant actually honored when passed via
vLLM's hf_overrides={"quantization_config": {...}}?

Loads a SMALL model (not one of the 4 target models, to keep this fast)
twice -- once with double_quant=False, once True -- each in its own
subprocess (clean VRAM teardown between loads), then compares peak VRAM.
Double-quant should show a measurably smaller footprint (~0.4 bits/param
less). If the two are statistically indistinguishable, hf_overrides is NOT
propagating the setting -- switch benchmark/run_one_combo.py's
--quant-config-mode to 'patched_config' for ALL bnb runs in the real matrix.

Run on the remote H100 box BEFORE trusting any *_doublequant_bnb rows:
    python spike_tests/spike_test_bnb_quant_args.py
"""
import argparse
import json
import subprocess
import sys

TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # small, fast, representative dense model


def _worker(double_quant: bool) -> None:
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=TEST_MODEL,
        dtype="bfloat16",
        quantization="bitsandbytes",
        hf_overrides={
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": double_quant,
            }
        },
        gpu_memory_utilization=0.5,
        max_model_len=2048,
    )
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

    results = {}
    for dq in ("false", "true"):
        print(f"=== Loading {TEST_MODEL} with bnb_4bit_use_double_quant={dq} ===")
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

    if delta > 5:
        print("PASS: bnb_4bit_use_double_quant appears to be honored via hf_overrides.")
    else:
        print(
            "FAIL: no measurable memory difference -- hf_overrides is likely NOT "
            "propagating bnb_4bit_use_double_quant. Re-run benchmark/run_one_combo.py "
            "with --quant-config-mode patched_config for all bnb runs in the full matrix."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
