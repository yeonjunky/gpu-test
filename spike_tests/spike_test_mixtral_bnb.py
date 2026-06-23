"""Spike test: does Mixtral-8x7B (MoE) + bitsandbytes actually work in the
pinned vLLM version? Historically unstable -- MoE in-flight bnb support was
added incrementally (vllm-project/vllm PR #20061, tracking issue #20480),
and vLLM has an open RFC (#39583) proposing to deprecate bnb entirely. Run
this BEFORE trusting any mixtral-8x7b rows in the full run_matrix.

PASS -> proceed with configs/run_matrix.yaml as-is.
FAIL -> apply the fallback noted in run_matrix.yaml's `known_risk` field for
        mixtral-8x7b: use HF transformers + BitsAndBytesConfig for the
        Mixtral leg only, and flag in the report that Mixtral's
        throughput/latency numbers are then not directly comparable to the
        vLLM-served numbers for the other 3 models.

Usage:
    python spike_tests/spike_test_mixtral_bnb.py
"""
import sys

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


def main() -> None:
    from vllm import LLM, SamplingParams

    print(f"Loading {MODEL} with bnb INT8 ...")
    try:
        llm = LLM(
            model=MODEL,
            dtype="float16",
            quantization="bitsandbytes",
            hf_overrides={"quantization_config": {"load_in_8bit": True, "load_in_4bit": False}},
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
        )
    except Exception as e:
        print(f"FAIL: model load raised an exception: {e}", file=sys.stderr)
        print("See configs/run_matrix.yaml known_risk field for the transformers-native fallback plan.")
        sys.exit(1)

    try:
        outputs = llm.generate(
            ["What is the capital of France? Answer in one word."],
            SamplingParams(max_tokens=16, temperature=0),
        )
        text = outputs[0].outputs[0].text.strip()
    except Exception as e:
        print(f"FAIL: generation raised an exception: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated: {text!r}")
    if not text:
        print("FAIL: generation produced empty output (possible silently broken MoE+bnb kernel).")
        sys.exit(1)

    print("PASS: Mixtral-8x7B + bitsandbytes loads and generates in this vLLM version.")
    print("NOTE: this only checks load+generate succeed, not output quality -- spot-check")
    print("a few real Task A/B/C outputs manually after the first Mixtral combo completes.")


if __name__ == "__main__":
    main()
