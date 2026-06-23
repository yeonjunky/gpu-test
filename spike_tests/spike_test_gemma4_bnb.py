"""Spike test: does google/gemma-4-31B + bitsandbytes work in the pinned
vLLM version? The official vLLM Gemma4 recipe documents only W4A16 (QAT) and
int8-per-channel (MoE-variant-only) quantization paths -- NOT bitsandbytes.
gemma-4-31B is a brand-new dense architecture (Gemma4ForConditionalGeneration),
so generic bnb nn.Linear-replacement should work in principle, but is
unverified. Run BEFORE trusting any gemma4-31b rows in the full run_matrix.

PASS -> proceed with configs/run_matrix.yaml as-is.
FAIL -> drop bnb for this model and use vLLM's officially documented gemma4
        quantization path instead (W4A16/int8-per-channel), noting the
        substitution explicitly in the report's limitations section since it
        is no longer an apples-to-apples bnb comparison with the other models.

Usage:
    python spike_tests/spike_test_gemma4_bnb.py
"""
import sys

MODEL = "google/gemma-4-31B"


def main() -> None:
    from vllm import LLM, SamplingParams

    print(f"Loading {MODEL} with bnb INT8 (trust_remote_code=True, text-only) ...")
    try:
        llm = LLM(
            model=MODEL,
            dtype="bfloat16",
            quantization="bitsandbytes",
            hf_overrides={"quantization_config": {"load_in_8bit": True, "load_in_4bit": False}},
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
        )
    except Exception as e:
        print(f"FAIL: model load raised an exception: {e}", file=sys.stderr)
        print(
            "If this looks like an unsupported-quantization error, gemma4 may only "
            "support W4A16/int8-per-channel per the official vLLM recipe -- drop bnb "
            "for this model and use that documented path instead, noting the "
            "substitution in the report's limitations section."
        )
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
        print("FAIL: generation produced empty output.")
        sys.exit(1)

    print("PASS: gemma-4-31B + bitsandbytes loads and generates in this vLLM version.")


if __name__ == "__main__":
    main()
