# LLM Quantization Benchmark Report

Comparing task accuracy, throughput, and memory footprint of 3 open-source
LLMs across multiple bitsandbytes quantization levels, served with vLLM on a
single rented H100 (80GB). MoE architectures (e.g. Mixtral) are out of scope
-- see the Known Gaps section below.

## Methodology

- **Models**: Qwen2.5-32B-Instruct, Llama-3.3-70B-Instruct, google/gemma-4-31B-it
- **Quantization**: bitsandbytes, pre-quantized offline via `transformers` + `BitsAndBytesConfig` and saved to a local checkpoint that vLLM loads directly (vLLM's in-flight `hf_overrides={"quantization_config": ...}` path was found to be broken -- a weight-shape `AssertionError` reproduced across every vLLM version tested, 0.9.2-0.25.1 -- see `docs/spike_test_error_report.md`). Qwen2.5-32B and gemma-4-31B-it start at FP16/BF16; Llama-3.3-70B starts at INT8 because its FP16 footprint (~140GB) exceeds a single 80GB H100.
- **Tasks**: Task A (tool-use/JSON correctness, BFCL, 50 samples, key+type match scoring), Task B (code generation, LiveCodeBench, 50 medium/hard samples released after 2025-01-31, sandboxed stdin/functional execution), Task C (Needle in a Haystack, 50 samples spanning 0-100% depth, exact substring match).
- **Sampling**: temperature=0 for all generations (deterministic).
- **Metrics**: task accuracy, throughput (tokens/sec), TTFT + end-to-end latency, peak VRAM (via `nvidia-smi` polling, cross-checked against `torch.cuda.max_memory_allocated()`).

- **Environment**: NVIDIA H100 NVL (93.1 GB), torch 2.11.0+cu130, vLLM 0.25.1, bitsandbytes 0.49.2.


**Important caveat**: Note: Llama-3.3-70B starts at INT8 (not FP16/BF16) due to VRAM limits, so the same x-axis position is NOT directly comparable across all models -- see methodology.

### Data Contamination

- **Task B (LiveCodeBench)**: filtered to `contest_date > 2025-01-31`, strictly after `google/gemma-4-31B-it`'s training cutoff (January 2025, the most recent of the 3 target models per its model card). None of the 3 models could have seen these problems during training. HumanEval was deliberately avoided for this reason -- it has been public since 2021 and is widely considered heavily memorized by modern open LLMs.
- **Task A (BFCL)**: released ~Feb 2024. Contamination risk varies by model: Llama-3.3-70B's pretraining data cutoff (December 2023, per its model card -- the model itself was released Dec 2024, but the underlying pretraining corpus predates BFCL) predates BFCL's release, though later instruction-tuning data collection could theoretically have included it; Qwen2.5-32B and gemma-4-31B-it were trained after BFCL's release and may have been exposed to it directly or indirectly. Treat Task A cross-model comparisons with this asymmetry in mind.
- **Task C (Needle in a Haystack)**: the Paul Graham essay source text is almost certainly known to all 3 models, but the injected secret string is freshly randomly generated per data-prep run and cannot appear in any training corpus -- the haystack source being memorized does not let a model "already know" the answer being tested, so this task is comparatively robust to contamination.

## Known Gaps / Risks Encountered


All 11 expected (model, quant_level) combinations are present in this report.


**MoE models (e.g. Mixtral-8x7B) are excluded from this benchmark's scope
entirely**, not just missing a combo to backfill: this transformers version
stores every MoE architecture's experts as fused 3D tensors
(`gate_up_proj`/`down_proj`) rather than individual `nn.Linear` modules, so
bitsandbytes' int8/int4 quantization paths cannot reach them at all regardless
of `device_map`, CPU offload, or streaming technique -- confirmed to affect
essentially every current MoE architecture in this transformers version, not
just Mixtral (see `docs/Updates.md` and `docs/spike_test_error_report.md` for
the full investigation).

Known a-priori risks tracked during this project (see `configs/run_matrix.yaml` `known_risk` fields, `spike_tests/`, and `docs/spike_test_error_report.md`):
- `bnb_4bit_use_double_quant` propagation and gemma-4-31B-it + bitsandbytes were both verified end-to-end via `spike_tests/spike_test_gemma4_bnb.py`, which calls `benchmark.engine.build_llm` directly against the real `configs/run_matrix.yaml` rows -- the same code path this run used -- rather than a hand-typed stand-in config.
- **Llama-3.3-70B shows a non-monotonic accuracy pattern across quant levels**: Task C (needle-in-a-haystack) scores 0.26 at INT8, 0.36 at NF4, but 0.98 at NF4+double-quant -- the *least* precise variant performs best, the opposite of the usual quantization/quality tradeoff. Inspecting raw generations shows the INT8/NF4 runs frequently fall into repetitive, rambling text instead of directly stating the answer (avg ~50-60 output tokens, often hitting the task's 64-token cap before answering), while NF4+double-quant answers tersely and directly (avg ~10 tokens, matching Qwen's style). All three checkpoints were freshly quantized for this run (not a stale-cache artifact). This looks like quantization noise nondeterministically nudging greedy (temperature=0) decoding into or out of a repetition-loop failure mode specific to this model, rather than a straightforward precision-vs-quality effect -- treat Llama-3.3-70B's Task B/C numbers as a real, reportable finding rather than a clean quantization comparison.

## Results Table (per model x quant_level x task)

| Model | Quant Level | Task | Accuracy | n_samples | Throughput (tok/s) | Avg TTFT (ms) | Avg E2E (ms) | Peak VRAM (MB) |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-32b | fp16_baseline | task_a | 0.90 | 50 | 298.4 | nan | nan | 89065 |
| qwen2.5-32b | fp16_baseline | task_b | 0.26 | 50 | 584.9 | nan | nan | 89065 |
| qwen2.5-32b | fp16_baseline | task_c | 1.00 | 50 | 5.1 | nan | nan | 89065 |
| qwen2.5-32b | int8_bnb | task_a | 0.89 | 50 | 81.3 | nan | nan | 88237 |
| qwen2.5-32b | int8_bnb | task_b | 0.18 | 50 | 109.7 | nan | nan | 88237 |
| qwen2.5-32b | int8_bnb | task_c | 1.00 | 50 | 3.4 | nan | nan | 88237 |
| qwen2.5-32b | int4_nf4_bnb | task_a | 0.89 | 50 | 197.1 | nan | nan | 90145 |
| qwen2.5-32b | int4_nf4_bnb | task_b | 0.16 | 50 | 236.6 | nan | nan | 90145 |
| qwen2.5-32b | int4_nf4_bnb | task_c | 1.00 | 50 | 5.0 | nan | nan | 90145 |
| qwen2.5-32b | int4_nf4_doublequant_bnb | task_a | 0.89 | 50 | 196.0 | nan | nan | 90167 |
| qwen2.5-32b | int4_nf4_doublequant_bnb | task_b | 0.16 | 50 | 187.6 | nan | nan | 90167 |
| qwen2.5-32b | int4_nf4_doublequant_bnb | task_c | 1.00 | 50 | 5.0 | nan | nan | 90167 |
| gemma4-31b | bf16_baseline | task_a | 0.84 | 50 | 173.7 | nan | nan | 90033 |
| gemma4-31b | bf16_baseline | task_b | 0.64 | 50 | 520.4 | nan | nan | 90033 |
| gemma4-31b | bf16_baseline | task_c | 1.00 | 50 | 5.3 | nan | nan | 90033 |
| gemma4-31b | int8_bnb | task_a | 0.84 | 50 | 46.9 | nan | nan | 89959 |
| gemma4-31b | int8_bnb | task_b | 0.62 | 50 | 128.8 | nan | nan | 89959 |
| gemma4-31b | int8_bnb | task_c | 1.00 | 50 | 3.3 | nan | nan | 89959 |
| gemma4-31b | int4_nf4_bnb | task_a | 0.84 | 50 | 191.3 | nan | nan | 91527 |
| gemma4-31b | int4_nf4_bnb | task_b | 0.60 | 50 | 362.4 | nan | nan | 91527 |
| gemma4-31b | int4_nf4_bnb | task_c | 1.00 | 50 | 4.9 | nan | nan | 91527 |
| gemma4-31b | int4_nf4_doublequant_bnb | task_a | 0.84 | 50 | 191.2 | nan | nan | 91701 |
| gemma4-31b | int4_nf4_doublequant_bnb | task_b | 0.56 | 50 | 352.8 | nan | nan | 91701 |
| gemma4-31b | int4_nf4_doublequant_bnb | task_c | 1.00 | 50 | 4.9 | nan | nan | 91701 |
| llama3.3-70b | int8_baseline | task_a | 0.83 | 50 | 65.2 | nan | nan | 91157 |
| llama3.3-70b | int8_baseline | task_b | 0.20 | 50 | 31.3 | nan | nan | 91157 |
| llama3.3-70b | int8_baseline | task_c | 0.26 | 50 | 5.1 | nan | nan | 91157 |
| llama3.3-70b | int4_nf4_bnb | task_a | 0.90 | 50 | 94.5 | nan | nan | 93227 |
| llama3.3-70b | int4_nf4_bnb | task_b | 0.34 | 50 | 117.0 | nan | nan | 93227 |
| llama3.3-70b | int4_nf4_bnb | task_c | 0.36 | 50 | 3.2 | nan | nan | 93227 |
| llama3.3-70b | int4_nf4_doublequant_bnb | task_a | 0.90 | 50 | 93.8 | nan | nan | 93277 |
| llama3.3-70b | int4_nf4_doublequant_bnb | task_b | 0.40 | 50 | 124.9 | nan | nan | 93277 |
| llama3.3-70b | int4_nf4_doublequant_bnb | task_c | 0.98 | 50 | 1.9 | nan | nan | 93277 |


## Per-Combo Summary (memory & load time, not task-specific)

| Model | Quant Level | Load Time (s) | Peak VRAM (MB) | Avg Accuracy (3 tasks) |
|---|---|---|---|---|
| qwen2.5-32b | fp16_baseline | 401.5 | 89065 | 0.72 |
| qwen2.5-32b | int8_bnb | 87.6 | 88237 | 0.69 |
| qwen2.5-32b | int4_nf4_bnb | 151.1 | 90145 | 0.68 |
| qwen2.5-32b | int4_nf4_doublequant_bnb | 152.6 | 90167 | 0.68 |
| gemma4-31b | bf16_baseline | 272.8 | 90033 | 0.83 |
| gemma4-31b | int8_bnb | 135.5 | 89959 | 0.82 |
| gemma4-31b | int4_nf4_bnb | 357.2 | 91527 | 0.81 |
| gemma4-31b | int4_nf4_doublequant_bnb | 362.6 | 91701 | 0.80 |
| llama3.3-70b | int8_baseline | 799.3 | 91157 | 0.43 |
| llama3.3-70b | int4_nf4_bnb | 437.2 | 93227 | 0.53 |
| llama3.3-70b | int4_nf4_doublequant_bnb | 408.0 | 93277 | 0.76 |


## Charts

### Accuracy vs Quantization Level (per model, per task)

![Accuracy vs Quantization Level (per model, per task)](figures/accuracy_vs_quant_per_model.png)

### Throughput vs Quantization Level

![Throughput vs Quantization Level](figures/throughput_vs_quant_per_model.png)

### Peak VRAM vs Quantization Level

![Peak VRAM vs Quantization Level](figures/memory_vs_quant_per_model.png)

### Throughput vs Accuracy Trade-off

![Throughput vs Accuracy Trade-off](figures/tradeoff_scatter.png)

### Per-task Accuracy Breakdown

![Per-task Accuracy Breakdown](figures/per_task_breakdown.png)



## Limitations

- Task B grading uses only LiveCodeBench's public test cases (2-4 per problem), not its full private suite (which is encoded and used by the official leaderboard for stronger held-out grading) -- a weaker but model-agnostic signal applied identically across all 3 models.
- Task A scoring is a project-specific simplified key/type matcher, not a reproduction of the official BFCL AST-equivalence grading.
- Llama-3.3-70B's comparisons start from INT8, not FP16/BF16 -- it is not on the same starting baseline as Qwen2.5-32B and gemma-4-31B.
- Token counts in the Needle-in-a-Haystack haystack use a fixed reference tokenizer (cl100k_base) rather than each model's native tokenizer, for cross-model consistency.
- See the Data Contamination subsection above for per-task and per-model caveats on result validity.