# LLM Quantization Benchmark

Compares task accuracy, throughput, and memory footprint of 4 open-source
LLMs across multiple bitsandbytes quantization levels, served with vLLM on a
single rented H100 (80GB), and produces a written comparison report.

## Models & quantization ladder

| Model | HF repo | Levels (baseline -> most aggressive) |
|---|---|---|
| Qwen2.5-32B | `Qwen/Qwen2.5-32B-Instruct` | FP16 -> INT8 -> INT4-NF4 -> INT4-NF4+double-quant |
| gemma-4-31B | `google/gemma-4-31B` | BF16 -> INT8 -> INT4-NF4 -> INT4-NF4+double-quant |
| Llama-3.3-70B | `meta-llama/Llama-3.3-70B-Instruct` (gated) | INT8 -> INT4-NF4 -> INT4-NF4+double-quant |
| Mixtral-8x7B | `mistralai/Mixtral-8x7B-Instruct-v0.1` | INT8 -> INT4-NF4 -> INT4-NF4+double-quant |

Llama and Mixtral skip the FP16 baseline because their full-precision
footprint (~140GB and ~93GB respectively) exceeds a single 80GB H100. All
quantization is bitsandbytes, applied on-the-fly by vLLM (no separate
pre-quantized checkpoints). See `configs/run_matrix.yaml` for the exact
14-combo declarative matrix.

## Tasks

- **Task A** -- Tool-use / JSON correctness (BFCL, 50 samples, key+type match scoring)
- **Task B** -- Code generation (LiveCodeBench, medium/hard, 50 samples, sandboxed stdin/functional execution)
- **Task C** -- Needle in a Haystack (50 samples, 0-100% depth, exact substring match)

### Data contamination note (Task B)

HumanEval was deliberately NOT used for Task B: it has been public since 2021
and is almost certainly memorized by all 4 target models (their solutions
have been copied into countless repos/posts since release), which would
conflate "can solve" with "can recall." LiveCodeBench tags every problem with
a `contest_date`, so Task B is filtered to problems released after
**2025-01-31** -- strictly after `google/gemma-4-31B`'s training cutoff
(January 2025, the most recent of the 4 target models), guaranteeing none of
the 4 models could have seen these problems during training. See
`data/prep_scripts/prepare_task_b_livecodebench.py` for details.

Task A (BFCL, released ~Feb 2024) and Task C's haystack source (Paul Graham
essays, long pre-dating all 4 models) still carry some contamination risk --
see the report's Limitations section for the per-model reasoning.

## Repo layout

This machine (no GPU) authors and validates everything that doesn't need a
GPU; the remote H100 box only runs `benchmark/` and `spike_tests/`.

```
configs/            run_matrix.yaml (the 14 combos), tasks.yaml
data/prep_scripts/  build data/prepared/*.jsonl -- run locally, no GPU
scorers/             Task A/B/C scorers + sandboxed code-exec grader -- run locally, no GPU
tests/               pytest unit tests for the scorers -- run locally, no GPU
setup/               remote H100 bootstrap (install.sh, verify_env.py, download_models.py)
benchmark/            vLLM inference runner -- GPU only
spike_tests/          pre-flight checks for risky bnb/quant combos -- GPU only
aggregate/            results/ -> results.csv
report/               results.csv -> report/report.md + figures/*.png
```

## Quickstart

See `RUNBOOK.md` for the full step-by-step sequence. Short version:

```bash
# locally (no GPU needed)
python -m venv .venv-local && source .venv-local/bin/activate
pip install -r requirements-local.txt
python data/prep_scripts/prepare_task_a_bfcl.py
python data/prep_scripts/prepare_task_b_livecodebench.py
python data/prep_scripts/prepare_task_c_niah.py
pytest tests/

# request meta-llama/Llama-3.3-70B-Instruct gated access on HF NOW (day 1) --
# approval can take hours to a few days and should overlap with everything else

# then rsync/clone this repo to the rented H100 box and follow RUNBOOK.md
```

## Known risks

See `configs/run_matrix.yaml`'s `known_risk` field (Mixtral MoE+bnb) and
`spike_tests/` (must all pass before trusting the full 14-combo run):
Mixtral-8x7B (MoE) + bitsandbytes support in vLLM is newer/less mature than
dense-model bnb support; `bnb_4bit_use_double_quant` propagation through
vLLM's `LLM()` API is unverified until `spike_test_bnb_quant_args.py` passes;
gemma-4-31B + bitsandbytes is unverified (official vLLM docs only cover
W4A16/int8-per-channel for this model).
