# Ablation-style quantization study: isolate each technique's contribution (Qwen2.5-32B pilot)

## Context

The current benchmark (`configs/run_matrix.yaml`, 11 combos across qwen2.5-32b/
gemma4-31b/llama3.3-70b) only measures each model's accuracy/throughput/VRAM at a
handful of *pre-combined* bitsandbytes quant levels (e.g. `int4_nf4_doublequant_bnb`).
It cannot answer "how much of that combo's accuracy drop came from the weight
quantization vs. some other factor?" — there's no ablation structure, just flat
end-state combos.

The user wants a proper ablation design: hold the model fixed, vary one technique at
a time, and log accuracy/throughput/peak-VRAM at every step, so each technique's
individual contribution is isolated instead of only seeing the combined effect. This
plan designs that as a new, separate pipeline invocation (not touching the existing
11 production combos or their results).

Scope decisions already made with the user (do not re-litigate):
- **Pilot on one model only**: qwen2.5-32b (smallest/fastest of the 3, fewest
  unknowns while validating the new machinery).
- **AWQ/GPTQ**: use official pre-quantized checkpoints, not self-quantized —
  confirmed to exist and be first-party Qwen-team releases:
  [`Qwen/Qwen2.5-32B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ)
  and [`Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4).
- **Accuracy axis**: add perplexity (WikiText-2) alongside the existing Task A/B/C
  downstream accuracy, not a replacement.
  - Note: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, `test` split.
- **CPU offload**: excluded from this phase entirely (out of scope, future work only).

Technical ground-truth confirmed by reading vLLM 0.25.1's own source
(`.venv/lib/python3.12/site-packages/vllm/`), not just docs:
- **AWQ/GPTQ**: vLLM 0.25.1 can *load* a pre-quantized checkpoint via
  `LLM(model=..., quantization="awq_marlin")` / `"gptq_marlin"` — no on-the-fly
  quantizer exists, checkpoint's `config.json` must already declare it. Fine here
  since we're using the official pre-quantized repos above.
- **KV-cache fp8**: `LLM(..., kv_cache_dtype="fp8")` is a genuinely independent axis
  (separate `CacheConfig`, no cross-validation against the weight-quant config found
  in source) — fully dynamic, scales default to 1.0, no calibration needed. Composes
  with *any* weight-quant method.
- **"Activation quant"**: the simplest calibration-free path is
  `LLM(model=<full-precision repo>, quantization="fp8")`, vLLM's on-the-fly dynamic
  W8A8 FP8 quantizer. **Important finding — this does not compose the way the user's
  original ladder assumed**: `quantization="fp8"` is a *complete alternative*
  weight+activation scheme applied to a full-precision checkpoint, not a layer you
  stack on top of an already bnb/AWQ/GPTQ-quantized model. There is no "weight-quant
  X + activation-quant" combo available in this tooling. `kv_cache_dtype="fp8"`
  still layers on top of `fp8`-weight-quant fine, though.

**Reframing the ladder to match reality** (stated here so the report methodology can
say this plainly instead of silently reinterpreting the brief):
- "Weight quant only" row becomes **5 mutually-exclusive peer methods** (all at
  `kv_cache_dtype=auto`): `bnb_int8`, `bnb_int4_nf4_doublequant`, `awq`, `gptq`,
  `fp8_online` (the last one *is* the activation-quant story, packaged as its own
  weight+activation scheme) — plus the `fp16_baseline` reference. 6 combos.
  This alone is the minimum viable deliverable for "compare quantization algorithms
  head-to-head" with perplexity + Task A/B/C accuracy + VRAM + throughput.
- "+ KV cache quant" row: each of the 6 above, re-run with `kv_cache_dtype=fp8` —
  this axis genuinely is additive/orthogonal. Optional stretch goal, up to 6 more
  combos (12 total) — run only after the first 6 look sane, since GPU-hours are the
  binding cost on a rented box.
- There is no valid "+ activation quant on top of weight quant" row for this
  tooling — drop it, and say why in the report rather than pretending otherwise.
- CPU offload: not implemented this phase; note as future work only.

## Approach

### 1. New `configs/ablation_matrix.yaml` (separate file, not an extension of `run_matrix.yaml`)

Why separate: keeps the existing 11 production combos and `results/` physically
untouched (zero risk to already-completed data or `--resume` logic), and
`report/generate_report.py`'s `QUANT_ORDER`/report template are tuned to the
3-model production framing — a 2D ablation grid is a different enough shape to
warrant its own file and its own results/report tree. `orchestrator.py` and
`aggregate/aggregate_results.py` already take `--config`/`--run-matrix` +
`--results-dir`/`--out-dir` as CLI args, so this costs nothing structurally — no
plumbing changes needed in either file.

Each `runs[]` entry keeps the existing required keys (`quant_level`, `quant_method`,
`dtype`, `bnb_args`) so `find_entries()`/`expected_combos()` need zero changes, plus
four new optional keys only `engine.py`/`aggregate` need to understand:
- `weight_quant_method` (str) — axis-1 label for reporting/plotting.
- `kv_cache_dtype` (str) — axis-2 label, passed straight to vLLM's `kv_cache_dtype=`.
- `hf_repo_override` (str|null) — AWQ/GPTQ live at a different HF repo than the base
  model; null means "use `model_entry.hf_repo`".
- `vllm_quantization` (str|null) — literal `quantization=` value for vLLM
  (`awq_marlin`/`gptq_marlin`/`fp8`); null for `bitsandbytes`/`none`.

The 6-combo Phase-1 deliverable's `runs[]`:
```yaml
- quant_level: fp16_baseline
  weight_quant_method: none
  kv_cache_dtype: auto
  quant_method: none
  dtype: bfloat16
  hf_repo_override: null
  vllm_quantization: null
  bnb_args: null

- quant_level: bnb_int8_kv_auto
  weight_quant_method: bnb_int8
  kv_cache_dtype: auto
  quant_method: bitsandbytes
  dtype: float16
  hf_repo_override: null
  vllm_quantization: null
  bnb_args: {load_in_8bit: true, load_in_4bit: false}

- quant_level: bnb_int4_nf4_doublequant_kv_auto
  weight_quant_method: bnb_int4_nf4_doublequant
  kv_cache_dtype: auto
  quant_method: bitsandbytes
  dtype: bfloat16
  hf_repo_override: null
  vllm_quantization: null
  bnb_args: {load_in_8bit: false, load_in_4bit: true, bnb_4bit_quant_type: nf4,
             bnb_4bit_compute_dtype: bfloat16, bnb_4bit_use_double_quant: true}

- quant_level: awq_kv_auto
  weight_quant_method: awq
  kv_cache_dtype: auto
  quant_method: awq
  dtype: float16
  hf_repo_override: Qwen/Qwen2.5-32B-Instruct-AWQ
  vllm_quantization: awq_marlin
  bnb_args: null

- quant_level: gptq_kv_auto
  weight_quant_method: gptq
  kv_cache_dtype: auto
  quant_method: gptq
  dtype: float16
  hf_repo_override: Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4
  vllm_quantization: gptq_marlin
  bnb_args: null

- quant_level: fp8_online_kv_auto
  weight_quant_method: fp8_online
  kv_cache_dtype: auto
  quant_method: fp8
  dtype: bfloat16
  hf_repo_override: null
  vllm_quantization: fp8
  bnb_args: null
```
Stretch-goal Phase 2 (optional, up to 6 more): duplicate each row above with
`quant_level: <name>_kv_fp8`, `kv_cache_dtype: fp8`.

### 2. `benchmark/engine.py` changes

Add `kv_cache_dtype` to `_build_base_kwargs()`, defaulting to `"auto"` (vLLM's own
default, so all 11 existing production combos are unaffected — `.get()` returns
`"auto"` for entries that don't set it, a no-op):
```python
kwargs["kv_cache_dtype"] = run_entry.get("kv_cache_dtype", "auto")
```

Extend `build_llm()`'s dispatch with 3 new `quant_method` branches. None of them need
a pre-quantize-and-cache step (unlike bitsandbytes) — they either load an
already-quantized community checkpoint or let vLLM quantize online:
```python
elif quant_method in ("awq", "gptq", "fp8"):
    kwargs["model"] = run_entry.get("hf_repo_override") or model_entry["hf_repo"]
    kwargs["quantization"] = run_entry["vllm_quantization"]
```
Also fix `quant_method == "none"` to respect `hf_repo_override` (currently always
uses `model_entry["hf_repo"]`) for symmetry, even though the pilot's `none` row
doesn't set it.

Update the module docstring: note `awq_marlin`/`gptq_marlin` require the checkpoint's
own `config.json` to declare the quant method (no on-the-fly AWQ/GPTQ quantizer in
this vLLM version), and that `quantization="fp8"` is a complete alternative
weight+activation scheme, not a layer stacked on bnb/awq/gptq.

Add `spike_tests/spike_test_build_llm_awq_gptq_fp8.py`, following
`spike_tests/spike_test_build_llm_bnb.py`'s existing pattern, calling
`build_llm()` against the 3 new `quant_method` rows pulled from
`configs/ablation_matrix.yaml` via `find_entries()` — validate before trusting the
orchestrator run, same spirit as every other spike test in this repo.

### 3. Perplexity measurement (new): vLLM `prompt_logprobs`, not a `transformers` fallback

Recommendation: measure perplexity on the *same already-loaded* `llm` object inside
`run_one_combo.py`, via `SamplingParams(max_tokens=1, prompt_logprobs=0)` +
`llm.generate()` on raw WikiText-2 text chunks (no chat template — this is plain
continuation, not `llm.chat()`).

Why not fall back to `transformers.forward(labels=...)` (the pattern
`_quantize_and_cache` already uses for bnb): it can't see two of this study's own
axes at all — no KV-cache-dtype concept in a plain transformers forward pass, and no
"dynamic online fp8" activation-quant equivalent either. Using vLLM's own
`prompt_logprobs` measures perplexity on the actual served model, including whatever
KV-cache/activation-quant is active for that combo.

New files/changes:
- `data/prep_scripts/prepare_perplexity_wikitext2.py` — same convention as
  `prepare_task_c_niah.py` (`huggingface_hub.hf_hub_download`, not `datasets`, which
  isn't installed): download WikiText-2 raw test split, chunk into fixed-size
  non-overlapping windows sized to fit `max_model_len`, seed=42 (repo convention).
  Output: `data/prepared/perplexity_wikitext2_50.jsonl`,
  `{"chunk_id": int, "text": str, "ref_token_count": int}` per line.
- `configs/tasks.yaml`: add a `perplexity` section (`data_file`, `n_samples`,
  `max_chunk_tokens`), mirroring `task_a`/`b`/`c`'s shape.
- `benchmark/task_runners/run_perplexity.py` — matches the existing
  `run(llm, data_path, task_cfg) -> (raw_outputs, scores_summary, perf_summary)`
  signature exactly, so it slots into `run_one_combo.py`'s `TASK_RUNNERS` dict
  unchanged in structure. Sums NLL from `output.prompt_logprobs` (dropping position
  0, which has no preceding context — standard WikiText2-perplexity convention),
  computes `perplexity = exp(total_nll / total_tokens)`. Output:
  `scores_summary = {"perplexity": float, "avg_nll": float, "n_chunks": int, "n_tokens": int}`.
  Confirm `RequestOutput.prompt_logprobs`'s exact shape against the pinned vLLM
  0.25.1 API in the spike test above before trusting it, same defensive spirit as
  `benchmark/timing.py`'s handling of vLLM's shifting metrics field names.
- `benchmark/run_one_combo.py`: register `"perplexity": run_perplexity` in
  `TASK_RUNNERS`. Fix the hardcoded `print(f"...accuracy={scores_summary['accuracy']:.3f}")`
  to branch on which key is present (perplexity's summary has no `"accuracy"` key,
  would otherwise `KeyError`). No separate `perplexity.json` needed — it folds into
  the existing `scores.json`/`perf_metrics.json["per_task"]` dict-of-tasks shape as
  just another key.
  Note: since `TASK_RUNNERS` is global, this also silently adds a cheap
  perplexity pass to any *future* production-matrix combo re-run — harmless/additive,
  worth a one-line mention in docs rather than gating behind a flag.

### 4. `aggregate/schema.py` + `aggregate/aggregate_results.py` changes

`perplexity` is a per-combo scalar (not per-task) → add to
`RESULTS_BY_COMBO_CSV_COLUMNS` only. `weight_quant_method`/`kv_cache_dtype` are
per-combo axis labels → add to both `RESULTS_CSV_COLUMNS` and
`RESULTS_BY_COMBO_CSV_COLUMNS`.

In `build_rows()`, derive with fallbacks so the existing 11 production
`perf_metrics.json` files need no regeneration:
```python
weight_quant_method = perf.get("weight_quant_method") or perf.get("quant_method", "unknown")
kv_cache_dtype = perf.get("kv_cache_dtype", "auto")
perplexity_score = scores.get("perplexity", {}).get("perplexity")
```
This requires `run_one_combo.py`'s `perf_metrics` dict to populate
`weight_quant_method`/`kv_cache_dtype` going forward (both `None`/`"auto"` for
old-style entries that don't set them — harmless).

`expected_combos()`/`load_combo()` need no changes at all (already generic over
whatever config/results path is passed via CLI flags).

### 5. `report/` — note only, don't redesign charts in this pass

- `report/generate_report.py`'s `QUANT_ORDER` list needs the 6 (or 12) new
  `quant_level` slugs appended so `order_quant_levels()` doesn't dump them
  unordered at the end.
- Since this pilot runs through a separate `results_ablation/`/`report_ablation/`
  tree, `generate_report.py` works unchanged via its existing `--results-dir`/
  `--out-dir` flags — only the ordering list needs touching to run cleanly.
  `report/templates/report_template.md.j2`'s prose is tuned to the "3 models x
  bitsandbytes" framing; a separate ablation-specific template (or at least a
  separate methodology paragraph) is cleaner than shoehorning both framings into one
  template — flagged as follow-up work, not designed in detail here.
- A perplexity-vs-weight-quant-method chart (grouped by `kv_cache_dtype`) is the
  natural addition once real numbers exist — follow-up, not designed here.

### 6. Docs

Once implemented and run, update (established pattern this session — every prior
schema/pipeline change kept these in lockstep):
- `docs/RUNBOOK.md` — new "Phase X: ablation pilot" command block (WikiText2 prep,
  new spike test, orchestrator/aggregate/report invocations against
  `configs/ablation_matrix.yaml`/`results_ablation/`).
- `docs/Updates.md` — entry documenting the new axes and the
  activation-quant/weight-quant non-composability finding (Section "ladder-vs-reality
  mismatch" above), so it's on record like every other plan deviation this session.
- `README.md` — mention the ablation matrix as a second, parallel entry point
  alongside the production `run_matrix.yaml`.

## Verification

1. `pytest tests/ -v` (no GPU needed) — confirm nothing regresses from the
   `_build_base_kwargs`/`engine.py` changes (existing tests assert specific keys,
   not full-dict equality, so the new `kv_cache_dtype` key shouldn't break them —
   but verify).
2. Run the new `spike_tests/spike_test_build_llm_awq_gptq_fp8.py` first, before any
   full combo — confirms `awq_marlin`/`gptq_marlin`/`fp8` all load correctly against
   the real Qwen2.5-32B-Instruct checkpoints on this vLLM 0.25.1 install.
3. Run Phase 1 (3 combos: `fp16_baseline`, `bnb_int4_nf4_doublequant_kv_auto`, `awq_kv_auto`)
   via `python -m benchmark.orchestrator --config configs/ablation_matrix.yaml --results-dir results_ablation`
   first — the bnb row is a known-good sanity check that the new perplexity/
   kv_cache_dtype plumbing doesn't silently break something that already worked;
   the awq row exercises the fully-new `quant_method="awq"` path end-to-end.
4. Manually inspect one combo's `scores.json` to confirm `perplexity` is a
   sane finite number (not NaN/inf/None) and roughly in the range expected for a
   32B instruct model on WikiText-2 (low-to-mid single digits to ~10s, not
   thousands) before trusting the rest of the run.
5. Run Phase 2 (`bnb_int8_kv_auto`, `gptq_kv_auto`, `fp8_online_kv_auto`) to
   complete the 6-combo primary deliverable.
6. `python aggregate/aggregate_results.py --run-matrix configs/ablation_matrix.yaml --results-dir results_ablation --out-dir results_ablation`
   then `python report/generate_report.py --results-dir results_ablation --out-dir report_ablation`
   — confirm `results.csv`/`results_by_combo.csv` have the new
   `weight_quant_method`/`kv_cache_dtype`/`perplexity` columns populated for all 6
   combos and `MISSING_COMBOS.txt` is empty.
7. Only after 3-6 look sane: decide whether to run the optional Phase 3
   (`kv_cache_dtype=fp8` variants, up to 6 more combos) given GPU-hour budget.
8. Update docs per section 6, then commit (confirm author identity/commit-message
   split with the user beforehand, matching this session's established pattern).

## Critical files

- `configs/ablation_matrix.yaml` (new)
- `benchmark/engine.py` (`_build_base_kwargs`, `build_llm` — new branches, no change
  to existing bnb/none behavior)
- `benchmark/task_runners/run_perplexity.py` (new)
- `data/prep_scripts/prepare_perplexity_wikitext2.py` (new)
- `configs/tasks.yaml` (add `perplexity` section)
- `benchmark/run_one_combo.py` (register `run_perplexity`, fix accuracy/perplexity
  print branch, populate `weight_quant_method`/`kv_cache_dtype` in `perf_metrics`)
- `aggregate/schema.py` (new CSV columns)
- `aggregate/aggregate_results.py` (`build_rows()` — new fields with backward-compat
  fallbacks)
- `report/generate_report.py` (`QUANT_ORDER` — append new slugs)
- `spike_tests/spike_test_build_llm_awq_gptq_fp8.py` (new)
- `docs/RUNBOOK.md`, `docs/Updates.md`, `README.md` (post-implementation doc sync)