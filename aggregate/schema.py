"""Canonical schema for per-combo result files written by
benchmark/run_one_combo.py and read by aggregate/aggregate_results.py.
Single source of truth so the two ends of the pipeline don't drift apart.

results/<model_id>/<quant_level>/
    task_a_raw_outputs.jsonl   one line per sample: {id, prompt, model_output,
                                ttft_ms, e2e_ms, output_tokens}
    task_b_raw_outputs.jsonl   same shape, task-specific id field
    task_c_raw_outputs.jsonl   same shape
    perplexity_raw_outputs.jsonl   ablation-study only (configs/ablation_matrix.yaml):
                                one line per chunk: {chunk_id, n_tokens, sum_nll}
    scores.json:
        {
          "task_a": {"accuracy": 0.0-1.0, "n_samples": int, "n_valid_json": int},
          "task_b": {"accuracy": 0.0-1.0, "n_samples": int, "n_passed": int},
          "task_c": {"accuracy": 0.0-1.0, "n_samples": int, "n_found": int},
          "perplexity": {"perplexity": float, "avg_nll": float, "n_chunks": int,
                          "n_tokens": int}   -- ablation-study only, no "accuracy" key
        }
    perf_metrics.json:
        {
          "model_id": str, "hf_repo": str, "quant_level": str, "quant_method": str,
          "weight_quant_method": str|null, "kv_cache_dtype": str,  -- ablation axes;
              null/"auto" for production run_matrix.yaml combos that predate them
          "load_time_sec": float,
          "peak_vram_mb_nvidia_smi": float, "peak_vram_mb_torch_allocator": float,
          "per_task": {
            "task_a": {"throughput_tok_per_sec": float, "avg_ttft_ms": float,
                       "avg_e2e_latency_ms": float, "n_samples": int},
            "task_b": {...}, "task_c": {...},
            "perplexity": {"wall_clock_sec": float, "n_samples": int,
                            "prefill_tok_per_sec": float}   -- different shape,
                            not read by the per-task row builder below
          }
        }
"""

# Downstream-accuracy generation tasks only -- perplexity is a per-combo
# scalar (see RESULTS_BY_COMBO_CSV_COLUMNS), not a per-task accuracy row, so
# it's deliberately excluded here.
TASKS = ["task_a", "task_b", "task_c"]

RESULTS_CSV_COLUMNS = [
    "model_id",
    "hf_repo",
    "quant_level",
    "quant_method",
    "weight_quant_method",
    "kv_cache_dtype",
    "task",
    "accuracy_score",
    "n_samples",
    "n_valid",
    "throughput_tok_per_sec",
    "avg_ttft_ms",
    "avg_e2e_latency_ms",
    "load_time_sec",
    "peak_vram_mb",
]

RESULTS_BY_COMBO_CSV_COLUMNS = [
    "model_id",
    "hf_repo",
    "quant_level",
    "quant_method",
    "weight_quant_method",
    "kv_cache_dtype",
    "load_time_sec",
    "peak_vram_mb",
    "avg_accuracy_score",
    "perplexity",
]

# the scores.json key that holds the "count of correct-ish things" per task,
# used purely for human-readable reporting (e.g. "32/50 passed")
N_VALID_KEY = {
    "task_a": "n_valid_json",
    "task_b": "n_passed",
    "task_c": "n_found",
}
