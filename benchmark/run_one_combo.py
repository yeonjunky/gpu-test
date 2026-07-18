"""Runs ONE (model, quant_level) combo from configs/run_matrix.yaml against
all 3 tasks, then exits. Always launched as its OWN subprocess by
benchmark/orchestrator.py (never looped in-process) so process exit
guarantees full CUDA context teardown between combos -- important given the
new gemma4 architecture and the large per-model memory footprints involved.
GPU-only; not runnable on the local authoring machine.

Usage:
    python -m benchmark.run_one_combo --model-id qwen2.5-32b --quant-level int8_bnb
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import yaml

from benchmark.engine import build_llm
from benchmark.memory_monitor import MemoryMonitor
from benchmark.task_runners import run_task_a, run_task_b, run_task_c

TASK_RUNNERS = {"task_a": run_task_a, "task_b": run_task_b, "task_c": run_task_c}


def find_entries(run_matrix: dict, model_id: str, quant_level: str) -> tuple[dict, dict]:
    for model_entry in run_matrix["models"]:
        if model_entry["id"] != model_id:
            continue
        for run_entry in model_entry["runs"]:
            if run_entry["quant_level"] == quant_level:
                return model_entry, run_entry
        raise ValueError(f"quant_level '{quant_level}' not found for model '{model_id}'")
    raise ValueError(f"model_id '{model_id}' not found in run_matrix")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quant-level", required=True)
    parser.add_argument("--config", default="configs/run_matrix.yaml")
    parser.add_argument("--tasks-config", default="configs/tasks.yaml")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        run_matrix = yaml.safe_load(f)
    with open(args.tasks_config) as f:
        tasks_cfg = yaml.safe_load(f)

    model_entry, run_entry = find_entries(run_matrix, args.model_id, args.quant_level)
    defaults = run_matrix["defaults"]

    combo_dir = Path(args.results_dir) / args.model_id / args.quant_level
    combo_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Combo: {args.model_id}/{args.quant_level} (quant_method={run_entry['quant_method']}) ===")

    with MemoryMonitor(out_csv_path=str(combo_dir / "vram_timeseries.csv")) as monitor:
        t0 = time.time()
        llm = build_llm(model_entry, run_entry, defaults)
        load_time_sec = time.time() - t0
        print(f"Model loaded in {load_time_sec:.1f}s")

        scores_all, perf_all = {}, {}
        for task_name, runner in TASK_RUNNERS.items():
            task_cfg = tasks_cfg[task_name]
            print(f"--- Running {task_name} ({task_cfg['data_file']}) ---")
            raw_outputs, scores_summary, perf_summary = runner.run(llm, task_cfg["data_file"], task_cfg)

            with open(combo_dir / f"{task_name}_raw_outputs.jsonl", "w") as f:
                for row in raw_outputs:
                    f.write(json.dumps(row) + "\n")

            scores_all[task_name] = scores_summary
            perf_all[task_name] = perf_summary
            print(f"{task_name}: accuracy={scores_summary['accuracy']:.3f}")

        del llm
        gc.collect()
        try:
            import torch
            torch_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.empty_cache()
        except Exception:
            torch_peak_mb = None

    peak_vram_mb = monitor.peak_mb

    with open(combo_dir / "scores.json", "w") as f:
        json.dump(scores_all, f, indent=2)

    perf_metrics = {
        "model_id": args.model_id,
        "hf_repo": model_entry["hf_repo"],
        "quant_level": args.quant_level,
        "quant_method": run_entry["quant_method"],
        "load_time_sec": round(load_time_sec, 2),
        "peak_vram_mb_nvidia_smi": peak_vram_mb,
        "peak_vram_mb_torch_allocator": torch_peak_mb,
        "per_task": perf_all,
    }
    with open(combo_dir / "perf_metrics.json", "w") as f:
        json.dump(perf_metrics, f, indent=2)

    print(f"=== Combo {args.model_id}/{args.quant_level} complete. peak_vram_mb={peak_vram_mb} ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: combo failed: {e}", file=sys.stderr)
        raise
