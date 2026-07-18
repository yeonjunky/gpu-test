"""Walks results/<model_id>/<quant_level>/ and flattens everything into
results.csv (one row per model x quant_level x task, 11x3=33 rows when the
full matrix succeeds) and results_by_combo.csv (one row per model x
quant_level, 11 rows, for combo-level metrics like peak VRAM and load time
that don't vary by task).

Also cross-checks against configs/run_matrix.yaml and writes
results/MISSING_COMBOS.txt for any (model, quant_level) that doesn't have a
complete results directory, so report generation can surface gaps honestly
instead of silently omitting them.

Run after benchmark/orchestrator.py finishes (or partially finishes):
    python aggregate/aggregate_results.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from aggregate.schema import N_VALID_KEY, RESULTS_BY_COMBO_CSV_COLUMNS, RESULTS_CSV_COLUMNS, TASKS


def expected_combos(run_matrix_path: str) -> list[tuple[str, str, str]]:
    with open(run_matrix_path) as f:
        cfg = yaml.safe_load(f)
    combos = []
    for model in cfg["models"]:
        for run in model["runs"]:
            combos.append((model["id"], model["hf_repo"], run["quant_level"]))
    return combos


def load_combo(results_dir: Path, model_id: str, quant_level: str) -> dict | None:
    combo_dir = results_dir / model_id / quant_level
    scores_path = combo_dir / "scores.json"
    perf_path = combo_dir / "perf_metrics.json"
    if not scores_path.exists() or not perf_path.exists():
        return None
    with open(scores_path) as f:
        scores = json.load(f)
    with open(perf_path) as f:
        perf = json.load(f)
    return {"scores": scores, "perf": perf}


def build_rows(results_dir: Path, combos: list[tuple[str, str, str]]) -> tuple[list[dict], list[dict], list[tuple]]:
    task_rows, combo_rows, missing = [], [], []

    for model_id, hf_repo, quant_level in combos:
        loaded = load_combo(results_dir, model_id, quant_level)
        if loaded is None:
            missing.append((model_id, quant_level))
            continue

        scores, perf = loaded["scores"], loaded["perf"]
        quant_method = perf.get("quant_method", "unknown")

        accs = []
        for task in TASKS:
            task_score = scores.get(task, {})
            task_perf = perf.get("per_task", {}).get(task, {})
            acc = task_score.get("accuracy")
            if acc is not None:
                accs.append(acc)
            task_rows.append({
                "model_id": model_id,
                "hf_repo": hf_repo,
                "quant_level": quant_level,
                "quant_method": quant_method,
                "task": task,
                "accuracy_score": acc,
                "n_samples": task_score.get("n_samples"),
                "n_valid": task_score.get(N_VALID_KEY.get(task, ""), None),
                "throughput_tok_per_sec": task_perf.get("throughput_tok_per_sec"),
                "avg_ttft_ms": task_perf.get("avg_ttft_ms"),
                "avg_e2e_latency_ms": task_perf.get("avg_e2e_latency_ms"),
                "load_time_sec": perf.get("load_time_sec"),
                "peak_vram_mb": perf.get("peak_vram_mb_nvidia_smi"),
            })

        combo_rows.append({
            "model_id": model_id,
            "hf_repo": hf_repo,
            "quant_level": quant_level,
            "quant_method": quant_method,
            "load_time_sec": perf.get("load_time_sec"),
            "peak_vram_mb": perf.get("peak_vram_mb_nvidia_smi"),
            "avg_accuracy_score": sum(accs) / len(accs) if accs else None,
        })

    return task_rows, combo_rows, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-matrix", default="configs/run_matrix.yaml")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    combos = expected_combos(args.run_matrix)
    task_rows, combo_rows, missing = build_rows(results_dir, combos)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_df = pd.DataFrame(task_rows, columns=RESULTS_CSV_COLUMNS)
    task_df.to_csv(out_dir / "results.csv", index=False)
    task_df.to_json(out_dir / "results.json", orient="records", indent=2)

    combo_df = pd.DataFrame(combo_rows, columns=RESULTS_BY_COMBO_CSV_COLUMNS)
    combo_df.to_csv(out_dir / "results_by_combo.csv", index=False)

    missing_path = out_dir / "MISSING_COMBOS.txt"
    with open(missing_path, "w") as f:
        if missing:
            for model_id, quant_level in missing:
                f.write(f"{model_id}/{quant_level}\n")
        else:
            f.write("(none -- all combos present)\n")

    print(f"Expected combos: {len(combos)}, found: {len(combos) - len(missing)}, missing: {len(missing)}")
    if missing:
        print("Missing combos:")
        for model_id, quant_level in missing:
            print(f"  {model_id}/{quant_level}")
    print(f"Wrote {out_dir / 'results.csv'} ({len(task_rows)} rows)")
    print(f"Wrote {out_dir / 'results_by_combo.csv'} ({len(combo_rows)} rows)")
    print(f"Wrote {missing_path}")


if __name__ == "__main__":
    main()
