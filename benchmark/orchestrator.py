"""Loops over every (model, quant_level) combo in configs/run_matrix.yaml,
launching benchmark/run_one_combo.py as a fresh subprocess each time (process
exit = guaranteed CUDA context teardown between combos). One combo failing
does not abort the rest of the matrix. Supports --resume to skip combos that
already have a complete results directory, and --only to re-run a single
model's combos after a config fix.

Usage:
    python -m benchmark.orchestrator --config configs/run_matrix.yaml
    python -m benchmark.orchestrator --resume
    python -m benchmark.orchestrator --only llama3.3-70b
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

COMBO_TIMEOUT_SECONDS = 2 * 60 * 60  # generous per-combo ceiling


def combo_already_done(results_dir: Path, model_id: str, quant_level: str) -> bool:
    combo_dir = results_dir / model_id / quant_level
    return (combo_dir / "perf_metrics.json").exists() and (combo_dir / "scores.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run_matrix.yaml")
    parser.add_argument("--tasks-config", default="configs/tasks.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--resume", action="store_true", help="skip combos that already have complete results")
    parser.add_argument("--only", help="only run combos for this model_id")
    args = parser.parse_args()

    with open(args.config) as f:
        run_matrix = yaml.safe_load(f)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    combos = []
    for model_entry in run_matrix["models"]:
        if args.only and model_entry["id"] != args.only:
            continue
        for run_entry in model_entry["runs"]:
            combos.append((model_entry["id"], run_entry["quant_level"]))

    print(f"Planned combos: {len(combos)}")
    outcomes = []

    for model_id, quant_level in combos:
        combo_label = f"{model_id}/{quant_level}"

        if args.resume and combo_already_done(results_dir, model_id, quant_level):
            print(f"[SKIP] {combo_label} already complete (--resume)")
            outcomes.append((combo_label, "SKIPPED"))
            continue

        print(f"\n[START] {combo_label}")
        t0 = time.time()
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "benchmark.run_one_combo",
                    "--model-id", model_id,
                    "--quant-level", quant_level,
                    "--config", args.config,
                    "--tasks-config", args.tasks_config,
                    "--results-dir", args.results_dir,
                ],
                timeout=COMBO_TIMEOUT_SECONDS,
            )
            elapsed = time.time() - t0
            if result.returncode == 0:
                print(f"[OK] {combo_label} ({elapsed:.0f}s)")
                outcomes.append((combo_label, "SUCCESS"))
            else:
                print(f"[FAILED] {combo_label} (returncode={result.returncode}, {elapsed:.0f}s)")
                outcomes.append((combo_label, f"FAILED (rc={result.returncode})"))
                (results_dir / model_id / quant_level).mkdir(parents=True, exist_ok=True)
                (results_dir / model_id / quant_level / "FAILED.marker").write_text(
                    f"returncode={result.returncode}\n"
                )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"[TIMEOUT] {combo_label} after {elapsed:.0f}s")
            outcomes.append((combo_label, "TIMEOUT"))
            (results_dir / model_id / quant_level).mkdir(parents=True, exist_ok=True)
            (results_dir / model_id / quant_level / "FAILED.marker").write_text("timeout\n")

    print("\n=== Summary ===")
    for combo_label, status in outcomes:
        print(f"{combo_label}: {status}")

    n_failed = sum(1 for _, status in outcomes if status not in ("SUCCESS", "SKIPPED"))
    if n_failed:
        print(f"\n{n_failed} combo(s) did not complete successfully. See FAILED.marker files under {results_dir}/.")
        sys.exit(1)


if __name__ == "__main__":
    main()
