"""Turns results/results.csv (+ results_by_combo.csv) into report/report.md
with comparison tables and matplotlib charts.

Run after aggregate/aggregate_results.py:
    python report/generate_report.py
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

TASK_LABELS = {
    "task_a": "Task A (Tool-use / JSON)",
    "task_b": "Task B (Code generation)",
    "task_c": "Task C (Needle in a Haystack)",
}

# Quant-level display order per model family (baseline -> most aggressive).
# Llama simply doesn't have a *_baseline fp16/bf16 entry (its full-precision
# footprint doesn't fit a single H100) -- the orchestrator/x-axis must not
# imply false comparability across models at "the same" position (see report
# note below).
QUANT_ORDER = [
    "fp16_baseline",
    "bf16_baseline",
    "int8_baseline",
    "int8_bnb",
    "int4_nf4_bnb",
    "int4_nf4_doublequant_bnb",
]


def order_quant_levels(levels: list[str]) -> list[str]:
    known = [q for q in QUANT_ORDER if q in levels]
    unknown = [q for q in levels if q not in QUANT_ORDER]
    return known + unknown


def plot_accuracy_vs_quant(df: pd.DataFrame, out_path: Path) -> None:
    models = sorted(df["model_id"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model_id in zip(axes[0], models):
        sub = df[df["model_id"] == model_id]
        for task in sub["task"].unique():
            task_sub = sub[sub["task"] == task]
            order = order_quant_levels(list(task_sub["quant_level"].unique()))
            task_sub = task_sub.set_index("quant_level").reindex(order).reset_index()
            ax.plot(task_sub["quant_level"], task_sub["accuracy_score"], marker="o", label=TASK_LABELS.get(task, task))
        ax.set_title(model_id)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7)
    fig.suptitle("Accuracy vs Quantization Level (per model, per task)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_metric_vs_quant(combo_df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_id in sorted(combo_df["model_id"].unique()):
        sub = combo_df[combo_df["model_id"] == model_id]
        order = order_quant_levels(list(sub["quant_level"].unique()))
        sub = sub.set_index("quant_level").reindex(order).reset_index()
        ax.plot(sub["quant_level"], sub[metric], marker="o", label=model_id)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    note = (
        "Note: Llama-3.3-70B starts at INT8 (not FP16/BF16) due to "
        "VRAM limits, so the same x-axis position is NOT directly comparable across "
        "all models -- see methodology."
    )
    return note


def plot_tradeoff_scatter(df: pd.DataFrame, combo_df: pd.DataFrame, out_path: Path) -> None:
    task_avg = df.groupby(["model_id", "quant_level"]).agg(
        accuracy_score=("accuracy_score", "mean"),
        throughput_tok_per_sec=("throughput_tok_per_sec", "mean"),
    ).reset_index()
    merged = task_avg.merge(
        combo_df[["model_id", "quant_level", "peak_vram_mb"]],
        on=["model_id", "quant_level"], how="left",
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_id in sorted(merged["model_id"].unique()):
        sub = merged[merged["model_id"] == model_id]
        sizes = (sub["peak_vram_mb"].fillna(0) / 200).clip(lower=20)
        ax.scatter(sub["throughput_tok_per_sec"], sub["accuracy_score"], s=sizes, label=model_id, alpha=0.7)
        for _, row in sub.iterrows():
            ax.annotate(row["quant_level"], (row["throughput_tok_per_sec"], row["accuracy_score"]), fontsize=6)
    ax.set_xlabel("Throughput (tokens/sec)")
    ax.set_ylabel("Avg accuracy (across 3 tasks)")
    ax.set_title("Throughput vs Accuracy trade-off (point size = peak VRAM)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_task_breakdown(df: pd.DataFrame, out_path: Path) -> None:
    tasks = sorted(df["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = df[df["task"] == task]
        for model_id in sorted(sub["model_id"].unique()):
            model_sub = sub[sub["model_id"] == model_id]
            order = order_quant_levels(list(model_sub["quant_level"].unique()))
            model_sub = model_sub.set_index("quant_level").reindex(order).reset_index()
            ax.plot(model_sub["quant_level"], model_sub["accuracy_score"], marker="o", label=model_id)
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="report")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "results.csv")
    combo_df = pd.read_csv(results_dir / "results_by_combo.csv")

    missing_path = results_dir / "MISSING_COMBOS.txt"
    missing_combos = []
    if missing_path.exists():
        lines = missing_path.read_text().strip().splitlines()
        if lines and lines[0] != "(none -- all combos present)":
            missing_combos = lines

    env_info_path = results_dir / "env_info.json"
    env_info = {}
    if env_info_path.exists():
        env_info = json.loads(env_info_path.read_text())

    plot_accuracy_vs_quant(df, fig_dir / "accuracy_vs_quant_per_model.png")
    throughput_note = plot_metric_vs_quant(
        combo_df.merge(
            df.groupby(["model_id", "quant_level"])["throughput_tok_per_sec"].mean().reset_index(),
            on=["model_id", "quant_level"], how="left",
        ),
        "throughput_tok_per_sec", "tokens/sec", "Throughput vs Quantization Level", fig_dir / "throughput_vs_quant_per_model.png",
    )
    plot_metric_vs_quant(combo_df, "peak_vram_mb", "Peak VRAM (MB)", "Peak VRAM vs Quantization Level", fig_dir / "memory_vs_quant_per_model.png")
    plot_tradeoff_scatter(df, combo_df, fig_dir / "tradeoff_scatter.png")
    plot_per_task_breakdown(df, fig_dir / "per_task_breakdown.png")

    env = Environment(loader=FileSystemLoader(str(out_dir / "templates")))
    template = env.get_template("report_template.md.j2")

    rendered = template.render(
        results_table=df.to_dict(orient="records"),
        combo_table=combo_df.to_dict(orient="records"),
        missing_combos=missing_combos,
        env_info=env_info,
        throughput_note=throughput_note,
        figures=[
            ("accuracy_vs_quant_per_model.png", "Accuracy vs Quantization Level (per model, per task)"),
            ("throughput_vs_quant_per_model.png", "Throughput vs Quantization Level"),
            ("memory_vs_quant_per_model.png", "Peak VRAM vs Quantization Level"),
            ("tradeoff_scatter.png", "Throughput vs Accuracy Trade-off"),
            ("per_task_breakdown.png", "Per-task Accuracy Breakdown"),
        ],
    )

    report_path = out_dir / "report.md"
    report_path.write_text(rendered)
    print(f"Wrote {report_path}")
    print(f"Wrote {len(list(fig_dir.glob('*.png')))} figures to {fig_dir}")


if __name__ == "__main__":
    main()
