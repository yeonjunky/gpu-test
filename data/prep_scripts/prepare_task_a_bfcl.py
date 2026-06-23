"""Prepares Task A (tool-use / JSON correctness) from BFCL.

Downloads the BFCL dataset (HF dataset repo, NOT loadable via
datasets.load_dataset() per BFCL's own README -- files are plain JSONL with a
.json extension) and samples 50 single-turn, multi-tool-call scenarios with
their ground truth, weighted toward higher-complexity categories.

Single-turn categories only (not multi_turn_* or exec_*): those have a much
more complex nested ground-truth structure (per-turn call lists, sometimes
involving simulated environment state) that doesn't fit a flat
json.loads()-based key/type comparison. The categories used here already
cover "multi-tool-call" (parallel/parallel_multiple) and "system control"
(live_* categories, sourced from real user-agent traffic) as specified.

Run locally, no GPU required:
    python data/prep_scripts/prepare_task_a_bfcl.py
"""
import json
import random
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
RAW_DIR = Path("data/raw/bfcl")
OUT_PATH = Path("data/prepared/task_a_bfcl_50.jsonl")
N_SAMPLES = 50
SEED = 42

# category -> exact sample quota (must sum to N_SAMPLES). Fixed quotas are used
# instead of per-item weights because raw category pool sizes vary wildly
# (live_multiple has ~1000 candidates, live_parallel only 16) -- per-item
# weighting would let pool size silently override the intended complexity
# skew. parallel_multiple / live_parallel_multiple = multiple distinct tools
# called at once (highest complexity); live_* = real-world agent/system-control
# traffic; plain multiple/parallel = simpler multi-call baselines for contrast.
CATEGORY_QUOTAS = {
    "parallel_multiple": 15,
    "live_parallel_multiple": 10,
    "multiple": 10,
    "live_multiple": 8,
    "parallel": 4,
    "live_parallel": 3,
}
assert sum(CATEGORY_QUOTAS.values()) == 50


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_user_content(question_turns: list) -> str:
    """BFCL 'question' is a list of turns, each turn a list of messages.
    Single-turn categories have exactly one non-empty turn."""
    for turn in question_turns:
        for msg in turn:
            if msg.get("role") == "user":
                return msg["content"]
    return ""


def normalize_expected(ground_truth: list[dict]) -> list[dict]:
    """BFCL ground_truth: [{func_name: {param: [acceptable_value, ...]}}, ...]
    -> [{"name": func_name, "args": {param: [acceptable_value, ...]}}, ...]
    Keeping the list-of-acceptable-values shape lets the scorer accept any
    one of several valid phrasings/defaults, as BFCL itself does.
    """
    calls = []
    for call in ground_truth:
        for func_name, args in call.items():
            calls.append({"name": func_name, "args": args})
    return calls


def main() -> None:
    print(f"Downloading {REPO_ID} ...")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
        allow_patterns=[f"BFCL_v3_{cat}.json" for cat in CATEGORY_QUOTAS]
        + [f"possible_answer/BFCL_v3_{cat}.json" for cat in CATEGORY_QUOTAS],
    )
    local_dir = Path(local_dir)

    rng = random.Random(SEED)
    chosen: list[dict] = []
    for category, quota in CATEGORY_QUOTAS.items():
        q_path = local_dir / f"BFCL_v3_{category}.json"
        a_path = local_dir / "possible_answer" / f"BFCL_v3_{category}.json"
        if not q_path.exists() or not a_path.exists():
            print(f"WARNING: missing files for category '{category}', skipping.")
            continue

        questions = {row["id"]: row for row in load_jsonl(q_path)}
        answers = {row["id"]: row for row in load_jsonl(a_path)}
        usable_ids = [qid for qid in questions if qid in answers]

        if len(usable_ids) < quota:
            print(
                f"WARNING: category '{category}' has only {len(usable_ids)} usable "
                f"scenarios, less than its quota of {quota}. Taking all of them."
            )
        picked_ids = rng.sample(usable_ids, min(quota, len(usable_ids)))

        for qid in picked_ids:
            q = questions[qid]
            chosen.append({
                "id": qid,
                "category": category,
                "question": first_user_content(q["question"]),
                "tools": q.get("function", []),
                "expected": {"function_calls": normalize_expected(answers[qid]["ground_truth"])},
            })

    rng.shuffle(chosen)
    print(f"Selected {len(chosen)} scenarios across {len(CATEGORY_QUOTAS)} categories.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for s in chosen:
            f.write(json.dumps(s) + "\n")

    print(f"Wrote {len(chosen)} samples to {OUT_PATH}")
    dist: dict[str, int] = {}
    for s in chosen:
        dist[s["category"]] = dist.get(s["category"], 0) + 1
    print("Category distribution:")
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
