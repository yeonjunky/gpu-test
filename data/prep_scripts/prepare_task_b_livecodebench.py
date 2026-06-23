"""Prepares Task B (code generation) from LiveCodeBench, filtered to problems
released AFTER the most recently trained target model's cutoff date.

Why LiveCodeBench instead of HumanEval: HumanEval (2021) is almost certainly
memorized by all 4 target models (it has been copied into countless GitHub
repos, blog posts, and papers since release) -- scoring on it conflates
"can solve" with "can recall." LiveCodeBench is built specifically to avoid
this: every problem carries a contest_date, so problems can be filtered to
ones released after a model's training cutoff, guaranteeing the model could
not have seen them. google/gemma-4-31B has the latest cutoff of the 4 target
models (January 2025, per its model card's Training Dataset section), so
filtering to contest_date > 2025-01-31 gives a "could not have been trained
on this" guarantee for ALL 4 models, not just gemma-4.

Only the `release_v6` shard (test6.jsonl) contains problems past the
2025-01-31 cutoff (release_v5 covers up to Jan 2025); earlier shards are not
downloaded since every row in them is excluded by the date filter anyway.

This project's grading uses ONLY the public test cases (plaintext, no
decoding needed) rather than LiveCodeBench's official private test suite
(zlib+base64-encoded, used by the official harness for stronger held-out
grading) -- a deliberate simplification, documented here and in the report's
limitations section. This means our pass/fail signal is weaker than the
official LiveCodeBench leaderboard's, but it is applied identically across
all 4 models, which is what this comparison needs.

Two problem formats appear in the post-cutoff medium/hard pool:
  - AtCoder problems: "stdin" testtype -- model must write a full program
    that reads from stdin and prints to stdout.
  - LeetCode problems: "functional" testtype -- model completes a method
    body inside the given `starter_code` class skeleton.
Both are kept (not just one) for closer fidelity to the official benchmark's
problem mix; scorers/score_task_b.py handles both via
scorers/sandbox/exec_worker.py's two grading modes.

Run locally, no GPU required:
    python data/prep_scripts/prepare_task_b_livecodebench.py
"""
import json
import random
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "livecodebench/code_generation_lite"
FILENAME = "test6.jsonl"  # release_v6 shard; the only one with post-2025-01-31 problems
RAW_PATH = Path("data/raw/livecodebench/test6.jsonl")
OUT_PATH = Path("data/prepared/task_b_livecodebench_50.jsonl")

CUTOFF_DATE = "2025-01-31"  # strictly after google/gemma-4-31B's Jan 2025 training cutoff
DIFFICULTIES = ("medium", "hard")
N_SAMPLES = 50
SEED = 42


def download_raw() -> list[dict]:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        downloaded = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FILENAME)
        RAW_PATH.write_bytes(Path(downloaded).read_bytes())
    return [json.loads(line) for line in RAW_PATH.read_text().splitlines() if line.strip()]


def normalize(row: dict) -> dict:
    public_tests = json.loads(row["public_test_cases"])
    test_type = public_tests[0]["testtype"] if public_tests else "unknown"
    return {
        "question_id": row["question_id"],
        "platform": row["platform"],
        "contest_date": row["contest_date"],
        "difficulty": row["difficulty"],
        "question_title": row["question_title"],
        "question_content": row["question_content"],
        "starter_code": row["starter_code"],
        "test_type": test_type,
        "public_tests": [{"input": t["input"], "output": t["output"]} for t in public_tests],
    }


def main() -> None:
    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    rows = download_raw()
    print(f"Loaded {len(rows)} raw problems from {FILENAME}.")

    post_cutoff = [r for r in rows if r["contest_date"] > CUTOFF_DATE and r["difficulty"] in DIFFICULTIES]
    print(f"Post-cutoff (contest_date > {CUTOFF_DATE}), medium+hard: {len(post_cutoff)} problems.")
    if len(post_cutoff) < N_SAMPLES:
        raise RuntimeError(
            f"Only {len(post_cutoff)} eligible problems found, need {N_SAMPLES}. "
            f"LiveCodeBench may need a newer release shard -- check "
            f"https://huggingface.co/datasets/{REPO_ID} for a release_v7+ file."
        )

    rng = random.Random(SEED)
    by_difficulty: dict[str, list[dict]] = {d: [] for d in DIFFICULTIES}
    for r in post_cutoff:
        by_difficulty[r["difficulty"]].append(r)

    # proportional stratified sample so the 50-problem set keeps roughly the
    # same medium:hard ratio as the eligible pool, rather than skewing
    # toward whichever difficulty happens to be more numerous.
    quotas = {}
    remaining = N_SAMPLES
    diffs = list(DIFFICULTIES)
    for i, d in enumerate(diffs):
        if i == len(diffs) - 1:
            quotas[d] = remaining
        else:
            q = round(N_SAMPLES * len(by_difficulty[d]) / len(post_cutoff))
            quotas[d] = q
            remaining -= q

    selected = []
    for d, quota in quotas.items():
        pool = by_difficulty[d]
        picked = rng.sample(pool, min(quota, len(pool)))
        selected.extend(picked)
    rng.shuffle(selected)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in selected:
            f.write(json.dumps(normalize(row)) + "\n")

    print(f"Wrote {len(selected)} samples to {OUT_PATH}")
    print(f"Difficulty distribution: {Counter(r['difficulty'] for r in selected)}")
    print(f"Platform/test_type distribution: {Counter((r['platform']) for r in selected)}")
    dates = sorted(r["contest_date"] for r in selected)
    print(f"Contest date range: {dates[0]} .. {dates[-1]}")


if __name__ == "__main__":
    main()
