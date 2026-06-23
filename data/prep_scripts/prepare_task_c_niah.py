"""Prepares Task C (Needle in a Haystack) test cases.

Builds a ~20-30k token haystack by concatenating public-domain Paul Graham
essays, then generates 50 distinct synthetic "needle" facts inserted at 50
depths spanning 0%-100%. Each of the 50 samples is its OWN haystack copy with
exactly one needle inserted -- this is the standard NIAH convention (one
needle per haystack copy), avoiding interference between needles that would
occur if all 50 were stuffed into a single document.

Token counts use a fixed reference tokenizer (cl100k_base) for consistency
across all 4 target models, which have different native tokenizers --
documented as a methodology note for the report.

Run locally, no GPU required:
    python data/prep_scripts/prepare_task_c_niah.py
"""
import csv
import io
import json
import random
import shutil
import string
from pathlib import Path

import tiktoken
from huggingface_hub import hf_hub_download

ESSAYS_REPO_ID = "sgoel9/paul_graham_essays"
ESSAYS_FILENAME = "pual_graham_essays.csv"
RAW_PATH = Path("data/raw/haystack_source/paul_graham_essays.csv")
OUT_PATH = Path("data/prepared/task_c_niah_50.jsonl")

N_SAMPLES = 50
TARGET_TOKENS_MIN = 20_000
TARGET_TOKENS_MAX = 30_000
SEED = 42

QUESTION_TEMPLATE = (
    "What is the secret admin override code mentioned in the document above? "
    "Respond with only the code, nothing else."
)


def download_essays() -> list[str]:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        downloaded = hf_hub_download(
            repo_id=ESSAYS_REPO_ID, repo_type="dataset", filename=ESSAYS_FILENAME
        )
        shutil.copy(downloaded, RAW_PATH)
    text = RAW_PATH.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return [row["text"] for row in reader if row.get("text")]


def build_haystack(essays: list[str], enc, rng: random.Random) -> str:
    """Concatenate shuffled essays until the token count lands in the target range."""
    shuffled = essays[:]
    rng.shuffle(shuffled)
    parts: list[str] = []
    total_tokens = 0
    for essay in shuffled:
        parts.append(essay)
        total_tokens += len(enc.encode(essay))
        if total_tokens >= TARGET_TOKENS_MIN:
            break
    haystack = "\n\n".join(parts)
    if total_tokens > TARGET_TOKENS_MAX:
        # truncate by token count, not char count, to stay within range
        tokens = enc.encode(haystack)[:TARGET_TOKENS_MAX]
        haystack = enc.decode(tokens)
    return haystack


def random_secret(rng: random.Random, idx: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    part1 = "".join(rng.choices(alphabet, k=4))
    part2 = "".join(rng.choices(alphabet, k=4))
    return f"{part1}-{part2}-{idx:03d}"


def insert_needle_at_depth(haystack_tokens: list[int], needle_tokens: list[int], depth_pct: float, enc) -> str:
    insert_at = int(len(haystack_tokens) * (depth_pct / 100))
    insert_at = max(0, min(insert_at, len(haystack_tokens)))
    new_tokens = haystack_tokens[:insert_at] + needle_tokens + haystack_tokens[insert_at:]
    return enc.decode(new_tokens)


def main() -> None:
    print("Downloading Paul Graham essays ...")
    essays = download_essays()
    print(f"Loaded {len(essays)} essays.")

    enc = tiktoken.get_encoding("cl100k_base")
    rng = random.Random(SEED)

    haystack = build_haystack(essays, enc, rng)
    haystack_tokens = enc.encode(haystack)
    print(f"Built haystack: {len(haystack_tokens)} tokens (target {TARGET_TOKENS_MIN}-{TARGET_TOKENS_MAX}).")

    depths = [round(i * 100 / (N_SAMPLES - 1), 2) for i in range(N_SAMPLES)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for i, depth_pct in enumerate(depths):
            secret = random_secret(rng, i)
            needle = f"\n\n[INFO] The secret admin override code is {secret}.\n\n"
            needle_tokens = enc.encode(needle)
            text_with_needle = insert_needle_at_depth(haystack_tokens, needle_tokens, depth_pct, enc)

            sample = {
                "sample_id": f"niah_{i:02d}",
                "depth_pct": depth_pct,
                "needle": needle.strip(),
                "secret_value": secret,
                "haystack_with_needle": text_with_needle,
                "question": QUESTION_TEMPLATE,
                "haystack_token_count_cl100k": len(haystack_tokens) + len(needle_tokens),
            }
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {N_SAMPLES} samples to {OUT_PATH}")
    print(f"Depths: {depths[0]}% .. {depths[-1]}%")


if __name__ == "__main__":
    main()
