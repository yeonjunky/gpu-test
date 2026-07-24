"""Prepares the perplexity ablation-study task: fixed-size, non-overlapping
chunks of WikiText-2 raw text (held-out test split), used to compute
per-token NLL / perplexity for a served model via vLLM's prompt_logprobs
(see benchmark/task_runners/run_perplexity.py) rather than a generation task.

Source: Salesforce/wikitext, config wikitext-2-raw-v1, test split -- a
standard, small, well-established perplexity benchmark corpus, distinct from
this project's Task A/B/C data so it measures something genuinely different
(next-token prediction quality, not downstream task accuracy).

Token counts use the same fixed reference tokenizer (cl100k_base) as
prepare_task_c_niah.py, for consistency across models with different native
tokenizers.

Run locally, no GPU required:
    python data/prep_scripts/prepare_perplexity_wikitext2.py
"""
import json
import random
import shutil
from pathlib import Path

import pandas as pd
import tiktoken
from huggingface_hub import hf_hub_download

DATASET_REPO_ID = "Salesforce/wikitext"
DATASET_FILENAME = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
RAW_PATH = Path("data/raw/wikitext2/test-00000-of-00001.parquet")
OUT_PATH = Path("data/prepared/perplexity_wikitext2_50.jsonl")

N_SAMPLES = 50
CHUNK_TOKENS = 1024
SEED = 42


def download_test_split() -> list[str]:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        downloaded = hf_hub_download(repo_id=DATASET_REPO_ID, repo_type="dataset", filename=DATASET_FILENAME)
        shutil.copy(downloaded, RAW_PATH)
    df = pd.read_parquet(RAW_PATH)
    return [row for row in df["text"].tolist() if row and row.strip()]


def build_chunks(rows: list[str], enc, rng: random.Random) -> list[dict]:
    """Concatenates the whole held-out split into one token stream (standard
    WikiText2-perplexity convention -- treat the split as continuous text,
    not independent documents), slices it into fixed-size non-overlapping
    windows, then samples N_SAMPLES of them so the corpus isn't
    over-represented by just its opening section."""
    full_text = "\n".join(rows)
    tokens = enc.encode(full_text)

    all_chunks = [tokens[i:i + CHUNK_TOKENS] for i in range(0, len(tokens), CHUNK_TOKENS)]
    all_chunks = [c for c in all_chunks if len(c) == CHUNK_TOKENS]  # drop the short final chunk

    rng.shuffle(all_chunks)
    selected = all_chunks[:N_SAMPLES]

    return [
        {
            "chunk_id": i,
            "text": enc.decode(chunk_tokens),
            "ref_token_count": len(chunk_tokens),
        }
        for i, chunk_tokens in enumerate(selected)
    ]


def main() -> None:
    print(f"Downloading {DATASET_REPO_ID}/{DATASET_FILENAME} ...")
    rows = download_test_split()
    print(f"Loaded {len(rows)} non-empty rows from the test split.")

    enc = tiktoken.get_encoding("cl100k_base")
    rng = random.Random(SEED)

    chunks = build_chunks(rows, enc, rng)
    print(f"Built {len(chunks)} chunks of {CHUNK_TOKENS} tokens each (from a pool of non-overlapping windows).")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Wrote {len(chunks)} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
