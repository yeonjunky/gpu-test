"""Downloads model weights from Hugging Face onto the remote H100 box.

Reads model definitions from configs/run_matrix.yaml so there is a single
source of truth for which HF repos this project depends on.

Usage:
    python setup/download_models.py --model qwen2.5-32b
    python setup/download_models.py --all
"""
import argparse
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError


def load_models(config_path: str) -> list[dict]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["models"]


def check_access(repo_id: str, token: str | None) -> None:
    api = HfApi(token=token)
    try:
        api.model_info(repo_id)
    except GatedRepoError:
        print(
            f"ERROR: '{repo_id}' is a gated repo and your HF_TOKEN does not yet have "
            f"approved access. Request access at https://huggingface.co/{repo_id} "
            f"and wait for approval (can take hours to a few days) before retrying.",
            file=sys.stderr,
        )
        sys.exit(1)
    except HfHubHTTPError as e:
        print(f"ERROR: could not reach HF for '{repo_id}': {e}", file=sys.stderr)
        sys.exit(1)


def download(model_entry: dict, cache_dir: str, token: str | None) -> None:
    repo_id = model_entry["hf_repo"]
    print(f"=== Downloading {repo_id} ({model_entry['id']}) ===")
    if model_entry.get("gated"):
        check_access(repo_id, token)
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        token=token,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "*.model",
            "tokenizer*",
        ],
    )
    print(f"=== Done: {repo_id} ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model id from run_matrix.yaml (e.g. qwen2.5-32b)")
    parser.add_argument("--all", action="store_true", help="download every model in run_matrix.yaml")
    parser.add_argument("--config", default="configs/run_matrix.yaml")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("specify --model <id> or --all")

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set in environment. Gated repos (Llama-3.3-70B) will fail.", file=sys.stderr)

    cache_dir = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    models = load_models(args.config)
    targets = models if args.all else [m for m in models if m["id"] == args.model]
    if not targets:
        print(f"ERROR: no model with id '{args.model}' found in {args.config}", file=sys.stderr)
        sys.exit(1)

    for model_entry in targets:
        download(model_entry, cache_dir, token)


if __name__ == "__main__":
    main()
