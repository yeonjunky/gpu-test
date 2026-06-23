"""Sanity-checks the remote H100 environment and records provenance info.

Run on the remote GPU box after `setup/install.sh`. Writes results/env_info.json
so the final report can cite exact library/hardware versions for reproducibility.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path


def check_gpu() -> dict:
    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed.", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False. No GPU visible to PyTorch.", file=sys.stderr)
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    info = {
        "gpu_name": props.name,
        "gpu_total_vram_gb": round(props.total_memory / (1024 ** 3), 1),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    print(f"GPU: {info['gpu_name']}  VRAM: {info['gpu_total_vram_gb']} GB")
    if "H100" not in props.name and "H800" not in props.name:
        print(f"WARNING: expected an H100, detected '{props.name}'. Continuing anyway.")
    return info


def check_package_versions() -> dict:
    versions = {}
    for pkg in ["vllm", "bitsandbytes", "transformers", "accelerate", "huggingface_hub"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = None
            print(f"WARNING: package '{pkg}' is not importable.", file=sys.stderr)
    for k, v in versions.items():
        print(f"{k}: {v}")
    return versions


def check_disk_space(path: str = ".") -> dict:
    total, used, free = shutil.disk_usage(path)
    free_gb = round(free / (1024 ** 3), 1)
    print(f"Free disk space at '{path}': {free_gb} GB")
    if free_gb < 500:
        print(
            "WARNING: less than 500GB free. The 4 models at native precision need "
            "~360GB combined; budget more if the bnb config-patch fallback is needed.",
            file=sys.stderr,
        )
    return {"free_disk_gb": free_gb}


def main() -> None:
    info = {}
    info["gpu"] = check_gpu()
    info["packages"] = check_package_versions()
    info["disk"] = check_disk_space()

    try:
        nvidia_smi_out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        ).stdout
        info["nvidia_smi_raw"] = nvidia_smi_out
    except Exception as e:
        print(f"WARNING: could not run nvidia-smi: {e}", file=sys.stderr)

    out_path = Path("results")
    out_path.mkdir(exist_ok=True)
    (out_path / "env_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nWrote {out_path / 'env_info.json'}")


if __name__ == "__main__":
    main()
