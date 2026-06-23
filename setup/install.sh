#!/usr/bin/env bash
# One-time bootstrap for the remote H100 box. Idempotent.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v nvidia-smi &> /dev/null; then
  echo "ERROR: nvidia-smi not found. This script must run on a box with an NVIDIA GPU." >&2
  exit 1
fi

echo "=== GPU detected ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
  echo "ERROR: Python >= 3.10 required." >&2
  exit 1
fi

if [ ! -d venv ]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements-remote.txt

echo "=== Verifying environment ==="
python setup/verify_env.py

echo "=== Setup complete. Activate with: source venv/bin/activate ==="
