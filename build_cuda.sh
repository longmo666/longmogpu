#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/root/rpow2_cuda_search.cu}"
OUT="${2:-/root/rpow2_cuda_search}"

NVCC_BIN="${NVCC:-}"
if [ -z "$NVCC_BIN" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
  elif [ -x /usr/local/cuda/bin/nvcc ]; then
    NVCC_BIN="/usr/local/cuda/bin/nvcc"
  fi
fi

if [ -z "$NVCC_BIN" ]; then
  echo "nvcc not found. Install CUDA toolkit first, then rerun this script." >&2
  echo "Ubuntu example: apt update && apt install -y nvidia-cuda-toolkit build-essential" >&2
  exit 1
fi

"$NVCC_BIN" -O3 -std=c++17 -o "$OUT" "$SRC"
chmod +x "$OUT"
echo "built $OUT"
