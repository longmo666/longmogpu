#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${RPOW2_INSTALL_DIR:-/root}"
DEVICE="${RPOW2_DEVICE:-0}"
LANES="${RPOW2_LANES:-2}"
PREFETCH="${RPOW2_PREFETCH:-256}"
FETCHERS="${RPOW2_FETCHERS:-16}"
MINTERS="${RPOW2_MINTERS:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

require_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root, for example: sudo bash install.sh" >&2
    exit 1
  fi
}

retry() {
  local attempts="$1"
  shift
  local delay=3
  local n=1
  until "$@"; do
    if [ "$n" -ge "$attempts" ]; then
      return 1
    fi
    echo "retry $n/$attempts failed; sleeping ${delay}s: $*" >&2
    sleep "$delay"
    n=$((n + 1))
  done
}

ensure_cuda_tools() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. This installer requires an NVIDIA GPU server." >&2
    exit 1
  fi

  if command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/nvcc ]; then
    return 0
  fi

  if command -v apt-get >/dev/null 2>&1; then
    echo "nvcc not found; installing nvidia-cuda-toolkit and build-essential..."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-cuda-toolkit build-essential
    return 0
  fi

  echo "nvcc not found and apt-get is unavailable. Install CUDA toolkit manually." >&2
  exit 1
}

install_files() {
  install -m 0755 "$SCRIPT_DIR/rpow2_gpu.py" "$INSTALL_DIR/rpow2_gpu.py"
  install -m 0644 "$SCRIPT_DIR/rpow2_cuda_search.cu" "$INSTALL_DIR/rpow2_cuda_search.cu"
  install -m 0755 "$SCRIPT_DIR/build_cuda.sh" "$INSTALL_DIR/build_cuda.sh"
  install -m 0755 "$SCRIPT_DIR/rpow2_gpu_fleet.sh" "$INSTALL_DIR/rpow2_gpu_fleet.sh"
}

write_cookie() {
  local session_input="${RPOW_SESSION:-}"
  if [ -z "$session_input" ]; then
    printf "Enter rpow_session: "
    IFS= read -r session_input
  fi

  if [ -z "$session_input" ]; then
    echo "rpow_session is empty" >&2
    exit 1
  fi

  if printf '%s' "$session_input" | grep -qi 'rpow_session='; then
    printf '%s\n' "$session_input" > "$INSTALL_DIR/.rpow2_cookie"
  else
    printf 'rpow_session=%s\n' "$session_input" > "$INSTALL_DIR/.rpow2_cookie"
  fi
  chmod 600 "$INSTALL_DIR/.rpow2_cookie"
}

stop_existing() {
  pkill -TERM -f 'rpow2_gpu.py run' 2>/dev/null || true
  pkill -TERM -f 'rpow2_cuda_search' 2>/dev/null || true
  sleep 2
  pkill -KILL -f 'rpow2_gpu.py run' 2>/dev/null || true
  pkill -KILL -f 'rpow2_cuda_search' 2>/dev/null || true
  rm -f "$INSTALL_DIR/rpow2_gpu_fleet.pids" "$INSTALL_DIR"/rpow2_gpu_lane*.pid
}

main() {
  require_root
  install_files
  write_cookie
  ensure_cuda_tools

  "$INSTALL_DIR/build_cuda.sh" "$INSTALL_DIR/rpow2_cuda_search.cu" "$INSTALL_DIR/rpow2_cuda_search"

  echo "checking API login..."
  retry 5 python3 "$INSTALL_DIR/rpow2_gpu.py" preflight

  echo "checking CUDA search..."
  retry 3 "$INSTALL_DIR/rpow2_cuda_search" 00 20 --device "$DEVICE" --progress-ms 1000

  echo "stopping existing miner processes..."
  stop_existing

  : > "$INSTALL_DIR/rpow2_gpu_fleet.log"
  echo "starting miner: lanes=${LANES} prefetch=${PREFETCH} fetchers=${FETCHERS} minters=${MINTERS}"
  RPOW2_PREFETCH="$PREFETCH" \
  RPOW2_FETCHERS="$FETCHERS" \
  RPOW2_MINTERS="$MINTERS" \
    "$INSTALL_DIR/rpow2_gpu_fleet.sh" start "$LANES"

  echo
  echo "install complete"
  echo "status:  $INSTALL_DIR/rpow2_gpu_fleet.sh status"
  echo "logs:    tail -f $INSTALL_DIR/rpow2_gpu_fleet.log"
  echo "balance: $INSTALL_DIR/rpow2_gpu_fleet.sh balance"
  echo "stop:    $INSTALL_DIR/rpow2_gpu_fleet.sh stop-all"
}

main "$@"
