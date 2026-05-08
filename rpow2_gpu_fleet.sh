#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
LANES="${2:-8}"
MINER="/root/rpow2_gpu.py"
FLEET_LOG="/root/rpow2_gpu_fleet.log"
PID_FILE="/root/rpow2_gpu_fleet.pids"
LEGACY_BASE="/root/rpow2_gpu"
PREFETCH="${RPOW2_PREFETCH:-256}"
FETCHERS="${RPOW2_FETCHERS:-16}"
MINTERS="${RPOW2_MINTERS:-16}"
RUN_ARGS=(
  --device 0
  --batch-iters 1024
  --progress-interval 10
  --retry-sleep 2
  --timeout 45
  --prefetch "$PREFETCH"
  --challenge-workers "$FETCHERS"
  --mint-workers "$MINTERS"
  --mint-backlog 4096
)

pid_running() {
  local pid="$1"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

fleet_pids() {
  [ -f "$PID_FILE" ] || return 0
  awk '{print $1}' "$PID_FILE"
}

running_count() {
  local count=0
  local pid
  while read -r pid; do
    if pid_running "$pid"; then
      count=$((count + 1))
    fi
  done < <(fleet_pids)
  echo "$count"
}

stop_pid() {
  local pid="$1"
  if ! pid_running "$pid"; then
    return 1
  fi

  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 40); do
    if ! pid_running "$pid"; then
      return 0
    fi
    sleep 0.25
  done
  kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  return 0
}

stop_fleet_pids() {
  local stopped=0
  local skipped=0
  local pid
  while read -r pid; do
    [ -n "$pid" ] || continue
    if stop_pid "$pid"; then
      stopped=$((stopped + 1))
    else
      skipped=$((skipped + 1))
    fi
  done < <(fleet_pids)
  rm -f "$PID_FILE"
  echo "fleet stop summary stopped=${stopped} skipped=${skipped}"
}

stop_legacy_lane_pids() {
  local stopped=0
  local skipped=0
  local pid_file
  for pid_file in "${LEGACY_BASE}"_lane*.pid; do
    [ -e "$pid_file" ] || continue
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if stop_pid "$pid"; then
      stopped=$((stopped + 1))
    else
      skipped=$((skipped + 1))
    fi
    rm -f "$pid_file"
  done
  echo "legacy stop summary stopped=${stopped} skipped=${skipped}"
}

case "$ACTION" in
  start)
    current="$(running_count)"
    if [ "$current" -gt 0 ]; then
      echo "already running lanes=${current}; use restart ${LANES} to replace"
      exit 0
    fi

    : > "$PID_FILE"
    echo "starting ${LANES} GPU mining lanes log=${FLEET_LOG} pid_file=${PID_FILE}"
    echo "internal_pipeline prefetch=${PREFETCH} challenge_workers=${FETCHERS} mint_workers=${MINTERS}"
    for i in $(seq 1 "$LANES"); do
      setsid env RPOW2_LANE="$i" python3 -u "$MINER" run "${RUN_ARGS[@]}" >> "$FLEET_LOG" 2>&1 &
      pid="$!"
      printf '%s %s\n' "$pid" "$i" >> "$PID_FILE"
      if [ $((i % 50)) -eq 0 ]; then
        echo "start progress ${i}/${LANES}"
      fi
    done
    echo "start summary started=${LANES} log=${FLEET_LOG} pid_file=${PID_FILE}"
    ;;
  stop)
    echo "stopping fleet pid_file=${PID_FILE}"
    stop_fleet_pids
    ;;
  stop-all)
    echo "stopping fleet and legacy lane pid files"
    stop_fleet_pids
    stop_legacy_lane_pids
    ;;
  restart)
    "$0" stop-all
    "$0" start "$LANES"
    ;;
  status)
    echo "running_lanes=$(running_count)"
    echo "pid_file=${PID_FILE}"
    echo "log=${FLEET_LOG}"
    if [ -f "$FLEET_LOG" ]; then
      echo "--- recent log ---"
      tail -n 40 "$FLEET_LOG"
    fi
    ;;
  count)
    running_count
    ;;
  balance)
    python3 "$MINER" preflight | grep '^me='
    ;;
  tail)
    tail -f "$FLEET_LOG"
    ;;
  clean)
    echo "cleaning old generated rpow2 logs and legacy lane pid files"
    rm -f "${LEGACY_BASE}"_lane*.pid
    rm -f /root/rpow2_gpu_lane*.log
    rm -f /root/rpow2_parallel_test.log /root/rpow2_gpu_miner.log /root/rpow2_fast_miner.log
    touch "$FLEET_LOG"
    ls -lh "$FLEET_LOG" "$PID_FILE" 2>/dev/null || true
    ;;
  *)
    echo "usage: $0 {start|stop|stop-all|restart|status|count|balance|tail|clean} [lanes]" >&2
    echo "examples:" >&2
    echo "  $0 restart 258" >&2
    echo "  $0 status" >&2
    echo "  $0 tail" >&2
    exit 2
    ;;
esac
