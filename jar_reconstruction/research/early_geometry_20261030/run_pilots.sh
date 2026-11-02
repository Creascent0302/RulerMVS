#!/usr/bin/env bash
set -uo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction

mode="${1:-formal}"
case "$mode" in
  smoke) prefix="smoke_" ;;
  formal) prefix="" ;;
  *) echo "用法: run_pilots.sh smoke|formal" >&2; exit 2 ;;
esac

root=runs/active/early_geometry_20261030
mkdir -p "$root/logs"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

worker() {
  local name code
  for name in "$@"; do
    official_methods/.venv/bin/python research/early_geometry_20261030/pilot.py \
      --config "research/early_geometry_20261030/configs/${prefix}${name}.json" \
      > "$root/logs/${prefix}${name}.log" 2>&1
    code=$?
    echo "$code" > "$root/logs/${prefix}${name}.exit_code.txt"
    if [ "$code" -ne 0 ]; then return "$code"; fi
  done
}

pids=()
worker public_control & pids+=("$!")
worker public_appearance & pids+=("$!")
worker public_early & pids+=("$!")
worker public_late & pids+=("$!")
worker public_both & pids+=("$!")
worker carla_control carla_early & pids+=("$!")
worker carla_late carla_both & pids+=("$!")
worker carla_appearance & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
echo "$failed" > "$root/logs/${mode}_queue.exit_code.txt"
exit "$failed"
