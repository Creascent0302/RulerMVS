#!/usr/bin/env bash
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin=official_methods/.venv/bin/python
root=runs/active/early_geometry_20261030
pids=()
for arm in matching_geometry matching_constant; do
  "$python_bin" research/early_geometry_20261030/pilot.py \
    --config "research/early_geometry_20261030/configs/public_${arm}.json" \
    > "$root/logs/public_${arm}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
echo "$failed" > "$root/logs/matching_training.exit_code.txt"
if [ "$failed" -ne 0 ]; then exit "$failed"; fi
"$python_bin" research/early_geometry_20261030/summarize_matching.py
"$python_bin" research/early_geometry_20261030/latency.py \
  --config research/early_geometry_20261030/configs/latency_matching.json
echo 0 > "$root/logs/matching_queue.exit_code.txt"
