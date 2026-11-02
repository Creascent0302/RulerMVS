#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/paper_retest_20261102
trap 'status=$?; printf "%s\n" "$status" > "$output/queue.exit_code.txt"' EXIT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
bash research/paper_retest_20261102/run_accuracy.sh
for group in carla eth3d public4 public2; do
    official_methods/.venv/bin/python research/paper_retest_20261102/speed.py \
        --config "$output/configs/$group.json" > "$output/speed_$group.log" 2>&1
done
official_methods/.venv/bin/python research/paper_retest_20261102/report.py \
    > "$output/report.log" 2>&1
