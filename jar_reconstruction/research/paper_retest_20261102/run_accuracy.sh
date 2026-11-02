#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/paper_retest_20261102
exec 8>"$output/queue.lock"
flock -n 8 || exit 1
trap 'status=$?; printf "%s\n" "$status" > "$output/accuracy.exit_code.txt"' EXIT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
for group in carla eth3d public4 public2; do
    official_methods/.venv/bin/python research/paper_retest_20261102/evaluate.py \
        --config "$output/configs/$group.json" > "$output/$group.log" 2>&1
done
