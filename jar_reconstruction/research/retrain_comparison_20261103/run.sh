#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/overnight_v6_v7_mvs_20261103/comparison
exec 8>"$output/queue.lock"
flock -n 8 || exit 1
trap 'status=$?; printf "%s\n" "$status" > "$output/queue.exit_code.txt"' EXIT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
official_methods/.venv/bin/python research/retrain_comparison_20261103/evaluate.py --config "$output/config.json" > "$output/accuracy.log" 2>&1
official_methods/.venv/bin/python research/retrain_comparison_20261103/speed.py --config "$output/config.json" > "$output/speed.log" 2>&1
official_methods/.venv/bin/python research/retrain_comparison_20261103/report.py --config "$output/config.json" > "$output/report.log" 2>&1
