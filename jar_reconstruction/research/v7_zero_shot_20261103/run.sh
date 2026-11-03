#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/v7_zero_shot_20261103
exec 8>"$output/queue.lock"
flock -n 8 || exit 1
trap 'status=$?; printf "%s\n" "$status" > "$output/queue.exit_code.txt"' EXIT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TZ=Asia/Shanghai
official_methods/.venv/bin/python -u research/v7_zero_shot_20261103/run.py > "$output/accuracy.log" 2>&1
official_methods/.venv/bin/python -u research/v7_zero_shot_20261103/speed.py > "$output/speed.log" 2>&1
official_methods/.venv/bin/python -u research/v7_zero_shot_20261103/report.py > "$output/report.log" 2>&1
