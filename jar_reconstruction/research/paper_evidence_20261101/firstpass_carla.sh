#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/rulermvs_v6_20261101/full_training/paper_firstpass
exec 9>"$output/carla_test.lock"
flock -n 9
trap 'status=$?; printf "%s\n" "$status" > "$output/carla_new_test.exit_code.txt"' EXIT
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    official_methods/.venv/bin/python research/rulermvs_v6_20261101/evaluation.py \
    --config "$output/configs/carla_new_test.json" >> "$output/carla_new_test.log" 2>&1
