#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/rulermvs_v6_20261101/full_training/paper_fast_test
exec 9>"$output/run.lock"
flock -n 9
trap 'status=$?; printf "%s\n" "$status" > "$output/exit_code.txt"' EXIT
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    official_methods/.venv/bin/python research/paper_evidence_20261101/fast_test.py \
    >> "$output/run.log" 2>&1
