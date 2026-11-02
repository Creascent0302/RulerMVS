#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/v2_retest_20261030
trap 'status=$?; printf "%s\n" "$status" > "${output}.exit_code.txt"' EXIT
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    official_methods/.venv/bin/python research/v2_retest_20261030/run.py \
    > "${output}_console.log" 2>&1
