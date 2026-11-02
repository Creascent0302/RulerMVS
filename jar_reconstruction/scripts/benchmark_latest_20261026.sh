#!/usr/bin/env bash
# 等待全量精度任务释放GPU，随后在同一张卡上公平计时。
set -euo pipefail
cd "$(dirname "$0")/.."
output="runs/active/full_test_20261026"
exec 9>"$output/evaluation.lock"
flock 9
trap 'status=$?; printf "%s\n" "$status" > "$output/timing_exit_code.txt"' EXIT
[[ "$(cat "$output/exit_code.txt")" == 0 ]]

for group in public carla; do
    tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
        official_methods/.venv/bin/python tools/benchmark_public_speed.py \
        --config "configs/full_test_20261026/timing_${group}.json"
done
official_methods/.venv/bin/python tools/report_full_test.py \
    --config configs/full_test_20261026/report.json
