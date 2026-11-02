#!/usr/bin/env bash
# 固定权重完整测试；配置决定数据/方法，不在测试后调参。
set -euo pipefail
cd "$(dirname "$0")/.."
output="runs/active/full_test_20261026"
mkdir -p "$output"
exec 9>"$output/evaluation.lock"
flock -n 9 || exit 1
trap 'status=$?; printf "%s\n" "$status" > "$output/exit_code.txt"' EXIT
rm -f "$output/exit_code.txt"

run_test() {
    tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
        official_methods/.venv/bin/python tools/compare_public_pretrained.py "$@"
}
run_test --config configs/full_test_20261026/public.json --smoke
run_test --config configs/full_test_20261026/carla.json --smoke
run_test --config configs/full_test_20261026/public.json
run_test --config configs/full_test_20261026/carla.json
