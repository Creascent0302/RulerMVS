#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/v2_evaluation_20261029
python=official_methods/.venv/bin/python
trap 'status=$?; printf "%s\n" "$status" > "$output/exit_code.txt"' EXIT
for group in public carla carla_other_refs; do
    echo "$(date -Is) SMOKE $group"
    "$python" research/near_mid_evaluation_20261028/evaluate.py \
        --config "research/v2_evaluation_20261029/${group}.json" --smoke \
        > "$output/${group}_smoke.log" 2>&1
done
for group in public carla carla_other_refs; do
    echo "$(date -Is) TEST $group"
    "$python" research/near_mid_evaluation_20261028/evaluate.py \
        --config "research/v2_evaluation_20261029/${group}.json" \
        > "$output/${group}_test.log" 2>&1
done
# 全部精度进程退出以后才独占GPU0计时，禁止用并行评测时长冒充推理延迟。
for group in public carla; do
    echo "$(date -Is) SPEED $group"
    "$python" tools/benchmark_public_speed.py \
        --config "research/v2_evaluation_20261029/${group}_speed.json" \
        > "$output/${group}_speed.log" 2>&1
done
echo "$(date -Is) COMPLETE"
