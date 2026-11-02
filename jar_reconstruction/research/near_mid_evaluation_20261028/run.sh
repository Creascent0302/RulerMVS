#!/usr/bin/env bash
set -u
cd /home/liuzonghao/3Dscene/jar_reconstruction
output=runs/active/near_mid_evaluation_20261028
mkdir -p "$output"
official_methods/.venv/bin/python research/near_mid_evaluation_20261028/evaluate.py \
    --config research/near_mid_evaluation_20261028/config.json > "$output/full_test.log" 2>&1
status=$?
echo "$status" > "$output/full_test.exit_code"
if [ "$status" -ne 0 ]; then
    exit "$status"
fi
# 精度任务全部退出后单卡交替计时，避免GPU并行争用。
official_methods/.venv/bin/python tools/benchmark_public_speed.py \
    --config research/near_mid_evaluation_20261028/speed.json > "$output/speed.log" 2>&1
status=$?
echo "$status" > "$output/speed.exit_code"
exit "$status"
