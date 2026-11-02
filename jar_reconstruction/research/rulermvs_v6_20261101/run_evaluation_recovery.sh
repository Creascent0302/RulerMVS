#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

output=runs/active/rulermvs_v6_20261101/full_training/evaluation_recovery
exec 8>"$output/queue.lock"
flock -n 8 || { echo "V6恢复评测已运行"; exit 1; }
trap 'status=$?; printf "%s\n" "$status" > "$output/queue.exit_code.txt"' EXIT

run_python() {
    tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
        official_methods/.venv/bin/python "$@"
}

# 仅改变执行顺序，让主要CARLA结果先产生；数据/权重/指标定义不变。
for split in val test; do
    echo "$(date -Is) START CARLA $split"
    run_python research/rulermvs_v6_20261101/evaluation.py \
        --config "$output/configs/carla_new_main_${split}.json" \
        >> "$output/carla_priority_${split}.log" 2>&1
    echo "$(date -Is) COMPLETE CARLA $split"
done

# 已完成的同协议分片会缓存命中；全部精度评测结束后再独占计时。
run_python research/rulermvs_v6_20261101/post_training.py \
    --config research/rulermvs_v6_20261101/configs/evaluation_recovery.json \
    >> "$output/full_comparison.log" 2>&1
echo "$(date -Is) COMPLETE ALL"
