#!/usr/bin/env bash
# 固定权重：先通过全部域的 VAL 首/中/末样本，再完成全 TEST。
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction
output="runs/active/ruler_flow_v2_20261031/verification/equal_pixels"
mkdir -p "$output"
exec >> "$output/queue.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$output/exit_code"' EXIT
date -u '+%Y-%m-%dT%H:%M:%SZ START equal_pixels'

for mode in smoke full; do
    for group in public carla carla_other_refs; do
        date -u "+%Y-%m-%dT%H:%M:%SZ START $mode $group"
        arguments=(--config "research/ruler_flow_v2_20261031/configs/verification/equal_pixels_$group.json")
        if [[ "$mode" == smoke ]]; then
            arguments+=(--smoke)
        fi
        tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
            official_methods/.venv/bin/python research/ruler_flow_v2_20261031/evaluation.py \
            "${arguments[@]}" > "$output/${mode}_${group}.log" 2>&1
        date -u "+%Y-%m-%dT%H:%M:%SZ COMPLETE $mode $group"
    done
done
date -u '+%Y-%m-%dT%H:%M:%SZ COMPLETE equal_pixels'
