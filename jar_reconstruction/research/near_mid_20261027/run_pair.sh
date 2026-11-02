#!/usr/bin/env bash
# 两组依次使用全部八卡：新表示及旧结构同预算续训。
# 训练参数只来自各自JSON；重复启动会从已有last.pt恢复。
set -euo pipefail
cd "$(dirname "$0")/../.."
output="runs/active/near_mid_20261027"
mkdir -p "$output"
exec 8>"$output/pair.lock"
flock -n 8 || { echo "该对照队列已经运行"; exit 1; }
trap 'status=$?; printf "%s\n" "$status" > "$output/pair_exit_code.txt"' EXIT
rm -f "$output/pair_exit_code.txt"
for variant in surface control; do
    echo "$(date -Is) 开始或恢复 $variant"
    bash research/near_mid_20261027/run_8gpu.sh \
        "research/near_mid_20261027/configs/train_${variant}.json" \
        >> "$output/${variant}_training_console.log" 2>&1
    echo "$(date -Is) 完成 $variant"
done
