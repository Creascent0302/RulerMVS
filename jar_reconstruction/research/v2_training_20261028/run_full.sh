#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/v2_training_20261028
mkdir -p "$output"
trap 'status=$?; printf "%s\n" "$status" > "$output/queue.exit_code.txt"' EXIT
rm -f "$output/queue.exit_code.txt"

# 先完成公共全域再训练，再由其合格导出权重适配新CARLA；任一失败则停止。
for name in public carla_new; do
    echo "$(date -Is) START $name"
    bash research/v2_training_20261028/run_8gpu.sh \
        "research/v2_training_20261028/configs/train_${name}.json" \
        >> "$output/${name}_console.log" 2>&1
    echo "$(date -Is) COMPLETE $name"
done
