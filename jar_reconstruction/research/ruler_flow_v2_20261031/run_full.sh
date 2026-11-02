#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/ruler_flow_v2_20261031
mkdir -p "$output"
exec 8>"$output/queue.lock"
flock -n 8 || { echo "本轮队列已运行"; exit 1; }
trap 'status=$?; printf "%s\n" "$status" > "$output/queue.exit_code.txt"' EXIT
for name in public carla_new; do
    echo "$(date -Is) START $name"
    bash research/ruler_flow_v2_20261031/run_8gpu.sh \
        "research/ruler_flow_v2_20261031/configs/train_${name}.json" \
        >> "$output/${name}_console.log" 2>&1
    echo "$(date -Is) COMPLETE $name"
done
echo "$(date -Is) START evaluation"
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    official_methods/.venv/bin/python research/ruler_flow_v2_20261031/after_training.py \
    >> "$output/evaluation_console.log" 2>&1
echo "$(date -Is) COMPLETE evaluation"
