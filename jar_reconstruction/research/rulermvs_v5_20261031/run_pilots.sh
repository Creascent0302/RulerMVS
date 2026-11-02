#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-pilot}"
if [[ "$mode" != pilot && "$mode" != smoke ]]; then
    echo "用法：run_pilots.sh [pilot|smoke]"
    exit 2
fi
queue="runs/active/rulermvs_v5_20261031/${mode}_logs"
mkdir -p "$queue"
pids=()
gpu=0
for phase in public carla_new; do
    for variant in control layer1 layer12; do
        name="${phase}_${variant}"
        (
            trap 'status=$?; printf "%s\n" "$status" > "$queue/${name}.exit_code.txt"' EXIT
            tools/with_server_network.sh env CUDA_VISIBLE_DEVICES="$gpu" \
                HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
                PYTHONUNBUFFERED=1 official_methods/.venv/bin/python \
                research/rulermvs_v5_20261031/pilot.py \
                --config "research/rulermvs_v5_20261031/configs/${mode}_${name}.json"
        ) > "$queue/${name}.log" 2>&1 &
        pids+=("$!")
        gpu=$((gpu + 1))
    done
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
