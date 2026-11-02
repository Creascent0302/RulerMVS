#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

output=runs/active/rulermvs_v6_20261101/full_training
mkdir -p "$output"
exec 8>"$output/queue.lock"
flock -n 8 || { echo "V6完整队列或冒烟正在运行"; exit 1; }
trap 'status=$?; printf "%s\n" "$status" > "$output/ddp_smoke.exit_code.txt"' EXIT

for phase in public carla_new; do
    tools/with_server_network.sh env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
        PYTHONUNBUFFERED=1 official_methods/.venv/bin/python \
        -m torch.distributed.run --standalone --nproc_per_node=8 \
        research/rulermvs_v6_20261101/training.py \
        --config "research/rulermvs_v6_20261101/configs/smoke_${phase}.json" \
        > "$output/ddp_smoke_${phase}.log" 2>&1
done
