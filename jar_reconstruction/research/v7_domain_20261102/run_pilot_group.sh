#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

group="$1"
case "$group" in
    control_original|control_augment|projective_original|projective_augment|prior_original|prior_augment) ;;
    *) echo "未知实验组：$group"; exit 2 ;;
esac
output="runs/active/v7_domain_20261102/pilots_round1/$group"
mkdir -p "$output"
exec 8>"$output/queue.lock"
flock -n 8 || exit 1
trap 'status=$?; printf "%s\n" "$status" > "$output/queue.exit_code.txt"' EXIT

for phase in public carla; do
    tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
        official_methods/.venv/bin/python research/v7_domain_20261102/train.py \
        --config "research/v7_domain_20261102/configs/${group}_${phase}.json" \
        > "$output/${phase}.log" 2>&1
done
