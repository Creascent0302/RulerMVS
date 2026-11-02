#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
config="${1:-research/v2_training_20261028/configs/train_public.json}"
python="official_methods/.venv/bin/python"
output="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$config")"
mkdir -p "$output"
exec 9>"$output/train.lock"
flock -n 9 || { echo "该训练目录已有实例：$output"; exit 1; }
trap 'status=$?; printf "%s\n" "$status" > "$output/exit_code.txt"' EXIT
rm -f "$output/exit_code.txt"
resume=()
if [[ -f "$output/last.pt" ]]; then
    resume=(--resume "$output/last.pt")
fi
tools/with_server_network.sh env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 "$python" -m torch.distributed.run --standalone --nproc_per_node=8 \
    research/v2_training_20261028/training.py --config "$config" "${resume[@]}"
