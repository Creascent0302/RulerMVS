#!/usr/bin/env bash
# 参数只用于选择配置、恢复checkpoint；训练超参数全部读取JSON。
set -euo pipefail

cd "$(dirname "$0")/.."
config="${1:-configs/pretraining/train_no_hero_v2.json}"
python="official_methods/.venv/bin/python"
output="$($python -c 'import json, sys; print(json.load(open(sys.argv[1]))["output"])' "$config")"
mkdir -p "$output"

# 同一个输出目录只能有一个训练实例，避免误启动覆盖权重。
exec 9>"$output/train.lock"
flock -n 9 || { echo "该目录已有训练实例：$output"; exit 1; }
trap 'status=$?; printf "%s\n" "$status" > "$output/exit_code.txt"' EXIT
rm -f "$output/exit_code.txt"

resume=()
if [[ -f "$output/last.pt" ]]; then
    resume=(--resume "$output/last.pt")
fi

# 初始化文件已离线准备；不使用SSH代理，不允许模型隐式下载。
tools/with_server_network.sh env \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    "$python" -m torch.distributed.run --standalone --nproc_per_node=8 \
    tools/train_public.py --config "$config" "${resume[@]}"
