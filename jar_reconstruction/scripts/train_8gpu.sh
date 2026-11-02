#!/usr/bin/env bash
# 单机多卡训练。学习率、总步数、每卡 batch 等只修改 JSON。
# 示例：bash scripts/train_8gpu.sh --output runs/carla_experiment
# 选卡：CUDA_VISIBLE_DEVICES=0,1 JAR_GPUS=2 bash scripts/train_8gpu.sh ...
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-official_methods/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo '默认使用 CARLA 四目配置和 8 张 GPU；JAR_GPUS 可改卡数，PYTHON 可改解释器。'
    echo '不传 --output 时创建带时间戳的运行目录；续训使用 --resume 和原输出目录。'
    exec "$PYTHON" train.py --help
fi

# 不复用固定输出目录，避免新实验覆盖已有权重。--output 可显式指定目录。
default_output="runs/carla_wide4_$(date +%Y%m%d_%H%M%S)_$$"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${JAR_GPUS:-8}" train.py \
    --config configs/carla/mf_et_jar_wide4_train_8gpu.json --device cuda \
    --output "$default_output" "$@"
