#!/usr/bin/env bash
# 训练后评测。按帧分给各 GPU；已有评测器统一汇总，不重复补样。
# 一次 test 同时输出全部、完整四目、缺失相机、1/2/3/4 目分组结果。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-official_methods/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if (( $# == 0 )); then
    echo '用法：bash scripts/eval_8gpu.sh checkpoint [--output 目录] [其他 eval.py 参数]' >&2
    exit 2
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo '用法：bash scripts/eval_8gpu.sh checkpoint [其他 eval.py 参数]'
    echo '默认 8 卡、test 集，读取 checkpoint 同目录的 config.json；可用 --config 指定。'
    echo 'JAR_GPUS 可改卡数。没有验证集时只跑 test；补齐后用 --split val 单独验证。'
    exec "$PYTHON" eval.py --help
fi

checkpoint="$1"
shift
if [[ ! -f "$checkpoint" ]]; then
    echo "找不到 checkpoint：$checkpoint" >&2
    exit 2
fi
config="$(dirname -- "$checkpoint")/config.json"
if [[ ! -f "$config" ]]; then
    echo 'checkpoint 旁没有 config.json，使用当前 CARLA 四目配置；其他架构请显式传 --config。' >&2
    config="configs/carla/mf_et_jar_wide4.json"
fi

# 当前没有独立验证集，默认只跑 test；不把测试数据自动改名为 val。
default_output="runs/carla_eval_$(date +%Y%m%d_%H%M%S)_$$"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${JAR_GPUS:-8}" eval.py \
    --config "$config" --device cuda --checkpoint "$checkpoint" --split test \
    --output "$default_output" "$@"
