#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
queue=runs/active/rulermvs_v5_20261031
mkdir -p "$queue"
exec 8>"$queue/queue.lock"
flock -n 8 || { echo "V5队列已在运行"; exit 1; }
trap 'status=$?; printf "%s\n" "$status" > "$queue/queue.exit_code.txt"' EXIT

# 原始数据和大权重都位于数据盘。启动预留220GiB，保存时runner保留200GiB。
official_methods/.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
root = Path("runs/active/rulermvs_v5_20261031").resolve()
if not str(root).startswith("/data/"):
    raise RuntimeError("训练产物必须通过既有软链接写入数据盘")
if shutil.disk_usage(root).free < 220 * 2 ** 30:
    raise RuntimeError("启动前数据盘空间不足220GiB，保留已有数据并停止")
PY

train_phase() (
    config="research/rulermvs_v5_20261031/configs/train_${1}.json"
    output="$(official_methods/.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$config")"
    mkdir -p "$output"
    exec 9>"$output/train.lock"
    flock -n 9 || { echo "该目录已有训练实例：$output"; exit 1; }
    trap 'status=$?; printf "%s\n" "$status" > "$output/exit_code.txt"' EXIT
    resume=()
    if [[ -f "$output/last.pt" ]]; then
        resume=(--resume "$output/last.pt")
    fi
    tools/with_server_network.sh env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
        PYTHONUNBUFFERED=1 official_methods/.venv/bin/python \
        -m torch.distributed.run --standalone --nproc_per_node=8 \
        research/rulermvs_v5_20261031/training.py --config "$config" "${resume[@]}"
)

for phase in public carla_new; do
    echo "$(date -Is) START $phase"
    train_phase "$phase" >> "$queue/${phase}_console.log" 2>&1
    echo "$(date -Is) COMPLETE $phase"
done
echo "$(date -Is) START evaluation"
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    official_methods/.venv/bin/python research/rulermvs_v5_20261031/evaluation.py --after-training \
    >> "$queue/evaluation_console.log" 2>&1
echo "$(date -Is) COMPLETE evaluation"
