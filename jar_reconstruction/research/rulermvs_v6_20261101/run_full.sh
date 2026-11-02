#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

queue=runs/active/rulermvs_v6_20261101/full_training
mkdir -p "$queue"
exec 8>"$queue/queue.lock"
flock -n 8 || { echo "V6完整训练队列已在运行"; exit 1; }
trap 'status=$?; printf "%s\n" "$status" > "$queue/queue.exit_code.txt"' EXIT

official_methods/.venv/bin/python - <<'PY'
from pathlib import Path
import shutil

root = Path("runs/active/rulermvs_v6_20261101/full_training").resolve()
if not root.is_relative_to(Path("/data")):
    raise RuntimeError("训练产物必须经既有软链接存入数据盘")
if shutil.disk_usage(root).free < 220 * 2 ** 30:
    raise RuntimeError("启动预留220GiB不足；保存时仍须保留200GiB")
PY

train_phase() (
    config="research/rulermvs_v6_20261101/configs/train_${1}.json"
    output="$(official_methods/.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$config")"
    mkdir -p "$output"
    exec 9>"$output/train.lock"
    flock -n 9 || { echo "该训练目录已被占用：$output"; exit 1; }
    trap 'status=$?; printf "%s\n" "$status" > "$output/exit_code.txt"' EXIT
    resume=()
    if [[ -f "$output/last.pt" ]]; then
        resume=(--resume "$output/last.pt")
    fi
    tools/with_server_network.sh env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
        PYTHONUNBUFFERED=1 official_methods/.venv/bin/python \
        -m torch.distributed.run --standalone --nproc_per_node=8 \
        research/rulermvs_v6_20261101/training.py --config "$config" "${resume[@]}"
)

for phase in public carla_new; do
    echo "$(date -Is) START $phase"
    train_phase "$phase" >> "$queue/${phase}_console.log" 2>&1
    echo "$(date -Is) COMPLETE $phase"
done

echo "$(date -Is) START fixed VAL/TEST comparison"
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    official_methods/.venv/bin/python research/rulermvs_v6_20261101/post_training.py \
    --config research/rulermvs_v6_20261101/configs/evaluation_full.json \
    >> "$queue/evaluation_console.log" 2>&1
echo "$(date -Is) COMPLETE fixed VAL/TEST comparison"
