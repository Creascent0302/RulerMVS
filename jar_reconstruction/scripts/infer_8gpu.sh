#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if (( $# < 1 )); then
    echo '用法：bash scripts/infer_8gpu.sh checkpoint [其他参数]' >&2
    exit 2
fi
checkpoint=$1
shift
PYTHON=${PYTHON:-official_methods/.venv/bin/python}
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${JAR_GPUS:-8}" infer.py \
    --config configs/gpu/mf_et_jar.json --device cuda --checkpoint "$checkpoint" \
    --output runs/mf_et_jar_inference_8gpu "$@"
