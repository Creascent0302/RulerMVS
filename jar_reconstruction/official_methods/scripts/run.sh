#!/usr/bin/env bash
# 例：bash official_methods/scripts/run.sh fast_mvsnet train --synthetic --config ...
set -euo pipefail
PACKAGE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "${PACKAGE_ROOT}/.." && pwd)
METHOD=${1:?需要方法名}
COMMAND=${2:?需要 train/infer/evaluate}
shift 2
METHOD_ENV="${PACKAGE_ROOT}/.venv"
if [[ "${METHOD}" == "gaussianformer2" || "${METHOD}" == "octreeocc" ]]; then
  METHOD_ENV="${PACKAGE_ROOT}/.venv_occ"
fi
cd "${PROJECT_ROOT}"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${METHOD_ENV}/bin:${PATH}"
export CXX=/usr/bin/g++-10
export CC=/usr/bin/gcc-10
export TORCH_CUDA_ARCH_LIST=12.0
if [[ "${COMMAND}" == "train" && "${GPUS:-1}" != "1" ]]; then
  exec "${METHOD_ENV}/bin/python" -m torch.distributed.run --standalone --nproc_per_node="${GPUS}" \
    -m official_methods.cli "${COMMAND}" --config "official_methods/methods/${METHOD}/configs/carla.json" "$@"
else
  exec "${METHOD_ENV}/bin/python" -m official_methods.cli "${COMMAND}" \
    --config "official_methods/methods/${METHOD}/configs/carla.json" "$@"
fi
