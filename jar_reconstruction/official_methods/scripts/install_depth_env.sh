#!/usr/bin/env bash
# 在独立目录写入包；既有 Conda 环境只读，不更改驱动。
set -euo pipefail
PACKAGE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_PYTHON=${BASE_PYTHON:-/data/liuzonghao_data/miniconda3/envs/simplebev/bin/python}
METHOD_ENV=${METHOD_ENV:-"${PACKAGE_ROOT}/.venv"}
PACKAGE_INDEX=${PACKAGE_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
if [[ ! -x "${METHOD_ENV}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --copies --without-pip "${METHOD_ENV}"
  # 本服务器已验证 torch/cu128 的独立副本，不使用 system-site-packages 或软链接共享包。
  SOURCE_SITE=$("${BASE_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')
  TARGET_SITE=$("${METHOD_ENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  cp -a --reflink=auto "${SOURCE_SITE}/." "${TARGET_SITE}/"
fi
"${METHOD_ENV}/bin/python" -c 'import torch; assert torch.__version__ == "2.8.0+cu128", torch.__version__'
"${METHOD_ENV}/bin/python" -m pip install --index-url "${PACKAGE_INDEX}" -r "${PACKAGE_ROOT}/requirements-depth.txt"
"${METHOD_ENV}/bin/python" -m pip check
"${METHOD_ENV}/bin/python" -m pip freeze > "${PACKAGE_ROOT}/artifacts/depth_environment.freeze.txt"
