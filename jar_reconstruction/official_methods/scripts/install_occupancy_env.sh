#!/usr/bin/env bash
# 独立环境与官方 CUDA 构建。默认只显示计划，不修改已有环境。
set -euo pipefail
PACKAGE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_PYTHON=${BASE_PYTHON:-/data/liuzonghao_data/miniconda3/envs/simplebev/bin/python}
METHOD_ENV=${METHOD_ENV:-"${PACKAGE_ROOT}/.venv_occ"}
METHOD_ENV=$(realpath -m "${METHOD_ENV}")
PACKAGE_INDEX=${PACKAGE_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
MODE=${1:---dry-run}
case "${MODE}" in
  --help|-h)
    cat <<'EOF'
用法：install_occupancy_env.sh [--dry-run|--install|--rebuild-extensions|--check]
  --dry-run            只打印重建计划（默认）
  --install            在不存在的新 METHOD_ENV 建立隔离环境、安装依赖并编译
  --rebuild-extensions  使用已有环境，仅重编译/安装本项目官方扩展
  --check              只读 pip check、版本和编译器检查，不宣称算子测试通过
环境变量：BASE_PYTHON、METHOD_ENV、PACKAGE_INDEX、CUDA_HOME、CC、CXX、MAX_JOBS。
默认 CUDA 12.8 / gcc,g++ 10 / sm_120；无 sudo，不修改驱动或基础 Conda。
重建可较慢；GPU 前后向需另运行 tools/probe_occupancy_cuda.py。
EOF
    exit 0 ;;
  --dry-run|--install|--rebuild-extensions|--check) ;;
  *) echo "未知选项：${MODE}" >&2; exit 2 ;;
esac
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export CC=${CC:-/usr/bin/gcc-10}
export CXX=${CXX:-/usr/bin/g++-10}
export TORCH_CUDA_ARCH_LIST=12.0
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export MAX_JOBS=${MAX_JOBS:-8}
export PATH="${CUDA_HOME}/bin:${METHOD_ENV}/bin:${PATH}"
PYTHON="${METHOD_ENV}/bin/python"
MMCV_SOURCE="${PACKAGE_ROOT}/build_src/mmcv-2.0.1"
GF="${PACKAGE_ROOT}/upstream/gaussianformer"
if [[ "${MODE}" == --dry-run ]]; then
  cat <<EOF
目标环境：${METHOD_ENV}
1. --install 只允许新目录：venv + torch 2.8.0/cu128 + torchvision 0.23.0/cu128。
2. 下载/复用 mmcv-2.0.1.tar.gz，校验固定 SHA-256，解压到 ${MMCV_SOURCE}。
3. 以 patches/mmcv_201_torch28.patch 将 C++14 改为 C++17，保留算子公式。
4. 使用 ${CUDA_HOME}/bin/nvcc、${CXX}、sm_120 构建 MMCV 及安装 requirements-occupancy.txt。
5. 构建官方 GF2 deformable_aggregation / localagg_prob、OctreeOcc octree_ops。
6. 从固定 Pointcept 源码构建 pointops；spconv 使用 cu126 2.3.8 原生包。
7. pip check 与 freeze；算子前后向和整网测试需显式另运行。
现有目录不会在 --install 中被重装。--rebuild-extensions 不重新安装 Python 依赖。
EOF
  exit 0
fi
if [[ "${MODE}" == --check ]]; then
  "${PYTHON}" -m pip check
  "${PYTHON}" -c 'import sys,torch; print(sys.executable, torch.__version__, torch.version.cuda)'
  "${CUDA_HOME}/bin/nvcc" --version
  "${CXX}" --version
  exit 0
fi
if [[ "${MODE}" == --install ]]; then
  if [[ -e "${METHOD_ENV}" ]]; then
    echo "保留已有目录 ${METHOD_ENV}；新建环境请显式设置其他 METHOD_ENV。" >&2
    exit 2
  fi
  "${BASE_PYTHON}" -m venv --copies "${METHOD_ENV}"
  "${PYTHON}" -m pip install --index-url "${PACKAGE_INDEX}" setuptools==80.9.0 wheel==0.47.0 ninja==1.13.2
  "${PYTHON}" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  "${PYTHON}" -m pip install --index-url "${PACKAGE_INDEX}" numpy==1.26.4 opencv-python==4.11.0.86 mmengine==0.10.7
fi
"${PYTHON}" -c 'import torch; assert torch.__version__ == "2.8.0+cu128", torch.__version__'
"${CUDA_HOME}/bin/nvcc" --version | grep -q 'release 12.8'
[[ -x "${CXX}" && -x "${CC}" ]]
mkdir -p "${PACKAGE_ROOT}/packages" "${PACKAGE_ROOT}/build_src" "${PACKAGE_ROOT}/artifacts"
MMCV_ARCHIVE="${PACKAGE_ROOT}/packages/mmcv-2.0.1.tar.gz"
if [[ ! -f "${MMCV_ARCHIVE}" ]]; then
  "${PYTHON}" -m pip download --no-deps --no-binary=:all: --index-url "${PACKAGE_INDEX}" \
    --dest "${PACKAGE_ROOT}/packages" mmcv==2.0.1
fi
"${PYTHON}" - "${MMCV_ARCHIVE}" <<'PY'
import hashlib,sys
assert hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest() == '2b9b2c23795b3f43cf2ebcaca46552d9b4ef7e903eacdcf7c08ee15195044071', 'MMCV 源码包 SHA-256 不符'
PY
if [[ ! -d "${MMCV_SOURCE}" ]]; then
  tar -xzf "${MMCV_ARCHIVE}" -C "${PACKAGE_ROOT}/build_src"
fi
if ! git -C "${MMCV_SOURCE}" apply --reverse --check "${PACKAGE_ROOT}/patches/mmcv_201_torch28.patch" 2>/dev/null; then
  git -C "${MMCV_SOURCE}" apply --check "${PACKAGE_ROOT}/patches/mmcv_201_torch28.patch"
  git -C "${MMCV_SOURCE}" apply "${PACKAGE_ROOT}/patches/mmcv_201_torch28.patch"
fi
# 构建原始完整 MMCV 扩展，并保留本地源码 .so，符合适配器显式加载路径。
(
  cd "${MMCV_SOURCE}"
  "${PYTHON}" setup.py build_ext --inplace
)
if [[ "${MODE}" == --install ]]; then
  "${PYTHON}" -m pip install --no-build-isolation --no-deps "${MMCV_SOURCE}"
  "${PYTHON}" -m pip install --index-url "${PACKAGE_INDEX}" -r "${PACKAGE_ROOT}/requirements-occupancy.txt"
fi
for SOURCE in "${GF}/model/encoder/gaussian_encoder/ops" "${GF}/model/head/localagg_prob" "${PACKAGE_ROOT}/upstream/octreeocc"; do
  [[ -f "${SOURCE}/setup.py" ]] || { echo "缺少官方源码：先运行 fetch_upstream.py --apply-patches" >&2; exit 1; }
  (
    cd "${SOURCE}"
    "${PYTHON}" setup.py build_ext --inplace
  )
done
"${PYTHON}" -m pip install --no-build-isolation --no-deps "${PACKAGE_ROOT}/upstream/pointcept/libs/pointops"
"${PYTHON}" -m pip check
"${PYTHON}" -m pip freeze > "${PACKAGE_ROOT}/artifacts/occupancy_environment.freeze.txt"
echo "构建结束；下一步必须执行官方 CUDA 前后向探针及整网合成测试，不能只以导入成功验收。"
