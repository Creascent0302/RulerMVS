#!/usr/bin/env bash
# 独立训练入口；所有超参数保存在 --config 指定的文件中。
set -euo pipefail
METHOD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${METHOD_ROOT}/../../scripts/run.sh" patchmatchnet train "$@"
