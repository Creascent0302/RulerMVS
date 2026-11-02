#!/usr/bin/env bash
# 独立训练入口；variant=diffmvs/casdiffmvs 在配置文件中选择。
set -euo pipefail
METHOD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${METHOD_ROOT}/../../scripts/run.sh" diffmvs train "$@"
