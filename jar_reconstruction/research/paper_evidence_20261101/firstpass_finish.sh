#!/usr/bin/env bash
# 主恢复队列成功退出后，核验全部产物，再独占运行已冻结的首遍对照测速。
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/rulermvs_v6_20261101/full_training
exec 9>"$output/paper_firstpass/finish.lock"
flock -n 9
trap 'status=$?; printf "%s\n" "$status" > "$output/paper_firstpass/finish.exit_code.txt"' EXIT

while [ ! -f "$output/evaluation_recovery/queue.exit_code.txt" ]; do
    sleep 30
done
if [ "$(cat "$output/evaluation_recovery/queue.exit_code.txt")" != "0" ]; then
    echo "主恢复队列没有成功退出，停止，不启动测速。" >&2
    exit 1
fi

run_python() {
    tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
        official_methods/.venv/bin/python "$@"
}

run_python research/paper_evidence_20261101/completion_audit.py
run_python research/paper_evidence_20261101/firstpass.py
run_python research/paper_analysis_20261101/analyze.py
run_python research/paper_evidence_20261101/firstpass_report.py
