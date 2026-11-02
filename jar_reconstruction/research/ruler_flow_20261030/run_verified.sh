#!/usr/bin/env bash
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction
python=official_methods/.venv/bin/python
study=research/ruler_flow_20261030
output=runs/active/ruler_flow_20261030/logs
mkdir -p "$output"
trap 'code=$?; echo "$code" > "$output/verified_queue.exit_code.txt"' EXIT

OMP_NUM_THREADS=1 "$python" -m unittest \
  research.ruler_flow_20261030.test_direction \
  research.ruler_flow_20261030.test_flow \
  research.ruler_flow_20261030.test_variants \
  research.ruler_flow_20261030.test_runner -v > "$output/cpu_all_tests.log" 2>&1
"$python" "$study/run_pilots.py" --config "$study/configs/queue_smoke.json"
"$python" "$study/verify_smoke.py"
"$python" "$study/run_pilots.py" --config "$study/configs/queue_formal.json"
"$python" "$study/summarize.py"
# 计时另行执行：确认其他GPU任务全部退出后，运行configs/latency_*.json。
