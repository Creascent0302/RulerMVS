#!/usr/bin/env bash
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction
python=official_methods/.venv/bin/python
study=research/ruler_flow_20261030
output=runs/active/ruler_flow_20261030/logs
trap 'code=$?; echo "$code" > "$output/followup_verified.exit_code.txt"' EXIT
OMP_NUM_THREADS=1 "$python" -m unittest \
  research.ruler_flow_20261030.test_direction \
  research.ruler_flow_20261030.test_flow \
  research.ruler_flow_20261030.test_followup \
  research.ruler_flow_20261030.test_variants \
  research.ruler_flow_20261030.test_runner -v > "$output/cpu_followup_tests.log" 2>&1
"$python" "$study/run_pilots.py" --config "$study/configs/queue_followup_smoke.json"
"$python" "$study/verify_smoke.py" --config "$study/configs/verify_followup.json"
"$python" "$study/run_pilots.py" --config "$study/configs/queue_followup_formal.json"
"$python" "$study/summarize.py" --config "$study/configs/summary_followup.json"
