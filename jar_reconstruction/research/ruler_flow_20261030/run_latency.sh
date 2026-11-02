#!/usr/bin/env bash
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction
python=official_methods/.venv/bin/python
study=research/ruler_flow_20261030
output=runs/active/ruler_flow_20261030/logs
trap 'code=$?; echo "$code" > "$output/latency_queue.exit_code.txt"' EXIT
"$python" "$study/latency.py" --config "$study/configs/latency_final_public.json" > "$output/latency_public.log" 2>&1
"$python" "$study/latency.py" --config "$study/configs/latency_final_carla.json" > "$output/latency_carla.log" 2>&1
