#!/usr/bin/env bash
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction

python=official_methods/.venv/bin/python
study=research/role_fusion_20261030
output=runs/active/role_fusion_20261030/logs
mkdir -p "$output"
trap 'code=$?; echo "$code" > "$output/verified_queue.exit_code.txt"' EXIT

"$python" "$study/run_pilots.py" --config "$study/configs/queue_smoke.json"
"$python" "$study/verify_smoke.py"
"$python" "$study/run_pilots.py" --config "$study/configs/queue_formal.json"
"$python" "$study/summarize.py" --config "$study/configs/summary.json"
# 队列全部退出后再交替测速，不把训练并发导致的GPU负载混入推理时延。
"$python" "$study/latency.py" --config "$study/configs/latency_public.json"
"$python" "$study/latency.py" --config "$study/configs/latency_carla.json"
