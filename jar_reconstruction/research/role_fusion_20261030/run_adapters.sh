#!/usr/bin/env bash
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction
python=official_methods/.venv/bin/python
study=research/role_fusion_20261030
output=runs/active/role_fusion_20261030/logs
mkdir -p "$output"
trap 'code=$?; echo "$code" > "$output/adapter_verified_queue.exit_code.txt"' EXIT

"$python" "$study/run_pilots.py" --config "$study/configs/queue_adapter_smoke.json"
"$python" "$study/verify_adapter_smoke.py"
"$python" "$study/run_pilots.py" --config "$study/configs/queue_adapters.json"
"$python" "$study/summarize.py" --config "$study/configs/summary_adapters.json"
"$python" "$study/evaluate_adapters.py" --config "$study/configs/evaluate_adapters_public.json" > "$output/adapter_fresh_public.log" 2>&1 &
public_pid=$!
"$python" "$study/evaluate_adapters.py" --config "$study/configs/evaluate_adapters_carla.json" > "$output/adapter_fresh_carla.log" 2>&1 &
carla_pid=$!
failed=0
wait "$public_pid" || failed=1
wait "$carla_pid" || failed=1
exit "$failed"
