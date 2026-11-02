#!/usr/bin/env bash
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin=official_methods/.venv/bin/python
"$python_bin" research/early_geometry_20261030/verify_smoke.py
bash research/early_geometry_20261030/run_pilots.sh formal
"$python_bin" research/early_geometry_20261030/summarize.py
"$python_bin" research/early_geometry_20261030/latency.py --config research/early_geometry_20261030/configs/latency_public.json
"$python_bin" research/early_geometry_20261030/latency.py --config research/early_geometry_20261030/configs/latency_carla.json
echo 0 > runs/active/early_geometry_20261030/logs/verified_queue.exit_code.txt
