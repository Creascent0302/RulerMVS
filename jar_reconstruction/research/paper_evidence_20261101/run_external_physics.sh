#!/usr/bin/env bash
# GPU7由主调度显式释放后启动；只跑准确率，不与独占测速重叠。
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction
run=runs/active/rulermvs_v6_20261101/full_training
exec 9>"$run/paper_external_physics.lock"
flock -n 9
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    official_methods/.venv/bin/python research/rulermvs_v6_20261101/evaluation.py \
    --config "$run/paper_eth3d/evaluation.json" > "$run/paper_eth3d/evaluation.log" 2>&1
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    official_methods/.venv/bin/python research/paper_evidence_20261101/physics.py \
    --config research/paper_evidence_20261101/physics.json > "$run/paper_physics_console.log" 2>&1
official_methods/.venv/bin/python research/paper_evidence_20261101/physics_report.py \
    --result "$run/paper_physics/results.json" > "$run/paper_physics_report.log" 2>&1
