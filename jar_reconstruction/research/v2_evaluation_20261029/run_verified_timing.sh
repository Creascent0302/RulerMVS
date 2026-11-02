#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/v2_evaluation_20261029
python=official_methods/.venv/bin/python
trap 'status=$?; printf "%s\n" "$status" > "$output/verified_timing.exit_code.txt"' EXIT
for group in public carla; do
    echo "$(date -Is) SPEED $group"
    "$python" tools/benchmark_public_speed.py \
        --config "research/v2_evaluation_20261029/${group}_speed.json" \
        > "$output/${group}_speed_verified.log" 2>&1
done
echo "$(date -Is) EXTRA_CHECKS"
"$python" research/v2_evaluation_20261029/extra_checks.py > "$output/extra_checks.log" 2>&1
"$python" research/v2_evaluation_20261029/report.py
echo "$(date -Is) COMPLETE"
