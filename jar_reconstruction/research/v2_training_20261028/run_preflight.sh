#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/v2_training_20261028
for name in resume_check resume_check resume_control pilot_a_flow pilot_b_risk pilot_c_range; do
    echo "$(date -Is) START $name"
    bash research/v2_training_20261028/run_8gpu.sh "research/v2_training_20261028/configs/$name.json" \
        >> "$output/${name}_console.log" 2>&1
    echo "$(date -Is) COMPLETE $name"
done
