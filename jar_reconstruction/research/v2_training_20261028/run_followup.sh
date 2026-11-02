#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output=runs/active/v2_training_20261028
for name in pilot_d_low_lr pilot_e_balanced smoke_carla; do
    echo "$(date -Is) START $name"
    bash research/v2_training_20261028/run_8gpu.sh "research/v2_training_20261028/configs/$name.json" \
        >> "$output/${name}_console.log" 2>&1
    echo "$(date -Is) COMPLETE $name"
done
