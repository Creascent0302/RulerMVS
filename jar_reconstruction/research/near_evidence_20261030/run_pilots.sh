#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
mode=${1:-smoke}
experiment_set=${2:-round1}
if [[ "$mode" != smoke && "$mode" != formal ]]; then
    echo '参数必须为smoke或formal' >&2
    exit 2
fi
case "$experiment_set" in
    round1) arms=(control relative_evidence intervention_only) ;;
    coarse) arms=(coarse_attention) ;;
    visibility) arms=(visibility_veto) ;;
    *) echo '实验组必须为round1、coarse或visibility' >&2; exit 2 ;;
esac
output=runs/active/near_evidence_20261030
mkdir -p "$output/logs"
pids=()
for phase in public carla; do
    for arm in "${arms[@]}"; do
        name="${phase}_${arm}"
        [[ "$mode" == smoke ]] && name="smoke_${name}"
        (
            tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
                PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
                official_methods/.venv/bin/python research/near_evidence_20261030/pilot.py \
                --config "research/near_evidence_20261030/configs/${name}.json" \
                > "$output/logs/${name}.log" 2>&1
            status=$?
            printf '%s\n' "$status" > "$output/logs/${name}.exit_code.txt"
            exit "$status"
        ) &
        pids+=("$!")
    done
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
prefix="${experiment_set}_${mode}"
printf '%s\n' "$status" > "$output/logs/${prefix}.exit_code.txt"
exit "$status"
