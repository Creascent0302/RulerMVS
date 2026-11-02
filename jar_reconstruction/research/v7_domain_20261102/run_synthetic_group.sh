#!/usr/bin/env bash
set -euo pipefail

cd /home/liuzonghao/3Dscene/jar_reconstruction
condition="${1:?usage: run_synthetic_group.sh new_only|joint_keep}"
case "$condition" in
    new_only|joint_keep) ;;
    *) exit 2 ;;
esac
base="runs/active/v7_domain_20261102/pilots_synthetic/coordinate_${condition}"
mkdir -p "$base"
trap 'code=$?; echo "$code" > "$base/queue.exit_code.txt"' EXIT

for phase in public carla; do
    official_methods/.venv/bin/python research/v7_domain_20261102/train_synthetic.py \
        --config "research/v7_domain_20261102/configs_synthetic/coordinate_${condition}_${phase}.json" \
        > "$base/${phase}.log" 2>&1
done

official_methods/.venv/bin/python - "$base" <<'PY'
import json
from pathlib import Path
import sys

from jar.pretraining.comparison import write_json

base = Path(sys.argv[1])
phases = {}
for phase in ("public", "carla"):
    training = json.loads((base / phase / "completed.json").read_text())
    audit = json.loads((base / phase / "synthetic_audit.json").read_text())
    if not training["completed"] or not audit["completed"]:
        raise RuntimeError("阶段没有通过完整验证")
    phases[phase] = {"steps": training["steps"], "exposures": audit["exposure"]}
write_json(base / "queue_completed.json", {"completed": True, "test_accessed": False,
                                           "phases": phases})
PY
