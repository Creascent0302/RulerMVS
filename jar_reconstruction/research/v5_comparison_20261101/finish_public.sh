#!/usr/bin/env bash
# 接管已有公共评测的收尾；若原终端中断，按原协议从完整分片继续。
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction
evaluation_pid=${1:?需要本轮原评测PID}
while test -r "/proc/${evaluation_pid}/cmdline" && rg -a -q 'patchmatchnet_public.json' "/proc/${evaluation_pid}/cmdline"; do
    sleep 10
done
if ! test -f runs/active/v5_comparison_20261101/patchmatchnet_public/full_test/summary.json; then
    OMP_NUM_THREADS=2 official_methods/.venv/bin/python research/v5_comparison_20261101/evaluate.py \
        --config research/v5_comparison_20261101/configs/patchmatchnet_public.json \
        >> runs/active/v5_comparison_20261101/patchmatchnet_public_resume.log 2>&1
fi
official_methods/.venv/bin/python research/v5_comparison_20261101/report.py
official_methods/.venv/bin/python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path('runs/active/v5_comparison_20261101')
summary = json.loads((root / 'patchmatchnet_public/full_test/summary.json').read_text())
marker = root / 'patchmatchnet_public/completion_recorded.json'
if not summary['completed'] or summary['split'] != 'test':
    raise RuntimeError('公共TEST未完成')
if not marker.exists():
    items = []
    for name, domain in summary['datasets'].items():
        record = domain['methods']['patchmatchnet_dtu']
        full = record['depth']['full']
        items.append(f"{name}({domain['units']}, n={record['frames']}): MAE={full['MAE']:.5f}, RMSE={full['RMSE']:.5f}, coverage={full['coverage']:.7f}")
    note = '\n\n### 实验 193 完成补充：PatchmatchNet七域公共TEST\n\n'
    note += '- 任务结束时间UTC：' + datetime.now(timezone.utc).isoformat() + '。原协议、官方DTU权重固定，完整11970组完成；中断只按协议哈希复用完整分片。\n'
    note += '- 实测：' + '；'.join(items) + '。\n'
    note += '- 保留该发布权重迁移对照，不将其与充分训练V5的差距解释为架构上限；条件误差与覆盖率同时保留，scene_unit不是米。没有根据TEST修改模型或当前训练。后续需共同公共预训练或官方域内协议后才讨论方法上限。\n'
    note += '- 证据：runs/active/v5_comparison_20261101/patchmatchnet_public/full_test/summary.json，完整表同步到docs/v5_evaluation_20261101.md。\n'
    with Path('docs/research_memory.md').open('a') as stream:
        stream.write(note)
    marker.write_text(json.dumps({'completed': True, 'experiment': 193}, indent=2))
PY
