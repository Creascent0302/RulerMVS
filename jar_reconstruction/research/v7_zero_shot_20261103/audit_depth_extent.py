"""推理全部结束后审计真值范围；结果仅作诊断，绝不回传模型或修改主指标。"""
import json
from pathlib import Path

import numpy as np

DATA = Path('/data/liuzonghao_data/dataset/V7_zero_shot_20261103')


def main():
    assert json.loads((DATA / 'results/delivery.json').read_text())['completed']
    result = {}
    for path in sorted((DATA / 'manifests').glob('*_4views.jsonl')):
        if path.name.startswith('timing_'):
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        record = dict(valid_pixels=0, below_input_near=0, above_input_far=0,
                      minimum=float('inf'), maximum=0.)
        for row in rows:
            info = row['evaluation']
            with np.load(info['native_depth']) as archive:
                depth = archive[info.get('array_key', 'depth')]
            valid = np.isfinite(depth) & (depth > 0)
            values = depth[valid]
            record['valid_pixels'] += int(valid.sum())
            record['below_input_near'] += int((values < .1).sum())
            record['above_input_far'] += int((values > 100).sum())
            record['minimum'] = min(record['minimum'], float(values.min()))
            record['maximum'] = max(record['maximum'], float(values.max()))
        result[path.stem] = record
        print(path.stem, record, flush=True)
    output = DATA / 'results/analysis/depth_extent_audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
