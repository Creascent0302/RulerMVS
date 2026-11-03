"""固定权重零样本外测：无标签前向，原分辨率GT分块计分。"""

import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from torch.nn import functional as F

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import add_statistics, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028 import evaluate as runner
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums
from research.retrain_comparison_20261103.models import Models


def score_native(depth, valid, truth):
    """保持稀疏GT格点，预测上采样；不用GT做中值/尺度对齐。"""
    shape = truth.shape[-2:]
    depth = F.interpolate(depth, shape, mode='bilinear', align_corners=False)
    valid = F.interpolate(valid.float(), shape, mode='nearest-exact') > .5
    result = {name: {} for name in ('full', 'near', 'mid', 'far')}
    for y in range(0, shape[0], 128):
        section = (..., slice(y, y + 128), slice(None))
        gt = truth[section]
        mask = torch.isfinite(gt) & (gt > 0)
        regions = dict(full=mask, near=mask & (gt < 5), mid=mask & (gt >= 5) & (gt < 20),
                       far=mask & (gt >= 20))
        for name, selected in regions.items():
            sums = depth_sums(depth[section], valid[section], gt, selected)
            sums['gt_depth_sum'] = float(gt[selected].double().sum())
            add_statistics(result[name], sums)
    return result


def run_part(config, task, models, datasets, device, output, protocol):
    domain_id, part, indices = task
    domain = config['datasets'][domain_id]
    path = output / domain['name'] / f'part_{part:04d}.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol'] == protocol and previous['completed']:
            return dict(event='cached', domain=domain['name'], part=part)
    if domain_id not in datasets:
        datasets[domain_id] = build_dataset(dataset_config(domain, config), config['split'], load_targets=False)
    rows = [json.loads(line) for line in Path(domain['manifests']['test']).read_text().splitlines()]
    totals, frames = {}, []
    start = time.monotonic()
    for index in indices:
        row = rows[index]
        batch = collate_frames([datasets[domain_id][index]])
        assert not batch['targets']
        inputs = move(batch['inputs'], device)
        predictions = {}
        with torch.inference_mode():
            for method in config['methods']:
                predicted = models.forward(method, domain, inputs)
                depth, valid = models.convert(predicted, inputs)
                predictions[method] = depth.cpu(), valid.cpu()
                del predicted, depth, valid
        # 到此所有模型前向均已结束，才读取独立的评价标签。
        with np.load(row['evaluation']['native_depth']) as stored:
            truth = torch.from_numpy(stored[row['evaluation'].get('array_key', 'depth')].copy()).view(
                1, 1, *row['evaluation']['native_shape'])
        record = dict(index=index, source_sample=row['source_sample'], methods={})
        small = {}
        for method, (depth, valid) in predictions.items():
            sums = score_native(depth, valid, truth)
            record['methods'][method] = sums
            for region, value in sums.items():
                add_statistics(totals.setdefault(method, {}).setdefault(region, {}), value)
            small[method] = depth.numpy()
            small[method + '_valid'] = valid.numpy()
        output_dir = output / 'predictions' / domain['name']
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / f'{index:03d}.npz', **small)
        frames.append(record)
        print(json.dumps(dict(domain=domain['name'], frame=index, processed=True)), flush=True)
    write_json(path, dict(completed=True, protocol=protocol, indices=indices, frames=frames, totals=totals))
    return dict(event='complete', domain=domain['name'], part=part, frames=len(indices),
                seconds=round(time.monotonic() - start, 2))


def merge(config, tasks, output, protocol):
    report = dict(completed=True, protocol=protocol, datasets={})
    for domain_id, domain in enumerate(config['datasets']):
        totals, actual = {}, []
        for _, part, indices in [t for t in tasks if t[0] == domain_id]:
            item = json.loads((output / domain['name'] / f'part_{part:04d}.json').read_text())
            assert item['protocol'] == protocol and item['completed'] and item['indices'] == indices
            actual += indices
            for method, regions in item['totals'].items():
                for region, value in regions.items():
                    add_statistics(totals.setdefault(method, {}).setdefault(region, {}), value)
        assert len(actual) == len(set(actual))
        for region in ('full', 'near', 'mid', 'far'):
            assert len({v[region]['gt_pixels'] for v in totals.values()}) == 1
        report['datasets'][domain['name']] = dict(domain=domain, frames=len(actual), sums=totals,
            methods={m: {r: depth_metrics(s) for r, s in values.items()} for m, values in totals.items()})
    write_json(output / 'summary.json', report)


runner.Models = Models
runner.run_part = run_part
runner.merge = merge

if __name__ == '__main__':
    runner.main()
