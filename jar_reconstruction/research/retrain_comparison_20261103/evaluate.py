"""仅计算主表所需深度统计；复用原GT掩码、指标与八卡分片调度。"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import add_statistics, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028 import evaluate as runner
from research.near_mid_evaluation_20261028.metrics import ground_truth, depth_sums, depth_metrics
from research.retrain_comparison_20261103.models import Models


def run_part(config, task, models, datasets, device, output, protocol):
    domain_id, part, indices = task
    domain = config['datasets'][domain_id]
    path = output / domain['name'] / f'part_{part:04d}.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('protocol') == protocol and old.get('completed'):
            return dict(event='cached', domain=domain['name'], part=part)
    if domain_id not in datasets:
        datasets[domain_id] = build_dataset(dataset_config(domain, config), config['split'], load_targets=True)
    loader = DataLoader(datasets[domain_id], batch_size=1, sampler=indices,
                        num_workers=config['loader_workers'], pin_memory=True, collate_fn=collate_frames)
    totals, records = {}, []
    started = time.monotonic()
    for index, batch in zip(indices, loader):
        inputs = move(batch['inputs'], device)
        # 监督只用于前向之后的统计，不传给任何模型。
        truth, mask = ground_truth(batch, domain)
        frame = {'index': index, 'methods': {}}
        for method in config['methods']:
            with torch.inference_mode():
                prediction = models.forward(method, domain, inputs)
                depth, valid = models.convert(prediction, inputs)
            sums = depth_sums(depth.cpu(), valid.cpu(), truth, mask)
            sums['gt_depth_sum'] = float(truth[mask].sum())
            add_statistics(totals.setdefault(method, {}), sums)
            frame['methods'][method] = sums
            del prediction, depth, valid
        records.append(frame)
    write_json(path, dict(completed=True, protocol=protocol, indices=indices, totals=totals, frames=records))
    return dict(event='complete', domain=domain['name'], part=part, frames=len(indices),
                seconds=round(time.monotonic() - started, 2))


def finish(sums):
    result = depth_metrics(sums)
    result['mean_gt_depth'] = sums['gt_depth_sum'] / sums['gt_pixels'] if sums['gt_pixels'] else None
    # 未预测像素不悄悄删去后给出貌似完备的横轴指标。
    result['NMAE'] = (sums['abs_sum'] / sums['gt_depth_sum']
                      if sums['gt_pixels'] == sums['predicted_pixels'] and sums['gt_depth_sum'] > 0 else None)
    return result


def merge(config, tasks, output, protocol):
    report = dict(completed=True, protocol=protocol, split=config['split'], datasets={})
    for domain_id, domain in enumerate(config['datasets']):
        selected = [t for t in tasks if t[0] == domain_id]
        expected = sorted(i for t in selected for i in t[2])
        totals, actual = {}, []
        for _, part, _ in selected:
            saved = json.loads((output / domain['name'] / f'part_{part:04d}.json').read_text())
            assert saved['completed'] and saved['protocol'] == protocol
            actual.extend(saved['indices'])
            for method, sums in saved['totals'].items():
                add_statistics(totals.setdefault(method, {}), sums)
        assert sorted(actual) == expected and len(set(actual)) == len(actual)
        assert len({s['gt_pixels'] for s in totals.values()}) == 1
        report['datasets'][domain['name']] = dict(frames=len(actual), units=domain['geometry_units'],
            num_views=domain['num_views'], sums=totals, methods={k: finish(v) for k, v in totals.items()})
    write_json(output / 'summary.json', report)


runner.Models = Models
runner.run_part = run_part
runner.merge = merge

if __name__ == '__main__':
    runner.main()
