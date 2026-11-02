"""固定两套旧权重测试新真实场景；原尺寸稀疏GT，不据结果调参。"""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import add_statistics, digest, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums
from research.rulermvs_v6_20261101.evaluation import Models
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import offline_only
from rmvd_subset import DATA


def main():
    offline_only()
    config = json.loads((PROJECT / 'runs/active/rulermvs_v6_20261101/full_training/'
                         'paper_eth3d/evaluation.json').read_text())
    config['methods'] = {name: config['methods'][name] for name in ('v6_best', 'mvsanywhere_refined')}
    # 延续此前真实ETH外测的事前固定公用范围，不读本场景GT确定输入范围。
    domain_template = dict(root=str(DATA), reference_index=0, geometry_units='meters',
                           near=.1, far=100., depth_scale=1.)
    device = configure(config, 7)
    models = Models(config, device)
    report = dict(completed=False, input_size=config['image_size'], datasets={},
                  weights_sha256={k: digest(v['weights']) for k, v in config['methods'].items()},
                  initial_range=[.1, 100.], test_range='所有有限正值GT，不截断范围外真值',
                  protocol='courtyard官方配组前4/8视图；预测双线性放回原尺寸；不是完整RMVD官方分数',
                  training=False, source_sha256=digest(__file__))
    write_json(DATA / 'evaluation_plan.json', report)
    for views in (4, 8):
        manifest = DATA / f'manifests/test_{views}views.jsonl'
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        domain = dict(domain_template, name=f'ETH3D_courtyard_{views}views', num_views=views,
                      manifests={'test': str(manifest)})
        dataset = build_dataset(dataset_config(domain, config), 'test', load_targets=False)
        sums, frames = {}, []
        for index, row in enumerate(rows):
            batch = collate_frames([dataset[index]])
            assert not batch['targets']
            inputs = move(batch['inputs'], device)
            predictions = {}
            with torch.inference_mode():
                for method in config['methods']:
                    prediction = models.forward(method, domain, inputs)
                    depth, valid = models.convert(prediction, inputs)
                    predictions[method] = depth.cpu(), valid.cpu()
            # 真值只在所有方法前向结束之后加载。
            evaluation = row['evaluation']
            shape = evaluation['native_shape']
            raw = np.fromfile(evaluation['native_depth'], np.float32).reshape(shape)
            truth = torch.from_numpy(raw).view(1, 1, *shape)
            mask = torch.isfinite(truth) & (truth > 0)
            frame = dict(index=index, source_sample=row['source_sample'], methods={})
            for method, (depth, valid) in predictions.items():
                depth = F.interpolate(depth, shape, mode='bilinear', align_corners=False)
                valid = F.interpolate(valid.float(), shape, mode='nearest-exact') > .5
                # 行分块累计，避免把6200x4100深度多次复制为大双精度数组。
                total = {}
                for y in range(0, shape[0], 128):
                    section = (..., slice(y, y + 128), slice(None))
                    add_statistics(total, depth_sums(depth[section], valid[section],
                                                    truth[section], mask[section]))
                add_statistics(sums.setdefault(method, {}), total)
                frame['methods'][method] = depth_metrics(total)
            frames.append(frame)
            print(views, index, {k: round(v['MAE'], 5) for k, v in frame['methods'].items()}, flush=True)
        report['datasets'][domain['name']] = dict(frames=len(frames),
            methods={k: depth_metrics(v) for k, v in sums.items()}, records=frames,
            manifest_sha256=digest(manifest))
        write_json(DATA / 'evaluation_progress.json', report)
    report['completed'] = True
    write_json(DATA / 'evaluation_results.json', report)


if __name__ == '__main__':
    main()
