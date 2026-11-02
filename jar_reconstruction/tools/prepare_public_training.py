"""仅从TRAIN估计公共预训练范围，生成可执行配置；不读取VAL/TEST标签。"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    registry = json.loads((ROOT / 'configs/pretraining/public_v1.json').read_text())
    output = ROOT / 'runs/active/public_pretrain_no_hero_v2'
    datasets, statistics = [], []
    for source in registry['datasets']:
        manifest = Path(source['manifests']['train'])
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        # 覆盖清单全段；不挑深度合适的样本，不遍历TEST。
        selected = np.linspace(0, len(rows) - 1, min(512, len(rows)), dtype=int)
        values = []
        for index in selected:
            row = rows[index]
            for camera in row['cameras']:
                label = camera.get('targets', {})
                if not label.get('depth'):
                    continue
                if label['depth_type'] != 'optical_z':
                    raise ValueError('范围统计要求已转换光轴深度，不静默使用射线距离')
                depth = np.load(Path(source['root']) / label['depth'], mmap_mode='r')[::16, ::16].astype('float64')
                depth *= label.get('depth_scale', 1.)
                valid = np.isfinite(depth) & (depth > 0) & (depth < label.get('max_valid_depth', 1000.))
                if label.get('depth_valid'):
                    valid &= np.load(Path(source['root']) / label['depth_valid'], mmap_mode='r')[::16, ::16].astype(bool)
                values.append(depth[valid])
        values = np.concatenate(values)
        low, median, high = np.quantile(values, [.0001, .5, .9995])
        near, far = float(low * .5), float(high * 1.25)
        scale = float(median / 10.)
        domain = {key: source[key] for key in ('name', 'root', 'num_views', 'reference_index', 'geometry_units', 'manifests')}
        domain.update(near=near, far=far, depth_scale=scale,
                      correspondence=source['name'] != 'MVS_Synth',
                      train_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        datasets.append(domain)
        statistics.append({'name': source['name'], 'sampled_train_groups': len(selected),
                           'sampled_depth_values': len(values), 'quantiles': [float(low), float(median), float(high)],
                           'fraction_inside': float(((values >= near) & (values <= far)).mean()),
                           'train_manifest_sha256': domain['train_manifest_sha256']})
        print(source['name'], 'near/far/scale', near, far, scale, flush=True)
    config_path = ROOT / 'configs/pretraining/train_no_hero_v2.json'
    config = json.loads(config_path.read_text())
    config['datasets'] = datasets
    (output / 'train_range_statistics.json').write_text(json.dumps(statistics, indent=2) + '\n')
    (ROOT / 'configs/pretraining/train_no_hero_v2.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
