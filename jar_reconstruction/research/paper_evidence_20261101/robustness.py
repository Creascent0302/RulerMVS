"""预登记VAL成对压力测试：同帧枚举相机子集及固定源退化，不调模型。"""

import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import add_statistics, digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from research.rulermvs_v6_20261101.evaluation import Models
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import fingerprint, offline_only

REGIONS = ('full', 'near', 'mid', 'far')


def make_conditions(reference, views, settings):
    sources = [i for i in range(views) if i != reference]
    conditions = []
    for count in (4, 3, 2):
        for subset in itertools.combinations(sources, count - 1):
            keep = sorted((reference,) + subset)
            conditions.append({'name': 'views_' + ''.join(map(str, keep)), 'kind': 'views', 'keep': keep})
    all_views = list(range(views))
    conditions += [{'name': f'noise_{std:g}', 'kind': 'noise', 'std': std, 'keep': all_views}
                   for std in settings['noise_std']]
    conditions += [{'name': f'blur_{size}', 'kind': 'blur', 'kernel': size, 'keep': all_views}
                   for size in settings['blur_kernels']]
    return conditions


def perturb(inputs, condition, source, seed):
    result = dict(inputs)
    keep = torch.zeros_like(inputs['view_mask'])
    keep[:, condition['keep']] = True
    result['view_mask'] = inputs['view_mask'] & keep
    for key in ('image_mask', 'matching_image_mask'):
        result[key] = inputs[key] & keep[..., None, None]
    if 'camera_valid' in inputs:
        result['camera_valid'] = inputs['camera_valid'] & keep
    if condition['kind'] != 'views':
        rgb = inputs['rgb'].clone()
        if condition['kind'] == 'noise':
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(rgb[:, source].shape, generator=generator, dtype=rgb.dtype)
            rgb[:, source] = (rgb[:, source] + condition['std'] * noise).clamp(0, 1)
        else:
            kernel = condition['kernel']
            pad = kernel // 2
            rgb[:, source] = F.avg_pool2d(F.pad(rgb[:, source], (pad,) * 4, mode='reflect'), kernel, stride=1)
        result['rgb'] = rgb
    else:
        assert torch.equal(result['rgb'], inputs['rgb'])
    return equal_pixel_inputs(result)


def plot_report(report, output):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    labels = {'v5_best': 'V5', 'v6_best': 'V6', 'mvsanywhere_firstpass': 'MVSAnywhere 1 pass',
              'mvsanywhere_refined': 'MVSAnywhere refined'}
    colors = ('#3574ae', '#d36b32', '#38926e', '#8a65b8')
    conditions = report['conditions']
    clean = next(c['name'] for c in conditions if c['kind'] == 'views' and len(c['keep']) == 4)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    for method, color in zip(report['methods'], colors):
        groups = report['view_count_equal_subset'][method]
        x = [2, 3, 4]
        means = [groups[str(n)]['full']['MAE']['mean'] for n in x]
        axes[0].plot(x, means, '-o', color=color, label=labels[method])
        axes[0].fill_between(x, [groups[str(n)]['full']['MAE']['min'] for n in x],
                            [groups[str(n)]['full']['MAE']['max'] for n in x], color=color, alpha=.12)
        for ax, kind, parameter, origin in ((axes[1], 'noise', 'std', 0.), (axes[2], 'blur', 'kernel', 1)):
            choices = [c for c in conditions if c['kind'] == kind]
            xs = [origin] + [c[parameter] for c in choices]
            ys = [report['totals'][clean][method]['full']['MAE']] + [
                report['totals'][c['name']][method]['full']['MAE'] for c in choices]
            ax.plot(xs, ys, '-o', color=color, label=labels[method])
            ax.set_xticks(xs)
    for ax, title, xlabel in zip(axes, ('Physical camera subsets', 'Noise on left source', 'Blur on left source'),
                                ('Available cameras', 'Gaussian noise std, RGB [0,1]', 'Mean-filter width (pixels)')):
        ax.set(title=title, xlabel=xlabel, ylabel='Depth MAE (m), lower is better')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_xticks([2, 3, 4])
    axes[0].legend(fontsize=8)
    fig.suptitle('Paired VAL stress test | 16 fixed frames | shaded range: all subsets, not a confidence interval')
    fig.savefig(output / 'paired_robustness.png', dpi=200)
    fig.savefig(output / 'paired_robustness.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    settings = json.loads(Path(parser.parse_args().config).read_text())
    output = Path(settings['output'])
    if not output.resolve().is_relative_to(Path('/data')):
        raise ValueError('结果必须写入数据盘')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    config = json.loads(Path(settings['evaluation_config']).read_text())
    domain = next(d for d in config['datasets'] if d['name'] == settings['dataset'])
    config['methods'] = {m: config['methods'][m] for m in ('v5_best', 'v6_best', 'mvsanywhere_refined')}
    firstpass = copy.deepcopy(config['methods']['mvsanywhere_refined'])
    firstpass['refinements'] = 0
    config['methods'] = dict(list(config['methods'].items())[:2], mvsanywhere_firstpass=firstpass,
                             mvsanywhere_refined=config['methods']['mvsanywhere_refined'])
    indices_plan = json.loads(Path(settings['diagnostics']).read_text())['plans'][domain['name']]
    assert digest(domain['manifests']['val']) == indices_plan['manifest_sha256']
    assert indices_plan['indices'] == settings['indices']
    reference = domain['reference_index']
    source = domain['camera']['wide_ids'].index(settings['corrupt_source'])
    assert reference != source
    conditions = make_conditions(reference, 4, settings)
    report = {'completed': False, 'split': 'VAL', 'test_accessed': False, 'settings': settings,
              'domain': domain, 'indices': settings['indices'], 'conditions': conditions,
              'methods': list(config['methods']), 'reference_index': reference, 'source_index': source,
              'source_sha256': digest(__file__), 'manifest_sha256': digest(domain['manifests']['val']),
              'weights_sha256': {m: digest(c['weights']) for m, c in config['methods'].items()},
              'notes': ['预先固定条件；不依据结果选择结构或参数。', 'GT只评分，参考RGB不变；扰动仅施加固定左源。',
                        '缺相机RGB不复制不篡改，只同步屏蔽view/image/matching/camera_valid；原生MVS压紧真实相机。',
                        '所有matching_rgb从共同RGB重建。相同帧同seed噪声方向在两个std间相同。',
                        '同输入同GT的成对压力测试；不是最终TEST鲁棒性或跨域泛化证明；不测速。'],
              'frames': [], 'totals': {}}
    write_json(output / 'plan.json', report)
    offline_only()
    device = configure(config, settings['gpu'])
    models = Models(config, device)
    dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=True)
    totals = {c['name']: {m: {r: {} for r in REGIONS} for m in config['methods']} for c in conditions}
    for index in settings['indices']:
        batch = collate_frames([dataset[index]])
        assert bool(batch['inputs']['view_mask'].all())
        frame = {'index': index, 'frame_id': batch['metadata'][0]['frame_id'],
                 'input_sha256': fingerprint(batch['inputs']), 'conditions': {}}
        truth, mask = ground_truth(batch, domain)
        masks = regions(truth, mask, domain)
        for condition in conditions:
            altered = perturb(batch['inputs'], condition, source, settings['seed'] + index)
            assert torch.equal(altered['rgb'][:, reference], batch['inputs']['rgb'][:, reference])
            assert torch.equal(altered['K'], batch['inputs']['K'])
            assert torch.equal(altered['T_cam_from_rig'], batch['inputs']['T_cam_from_rig'])
            inputs = move(altered, device)
            predictions = {}
            with torch.inference_mode():
                for method in config['methods']:
                    depth, valid = models.convert(models.forward(method, domain, inputs), inputs)
                    predictions[method] = (depth.cpu(), valid.cpu())
            item = {'valid_views': len(condition['keep']), 'input_sha256': fingerprint(altered), 'methods': {}}
            for method, (depth, valid) in predictions.items():
                sums = {r: depth_sums(depth, valid, truth, masks[r]) for r in REGIONS}
                for r in REGIONS:
                    assert sums[r]['gt_pixels'] == int(masks[r].sum())
                    add_statistics(totals[condition['name']][method][r], sums[r])
                item['methods'][method] = sums
            frame['conditions'][condition['name']] = item
        report['frames'].append(frame)
        print(json.dumps({'completed_frame': index, 'conditions': len(conditions), 'methods': len(config['methods'])}), flush=True)
        write_json(output / 'results.json', report)
    report['totals'] = {condition: {m: {r: depth_metrics(s) for r, s in regions_.items()}
                                   for m, regions_ in values.items()} for condition, values in totals.items()}
    report['view_count_equal_subset'] = {}
    for method in config['methods']:
        report['view_count_equal_subset'][method] = {}
        for count in (2, 3, 4):
            names = [c['name'] for c in conditions if c['kind'] == 'views' and len(c['keep']) == count]
            report['view_count_equal_subset'][method][str(count)] = {
                r: {metric: {'mean': float(np.mean(values)), 'min': float(min(values)), 'max': float(max(values))}
                    for metric in ('MAE', 'RMSE', 'AbsRel', 'coverage')
                    for values in [[report['totals'][name][method][r][metric] for name in names]]}
                for r in REGIONS}
    report['completed'] = True
    write_json(output / 'results.json', report)
    plot_report(report, output)
    lines = ['# 同帧VAL鲁棒性压力测试', '', '16个预定完整四目VAL，上参考固定；所有条件同帧、同GT、同参考图。',
             '7个物理视图子集覆盖4目1组、3目3组、2目3组；另外对左源施加两档高斯噪声和两档均值模糊。未按误差挑选子集。',
             '曲线阴影为所有子集的最小至最大MAE，不是统计置信区间。GT只用于评分，未用于决定扰动。没有计时、重训或模型调参。', '',
             '|条件|V5 MAE|V6 MAE|MVS首遍 MAE|MVS细化 MAE|', '|---|---:|---:|---:|---:|']
    for condition in conditions:
        lines.append('|' + condition['name'] + '|' + '|'.join(f"{report['totals'][condition['name']][m]['full']['MAE']:.4f}" for m in config['methods']) + '|')
    lines += ['', '完整JSON保留逐帧误差累加、全图/近中远指标、覆盖率和输入/权重hash。',
              '不同方法训练经历不同；只说明当前固定模型对这些已定义扰动的响应。退化不稳定或去掉相机反而改善也应如实保留，不能自动解释成学会了剔除坏视图。']
    (output / '结果与解释.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
