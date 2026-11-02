"""固定TEST索引的稠密深度定性图；复用正式预测和指标，不按结果挑帧。"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.evaluate import predict_with_stages
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth
from research.rulermvs_v6_20261101.evaluation import Models
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import fingerprint, offline_only

METHODS = ('v5_best', 'v6_best', 'mvsanywhere_refined')
TITLES = ('RGB', 'GT depth', 'V5', 'V6', 'MVSAnywhere', 'V6 absolute error', 'MVS absolute error')


def plot_cases(cases, destination, depth_range, error_max):
    """旧visualize_results.py面向稀疏点渲染；这里直接画原始稠密预测，不补洞。"""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(21, 2.7 * len(cases) + .7), layout='constrained')
    grid = fig.add_gridspec(len(cases) + 1, 7, height_ratios=[1] * len(cases) + [.08])
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(7)]
                     for row in range(len(cases))])
    depth_palette = plt.get_cmap('viridis').copy()
    error_palette = plt.get_cmap('magma').copy()
    depth_palette.set_bad('#d4d4d4')
    error_palette.set_bad('#d4d4d4')
    for row, case in enumerate(cases):
        arrays = case['arrays']
        mask = arrays['gt_valid']
        fields = [arrays['rgb'], arrays['gt']] + [arrays[m] for m in METHODS]
        fields += [np.abs(arrays[m] - arrays['gt']) for m in ('v6_best', 'mvsanywhere_refined')]
        for col, values in enumerate(fields):
            ax = axes[row, col]
            ax.set_axis_off()
            ax.set_title(TITLES[col], fontsize=11)
            if col == 0:
                ax.imshow(values, interpolation='nearest')
                ax.text(0., -.10, case['reference_label'] + ' | frame ' + str(case['frame_id']),
                        transform=ax.transAxes, fontsize=9)
            else:
                valid = mask.copy()
                method = METHODS[col - 2] if 2 <= col <= 4 else None
                if col >= 5:
                    method = 'v6_best' if col == 5 else 'mvsanywhere_refined'
                if method:
                    valid &= arrays[method + '_valid']
                image = ax.imshow(np.ma.masked_where(~valid, values), interpolation='nearest',
                                  cmap=depth_palette if col < 5 else error_palette,
                                  norm=Normalize(*depth_range) if col < 5 else Normalize(0, error_max))
                if 2 <= col <= 4:
                    metric = case['metrics'][method]
                    ax.text(.5, -.10, f"MAE {metric['MAE']:.3f} m | RMSE {metric['RMSE']:.3f} m",
                            transform=ax.transAxes, fontsize=8, ha='center')
                if col == 1:
                    depth_image = image
                elif col == 5:
                    error_image = image
    fig.colorbar(depth_image, cax=fig.add_subplot(grid[-1, 1:5]), orientation='horizontal',
                 extend='both', label='Depth (m), shared scale')
    fig.colorbar(error_image, cax=fig.add_subplot(grid[-1, 5:]), orientation='horizontal',
                 extend='max', label='Absolute error (m), shared scale')
    fig.suptitle('CARLA TEST | preselected index 87 | gray: no valid evaluation target | no cropping', fontsize=12)
    fig.savefig(destination.with_suffix('.png'), dpi=180, facecolor='white')
    fig.savefig(destination.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    config = json.loads(Path(settings['evaluation_config']).read_text())
    output = Path(settings['output'])
    if not output.resolve().is_relative_to(Path('/data')):
        raise ValueError('定性产物必须保存在数据盘')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    offline_only()
    device = configure(config, settings['gpu'])
    models = Models(config, device)
    source = Path(config['output']) / 'full_test'
    report = {'settings': settings, 'split': 'TEST', 'selection': '每参考预定索引87，不按误差选帧',
              'weights_sha256': {m: digest(config['methods'][m]['weights']) for m in METHODS},
              'source_sha256': digest(__file__), 'cases': []}
    cases = []
    for domain in config['datasets']:
        dataset = build_dataset(dataset_config(domain, config), 'test', load_targets=True)
        index = settings['index']
        batch = collate_frames([dataset[index]])
        inputs = move(batch['inputs'], device)
        predictions = {}
        # 只有inputs进入模型；targets仅在全部预测完成后进入指标和图像渲染。
        with torch.inference_mode():
            for method in METHODS:
                depth, valid, _ = predict_with_stages(models, method, config['methods'][method], domain, inputs)
                predictions[method] = (depth, valid)
        truth, mask = ground_truth(batch, domain)
        manifest = json.loads(Path(domain['manifests']['test']).read_text().splitlines()[index])
        reference = domain['reference_index']
        rgb = batch['inputs']['rgb'][0, reference].permute(1, 2, 0).numpy()
        assert rgb.min() >= 0 and rgb.max() <= 1
        arrays = {'rgb': rgb, 'gt': truth[0, 0].float().numpy(), 'gt_valid': mask[0, 0].numpy()}
        metrics = {}
        expected = None
        for part in (source / domain['name']).glob('part_*.json'):
            found = [f for f in json.loads(part.read_text())['frames'] if f['index'] == index]
            if found:
                expected = found[0]
                break
        assert expected is not None
        for method, (depth, valid) in predictions.items():
            metrics[method] = depth_metrics(depth_sums(depth, valid, truth, mask))
            old = depth_metrics(expected['methods'][method]['depth']['full'])
            for key in ('MAE', 'RMSE', 'AbsRel', 'coverage'):
                assert abs(metrics[method][key] - old[key]) < 1e-5, (domain['name'], method, key)
            arrays[method] = depth[0, 0].numpy()
            arrays[method + '_valid'] = valid[0, 0].numpy()
        case = {'dataset': domain['name'], 'index': index, 'frame_id': manifest['frame_id'],
                'sequence': manifest['sequence'], 'reference_index': reference,
                'reference_label': domain['name'].replace('CARLA_new_', '').replace('stereo_', ''),
                'input_sha256': fingerprint(batch['inputs']), 'manifest_sha256': digest(domain['manifests']['test']),
                'metrics': metrics, 'arrays': arrays, 'metric_consistency_with_completed_test': True}
        cases.append(case)
        report['cases'].append({k: v for k, v in case.items() if k != 'arrays'})
        np.savez_compressed(output / (domain['name'] + '.npz'), **arrays)
        plot_cases([case], output / domain['name'], settings['depth_range'], settings['error_max'])
    plot_cases(cases, output / 'CARLA_fixed_test_four_references', settings['depth_range'], settings['error_max'])
    report.update(completed=True, prediction_uses_targets=False,
                  display={'depth_range': settings['depth_range'], 'error_range': [0, settings['error_max']],
                           'clipping_is_display_only': True, 'invalid_target_color': '#d4d4d4'})
    write_json(output / 'audit.json', report)
    lines = ['# 固定TEST样例定性图', '',
             '按预先指定的每参考索引87选择完整图像，不依据获胜情况选帧，不裁剪局部区域。四参考来自同一采集帧，不能视作四个独立场景。',
             '各行依次为RGB、GT、V5、V6、MVSAnywhere细化深度、V6绝对误差、MVS绝对误差。深度统一2–80米线性色标；误差统一0–10米，超界只在显示中饱和，原始数值和指标不截断。',
             '灰色为没有有效评测GT的区域，也可表示无效预测；不是零误差或空闲空间。GT只用于离线指标与显示掩码，模型仅接收RGB和标定inputs。',
             '本图用于展示可见错误类型，不代替完整测试。audit.json保留输入/权重/清单hash，逐图重新推理指标与已完成TEST一致（容差1e−5）。', '',
             '|参考|索引|原始frame_id|V5 MAE|V6 MAE|MVS细化 MAE|', '|---|---:|---|---:|---:|---:|']
    for case in cases:
        lines.append(f"|{case['reference_label']}|{case['index']}|{case['frame_id']}|" +
                     '|'.join(f"{case['metrics'][m]['MAE']:.4f}" for m in METHODS) + '|')
    (output / '图注与选择说明.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'completed': True, 'cases': len(cases), 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
