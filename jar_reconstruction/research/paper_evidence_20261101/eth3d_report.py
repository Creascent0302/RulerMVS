"""汇总已完成的ETH3D外部测试；保留全部方法与失败场景。"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.paper_analysis_20261101 import analyze as plots


def main():
    folder = ROOT / 'runs/active/rulermvs_v6_20261101/full_training/paper_eth3d'
    source = folder / 'evaluation/full_test'
    result = plots.read(source / 'summary.json')
    preparation = plots.read(folder / 'preparation.json')
    if not result['completed']:
        raise ValueError('不能汇总未完成测试')
    output = folder / 'analysis'
    output.mkdir(exist_ok=True)
    plots.LABELS['mvsanywhere_firstpass'] = 'MVSAnywhere (first pass)'
    plots.COLORS['mvsanywhere_firstpass'] = '#8286c9'
    depth, stages, surfaces = plots.flatten(result, 'external_ETH3D', 'test')
    plots.csv_write(output / 'depth_metrics.csv', depth)
    plots.csv_write(output / 'stage_metrics.csv', stages)
    plots.csv_write(output / 'surface_metrics.csv', surfaces)
    names = list(result['datasets'])
    parts = [plots.read(source / name / 'part_0000.json') for name in names]
    assert all(part['completed'] and len(part['frames']) == 27 for part in parts)
    hash_checks = []
    for a, b in zip(parts[0]['frames'], parts[1]['frames']):
        assert a['index'] == b['index']
        if 'input_sha256' in a:
            assert a['input_sha256'] == b['input_sha256']
            hash_checks.append(a['index'])
    assert hash_checks == [0, 13, 26]
    config = plots.read(folder / 'evaluation.json')
    manifests = [[json.loads(line) for line in Path(domain['manifests']['test']).read_text().splitlines()]
                 for domain in config['datasets']]
    for a, b in zip(*manifests):
        for record in (a, b):
            for camera in record['cameras']:
                camera.pop('targets', None)
        assert a == b
    methods = ('v5_best', 'v6_best', 'mvsanywhere_firstpass', 'mvsanywhere_refined')
    grouped = {}
    for frame in parts[0]['frames']:
        scene = preparation['pairs'][frame['index']]['physical_scene']
        group = grouped.setdefault(scene, {method: [0., 0] for method in methods})
        for method in methods:
            sums = frame['methods'][method]['depth']['full']
            group[method][0] += sums['abs_sum']
            group[method][1] += sums['predicted_pixels']
    rows = [{'scene': scene, 'method': method, 'MAE_m': sums[0] / sums[1],
             'predicted_pixels': sums[1]} for scene, values in grouped.items()
            for method, sums in values.items()]
    plots.csv_write(output / 'scene_metrics.csv', rows)

    fig, axes = plots.plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, metric in zip(axes, ('MAE', 'RMSE', 'AbsRel')):
        for index, method in enumerate(methods):
            x = np.arange(2) + (index - 1.5) * .18
            values = [result['datasets'][name]['methods'][method]['depth']['full'][metric] for name in names]
            ax.bar(x, values, width=.17, color=plots.COLORS[method], label=plots.LABELS[method])
        ax.set(xticks=[0, 1], xticklabels=['All valid GT', 'Non-occluded GT'],
               ylabel=metric + (' (m)' if metric != 'AbsRel' else ''), title=metric + ' (lower is better)')
        ax.grid(axis='y', alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False, fontsize=8)
    plots.save(fig, output, 'external_eth3d',
               'ETH3D external test: all 27 official GT stereo pairs, 7 physical scene names. Frozen public checkpoints; 2 real views; 256x448 shared RGB; range 0.1-100 m fixed before download. Not the official disparity benchmark.')

    fig, ax = plots.plt.subplots(figsize=(10, 4), layout='constrained')
    for method in methods:
        ax.plot(range(len(grouped)), [values[method][0] / values[method][1] for values in grouped.values()],
                'o-', ms=4, color=plots.COLORS[method], label=plots.LABELS[method])
    ax.set(xticks=range(len(grouped)), xticklabels=list(grouped), ylabel='Pixel-pooled MAE (m)',
           title='ETH3D external test by physical scene name')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, ncol=2)
    plots.save(fig, output, 'external_eth3d_scenes',
               'All valid GT. l/s and capture variants are grouped by physical scene name; no scene removal or target-domain training.')

    lines = ['# ETH3D真实外部测试', '',
             '全部27个官方有GT双目对、7个场景名称；两个真实灰度视图转三通道，不复制为四目。',
             '公共权重在下载前固定，未训练或调参。输入范围0.1–100m同样提前固定，GT范围外不删除。',
             '这里是共同深度协议，不是ETH3D官方disparity榜单，也不是MVSAnywhere论文RMVD设置。', '',
             '|掩码|方法|MAE(m)↓|RMSE(m)↓|AbsRel↓|预测覆盖率|', '|---|---|---:|---:|---:|---:|']
    for name, domain in result['datasets'].items():
        for method, values in domain['methods'].items():
            metric = values['depth']['full']
            lines.append(f"|{name.removeprefix('ETH3D_two_view_')}|{plots.LABELS[method]}|{metric['MAE']:.6f}|{metric['RMSE']:.6f}|{metric['AbsRel']:.6f}|{100 * metric['coverage']:.4f}%|")
    a = result['datasets'][names[0]]['methods']
    v5, v6, ref = (a[m]['depth']['full'] for m in ('v5_best', 'v6_best', 'mvsanywhere_refined'))
    lines += ['', f"V6相V5 MAE变化{100 * (v6['MAE'] / v5['MAE'] - 1):+.2f}%，相MVS细化为{v6['MAE'] / ref['MAE']:.2f}倍。",
              '本次外测不支持普遍的真实域优势；不能只报告合成域的胜出。灰度、约6cm基线、真实噪声与外观迁移均是可能因素，当前没有隔离它们的因果作用。',
              'all/nocc的27对输入清单去除targets后完全相同，前/中/末实际输入hash相同；未保存全部预测hash，不宣称逐位输出相同。',
              '官方深度由Z=f×baseline/(disparity+doffs)转换，毫米基线转米；原始K使用角点原点，主点减0.5适配像素中心。',
              '双目精确投影回算disparity最大误差2.483e−6像素，27/27对正确方向的光度误差均低于零位移和反向位移。',
              '本项目训练清单未用ETH3D；通用预训练原始曝光不能排除。Patch/Diff沿用小预算CARLA权重，不能将其失败当同预算结构结论。',
              '官方来源：https://www.eth3d.net/datasets ，格式说明：https://www.eth3d.net/documentation 。',
              '下载仅服务器直连，两个压缩包共29,146,428字节，route/curl元数据及SHA保存于数据目录。']
    (output / '结果说明.md').write_text('\n'.join(lines) + '\n')
    plots.write_json(output / 'audit.json', {'completed': True, 'pairs': 27, 'scene_count': len(grouped),
                     'all_nocc_input_records_equal': 27, 'input_hash_checked_indices': hash_checks,
                     'prediction_bitwise_check': 'not_run',
                     'summary_sha256': plots.sha(source / 'summary.json'), 'script_sha256': plots.sha(__file__)})
    print(json.dumps({'completed': True, 'output': str(output)}))


if __name__ == '__main__':
    main()
