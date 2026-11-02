"""从已完成物理/置信度探针生成论文图表；不修改原始统计。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {'v5_best': 'V5', 'v6_best': 'V6', 'mvsanywhere_firstpass': 'MVSAnywhere 1 pass',
          'mvsanywhere_refined': 'MVSAnywhere refined'}
COLORS = {'v5_best': '#d28b32', 'v6_best': '#007f86', 'mvsanywhere_firstpass': '#8286c9',
          'mvsanywhere_refined': '#4452ae'}


def save(fig, output, name):
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(output / f'{name}.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main(path):
    result = json.loads(path.read_text())
    if not result['completed']:
        raise ValueError('不绘制未完成实验')
    output = path.parent / 'analysis'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    domains = result['datasets']
    rows = []
    checks = {}
    for name, domain in domains.items():
        checks[name] = {}
        for condition, methods in domain['methods'].items():
            for method, metrics in methods.items():
                if not metrics:
                    continue
                full = metrics['full']
                clean = domain['methods']['clean'][method]['full']
                changes = [frame['conditions'][condition][method]['relative_to_clean_prediction']
                           for frame in domain['frames']]
                delta = sum(v['sum_abs'] for v in changes) / max(1, sum(v['pixels'] for v in changes))
                peak = max(v['max_abs'] or 0 for v in changes)
                rows.append({'domain': name, 'units': domain['units'], 'condition': condition, 'method': method,
                             'MAE': full['MAE'], 'RMSE': full['RMSE'], 'AbsRel': full['AbsRel'],
                             'coverage': full['coverage'], 'MAE_ratio_to_clean': full['MAE'] / clean['MAE'],
                             'mean_prediction_difference': delta, 'max_prediction_difference': peak})
                if condition in ('unit_half', 'unit_double', 'rig_change'):
                    checks[name].setdefault(condition, {})[method] = {
                        'mean_difference': delta, 'max_difference': peak,
                        'all_valid_masks_equal': all(v['valid_mask_equal'] for v in changes)}
    with (output / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'equivariance_checks.json').write_text(json.dumps(checks, indent=2))

    for mode in ('scale', 'risk'):
        fig, axes = plt.subplots(3, 4, figsize=(14, 9), layout='constrained')
        for ax, (name, domain) in zip(axes.flat, domains.items()):
            if mode == 'scale':
                for method in LABELS:
                    baseline = domain['methods']['clean'][method]['full']['MAE']
                    y = [domain['methods'][c][method]['full']['MAE'] / baseline
                         for c in ('physical_half', 'clean', 'physical_double')]
                    ax.plot([.5, 1, 2], y, '-o', ms=4, color=COLORS[method], label=LABELS[method])
                ax.axhline(1, color='grey', lw=.7, ls=':')
                ax.set(xlabel='Physical scale (shared range scales too)', ylabel='MAE / own clean MAE', xticks=[.5, 1, 2])
            else:
                for method in ('v5_best', 'v6_best'):
                    record = domain['confidence'][method]
                    ax.plot(record['coverage'], record['risk_AbsRel'], color=COLORS[method], label=LABELS[method])
                    ax.plot(record['coverage'], record['oracle_risk_AbsRel'], color=COLORS[method], ls=':', alpha=.6)
                ax.set(xlabel='Prediction coverage', ylabel='Retained-pixel AbsRel')
            ax.set_title(name.replace('CARLA_new_stereo_', 'CARLA ').replace('_', ' '), fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=.15)
        axes.flat[-1].axis('off')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, labels, loc='center', frameon=False)
        title = ('Physical scene scaling | fixed VAL | predictions divided by scale before scoring'
                 if mode == 'scale' else 'Native confidence ranking | fixed VAL | dotted: offline oracle, not model output')
        fig.suptitle(title, fontsize=13)
        save(fig, output, 'physical_scale' if mode == 'scale' else 'risk_coverage')

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout='constrained')
    conditions = {'focal_1pct': 'Focal length error: +1%', 'focal_5pct': 'Focal length error: +5%',
                  'source_yaw_025': 'Source orientation error: 0.25 degrees',
                  'source_yaw_1': 'Source orientation error: 1 degree'}
    for ax, condition in zip(axes.flat, conditions):
        x = np.arange(len(domains))
        for method in LABELS:
            values = [domain['methods'][condition][method]['full']['MAE'] /
                      domain['methods']['clean'][method]['full']['MAE'] for domain in domains.values()]
            ax.plot(x, values, '-o', ms=3, label=LABELS[method], color=COLORS[method])
        ax.axhline(1, color='gray', lw=.7, ls=':')
        ax.set(xticks=x, xticklabels=['TA', 'MC', 'Hyp', 'Synth', 'VK', 'Blend', 'Dyn', 'Up', 'Left', 'Right', 'Down'],
               ylabel='MAE / own clean MAE', title=conditions[condition])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Input calibration stress | identical RGB and evaluation GT | fixed VAL')
    save(fig, output, 'calibration_stress')

    lines = ['# 尺度、标定与原生置信度诊断', '',
             '11分区各16个预登记VAL；无新训练、无TEST、无基于结果的阈值适配。', '',
             '单位换算仅检验自研depth_scale契约；物理尺度变化则保持depth_scale不变，',
             '给所有方法同步缩放的固定TRAIN范围，不能称不带范围条件的尺度泛化。',
             '标定扰动保持RGB/GT不变；结果改善也可能是系统偏差偶然抵消，不是更准确标定。', '',
             '|分区|V5 AURC↓|V6 AURC↓|V5 excess AURC↓|V6 excess AURC↓|',
             '|---|---:|---:|---:|---:|']
    for name, domain in domains.items():
        a, b = domain['confidence']['v5_best'], domain['confidence']['v6_best']
        lines.append(f"|{name}|{a['AURC']:.6f}|{b['AURC']:.6f}|{a['excess_AURC']:.6f}|{b['excess_AURC']:.6f}|")
    lines += ['', '风险为相对深度误差，不混合不同单位；原生置信度不是校准后的正确概率。',
              'excess AURC减去同一模型误差集合的离线最优排序，不能替代最终精度。',
              '两种MVS输出没有在当前公共接口暴露同定义原生confidence，因此不伪造baseline置信度。',
              '所有原始误差/覆盖、每帧偏移以及置信度分箱计数保留于results.json。']
    (output / '结果说明.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path)
    main(parser.parse_args().result)
