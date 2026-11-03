"""从真实完整结果生成唯一主表及精度—耗时图，不按结果筛选数据集。"""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np

from jar.pretraining.comparison import add_statistics, digest, write_json
from research.retrain_comparison_20261103.evaluate import finish

METHODS = {'v6': 'V6 重训·快速', 'v7': 'V7 重训·快速',
           'mvs_retrained': 'MVSAnywhere 重训·细化', 'mvs_hero': 'MVSAnywhere 官方 Hero·细化'}
ENGLISH = {'v6': 'RulerMVS-V6 (retrained)', 'v7': 'RulerMVS-V7 (retrained)',
           'mvs_retrained': 'MVSAnywhere (retrained, refined)',
           'mvs_hero': 'MVSAnywhere (Hero, refined)'}
LABELS = {'TartanAir_forest': 'TartanAir', 'MatrixCity_street': 'MatrixCity',
          'Hypersim_subset': 'Hypersim', 'MVS_Synth': 'MVS-Synth', 'VKITTI2': 'VKITTI2',
          'BlendedMVS': 'BlendedMVS', 'DynamicReplica_valid': 'DynamicReplica',
          'CARLA_new_upper': 'CARLA 上', 'CARLA_new_stereo_left': 'CARLA 左',
          'CARLA_new_stereo_right': 'CARLA 右', 'CARLA_new_stereo_down': 'CARLA 下',
          'ETH3D_two_view_all': 'ETH3D ALL', 'ETH3D_two_view_nocc': 'ETH3D 非遮挡',
          'CARLA_four': 'CARLA 四参考联合'}
PLOT_DATASETS = ['TartanAir_forest', 'MatrixCity_street', 'Hypersim_subset', 'MVS_Synth',
                 'VKITTI2', 'BlendedMVS', 'DynamicReplica_valid', 'CARLA_new_upper', 'ETH3D_two_view_all']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h']
COLORS = {'v6': '#009E73', 'v7': '#0072B2', 'mvs_retrained': '#D55E00', 'mvs_hero': '#777777'}


def nondominated(points):
    """仅在同一任务内求非支配点；两个目标均最小化。"""
    return [not any((x2 <= x and y2 <= y and (x2 < x or y2 < y))
                    for x2, y2 in points) for x, y in points]


def plot(rows, output):
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    frontier = {}
    for domain, marker in zip(PLOT_DATASETS, MARKERS):
        selected = [next(r for r in rows if r['dataset'] == domain and r['method'] == m) for m in METHODS]
        if any(r['NMAE'] is None for r in selected):
            raise ValueError('不完整预测覆盖不能用于无缺失惩罚的Pareto图')
        points = [(100 * r['NMAE'], r['ms']) for r in selected]
        front = nondominated(points)
        frontier[domain] = [r['method'] for r, chosen in zip(selected, front) if chosen]
        for row, (x, y), chosen in zip(selected, points, front):
            # 空心外圈表示该数据集内非支配，不代表所有数据集的全局前沿。
            if chosen:
                ax.scatter(x, y, marker=marker, s=205, facecolors='none', edgecolors=COLORS[row['method']],
                           linewidths=1.0, alpha=.55, zorder=2)
            ax.scatter(x, y, marker=marker, s=90, color=COLORS[row['method']],
                       edgecolors='white', linewidths=.55, alpha=.9, zorder=3)
        efficient = sorted([p for p, chosen in zip(points, front) if chosen])
        if len(efficient) > 1:
            ax.plot(*zip(*efficient), color='#777777', alpha=.22, linewidth=.8, zorder=1)
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 20, 50], labels=['2', '5', '10', '20', '50'])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Normalized MAE (%) = 100 × Σ|prediction − GT| / ΣGT', fontsize=10)
    ax.set_ylabel('Depth-output latency (ms), lower is better', fontsize=10)
    ax.set_title('Accuracy–latency trade-off across fixed test datasets', loc='left', fontsize=14)
    ax.text(.02, .02, 'Lower-left is better · outer ring: non-dominated within the same dataset',
            transform=ax.transAxes, fontsize=8, color='#555555')
    ax.grid(True, which='major', alpha=.16)
    ax.spines[['top', 'right']].set_visible(False)
    lo, hi = ax.get_ylim()
    ax.set_ylim(max(0, lo - 1), hi + 2)
    methods = [Line2D([0], [0], marker='o', linestyle='', color=COLORS[m], label=ENGLISH[m]) for m in METHODS]
    first = ax.legend(handles=methods, loc='upper left', bbox_to_anchor=(1.015, 1.02),
                      frameon=False, title='Method / color', fontsize=9)
    ax.add_artist(first)
    labels = ['TartanAir (4 views)', 'MatrixCity (4)', 'Hypersim (4)', 'MVS-Synth (4)', 'VKITTI2 (2)',
              'BlendedMVS (4)', 'DynamicReplica (2)', 'CARLA upper (4)', 'ETH3D ALL (2)']
    domains = [Line2D([0], [0], marker=m, linestyle='', color='#444444', label=l) for m, l in zip(MARKERS, labels)]
    second = ax.legend(handles=domains, loc='upper left', bbox_to_anchor=(1.015, .69),
                       frameon=False, title='Dataset / marker', fontsize=9)
    fig.subplots_adjust(left=.08, right=.69, top=.90, bottom=.12)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(output / ('accuracy_latency.' + extension), dpi=220, bbox_inches='tight',
                    bbox_extra_artists=(first, second))
    plt.close(fig)
    write_json(output / 'pareto_by_dataset.json', frontier)
    return frontier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config['output']).parent
    report = json.loads((root / 'accuracy/full_test/summary.json').read_text())
    assert report['completed']
    datasets = report['datasets']
    pooled = {}
    for name, item in datasets.items():
        if name.startswith('CARLA_new_'):
            for method, sums in item['sums'].items():
                add_statistics(pooled.setdefault(method, {}), sums)
    datasets['CARLA_four'] = dict(frames=174, units='meters', methods={m: finish(v) for m, v in pooled.items()})
    rows = []
    for domain, item in datasets.items():
        speed_name = 'CARLA_new_upper_four' if domain == 'CARLA_four' else domain
        speed = json.loads((root / 'speed' / (speed_name + '.json')).read_text())
        assert speed['completed']
        for method, metrics in item['methods'].items():
            rows.append(dict(dataset=domain, method=method, frames=item['frames'], units=item['units'],
                             **metrics, ms=speed['summary'][method]['depth_output_ms']['median_ms']))
    output = root / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'main_table.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 新数据共同预算重训：完整TEST主表', '',
        '每格依次为 **MAE / RMSE / 毫秒**，均越低越好。精度与速度来自相同部署路径；按每行每项指标分别加粗最优值。', '',
        '| 数据集 | ' + ' | '.join(METHODS.values()) + ' |',
        '|---|' + '---|' * len(METHODS)]
    for domain, item in datasets.items():
        values = [next(r for r in rows if r['dataset'] == domain and r['method'] == m) for m in METHODS]
        cells = []
        for row in values:
            parts = []
            for key in ('MAE', 'RMSE', 'ms'):
                digits = 2 if key == 'ms' else 4
                value = f'{row[key]:.{digits}f}'
                best = min(round(r[key], digits) for r in values)
                parts.append('**' + value + '**' if round(row[key], digits) == best else value)
            cells.append(' / '.join(parts))
        unit = 'm' if item['units'] == 'meters' else 'scene_unit'
        lines.append('| ' + LABELS[domain] + ' (' + unit + ') | ' + ' | '.join(cells) + ' |')
    frontier = plot(rows, output)
    lines.extend(['',
        '协议：七公共域11970组、CARLA174帧×四参考、ETH3D27对×两掩码；同256×448 RGB、真实2/4目、FP32/TF32关闭。公共范围只取本次TRAIN统计，CARLA 2–80m，ETH .1–100m；保留原评价GT掩码。三新模型均公共best，不对CARLA微调。官方Hero训练经历不同，仅作外部系统参考。', '',
        '时间：同一5090、batch1、每域固定16组（不足取全部）、预热10次、随机交替10次，中位完整深度输出耗时；含输入适配，不含IO/传输、首次编译或点云融合。最后一行是实际四参考完整运行，其余为单参考。训练用卡数不参与推理计时。', '',
        '图：横轴NMAE=Σ绝对误差/Σ真值深度，百分比，无单位；这是按真值深度加权的相对误差，与逐像素AbsRel不同，不能消除所有场景难度差异。纵轴为ms。颜色表示方法，形状表示数据集；外圈标记同数据集内的非支配点，灰线仅连接同数据集的前沿。没有将不同数据集连接成方法的虚假Pareto曲线。', '',
        '图包含全部七公共域、CARLA上参考和ETH ALL；其余参考与掩码完整保留在主表，避免同图重复计入CARLA或ETH。所有点均来自实际测试，不因模型优势或劣势筛选。', '',
        '![精度与耗时](accuracy_latency.png)', '',
        '帕累托标记按实测点估计计算，不代表统计显著性或所有方法中的最优；图中两套MVS均为一次范围细化，未将首遍输出混入。官方Hero部分公共TEST图像标识与其公开训练名单重合；本表不称所有数据集均为官方未见域。', '',
        'PDF/SVG保留矢量格式；完整累计量与coverage保存在accuracy分片，CSV另含NMAE与AbsRel。当前TEST历史上已用于研究分析，不称从未访问的盲测。'])
    (output / '主表与精度速度图.md').write_text('\n'.join(lines) + '\n')
    write_json(root / 'delivery.json', dict(completed=True, predictions=sum(v['frames'] for k, v in datasets.items()
        if k != 'CARLA_four') * len(METHODS), rows=len(datasets), methods=list(METHODS),
        report=str(output / '主表与精度速度图.md'), pareto=frontier,
        plot_source_sha256=digest(__file__)))
    print(json.dumps({'completed': True, 'rows': len(datasets), 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
