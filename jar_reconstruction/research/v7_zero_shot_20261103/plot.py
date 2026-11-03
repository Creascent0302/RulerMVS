"""用固定全部场景绘制零样本跨域与视图数量结果。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/liuzonghao/3Dscene/jar_reconstruction/runs/active/v7_zero_shot_20261103/analysis')
METHODS = ['v6', 'v7', 'mvs_retrained', 'mvs_hero']
LABELS = ['RulerMVS-V6', 'RulerMVS-V7', 'MVSAnywhere retrained', 'MVSAnywhere official']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']


def save(fig, name):
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(ROOT / f'{name}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    data = json.loads((ROOT / 'results.json').read_text())
    assert data['completed']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    for ax, group, title in zip(axes, ['ETH3D_DSLR_13场景', 'TanksAndTemples_4场景'],
                                ['ETH3D DSLR: 13 scenes', 'Tanks and Temples: 4 scenes']):
        for method, label, color in zip(METHODS, LABELS, COLORS):
            values = [100 * data['groups'][f'{group}/{n}views']['methods'][method]['scene_macro']['AbsRel']
                      for n in [2, 4, 8]]
            ax.plot([2, 4, 8], values, color=color, marker='o', label=label, linewidth=2)
        ax.set(title=title, xlabel='Number of input views', ylabel='Scene-macro AbsRel (%)')
        ax.set_xticks([2, 4, 8])
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, -.06))
    fig.tight_layout(rect=(0, .08, 1, 1))
    save(fig, 'zero_shot_view_generalization')

    records = [(name, row) for name, row in data['records'].items() if row['domain']['num_views'] == 4]
    labels = [row['domain']['scene'].removeprefix('Tanks_') for _, row in records]
    difference = [100 * (row['metrics']['v7']['full']['AbsRel'] - row['metrics']['v6']['full']['AbsRel'])
                  for _, row in records]
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    index = np.arange(len(records))
    ax.barh(index, difference, color=[COLORS[1] if d > 0 else COLORS[0] for d in difference], alpha=.85)
    ax.set_yticks(index, labels)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=.8)
    ax.axhline(12.5, color='.6', linewidth=.7, linestyle='--')
    ax.set_xlabel('V7 minus V6 AbsRel (percentage points; negative favors V7)')
    ax.set_title('Four-view zero-shot evaluation: every predefined scene')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    save(fig, 'zero_shot_scene_differences')


if __name__ == '__main__':
    main()
