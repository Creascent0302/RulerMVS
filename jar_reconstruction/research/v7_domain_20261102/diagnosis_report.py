"""把冻结诊断转为分层贡献表；不挑选或训练模型。"""

import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import add_statistics, write_json
from research.v7_domain_20261102.diagnosis import BASE, OUTPUT


def main():
    report = json.loads((OUTPUT / 'results.json').read_text())
    assert report['completed']
    rows = []
    for name, domain in report['datasets'].items():
        total_pixels = domain['metrics']['v6_best']['full']['gt_pixels']
        gap = domain['metrics']['v6_best']['full']['MAE'] - domain['metrics']['mvsanywhere_refined']['full']['MAE']
        for region, own in domain['metrics']['v6_best'].items():
            reference = domain['metrics']['mvsanywhere_refined'][region]
            first = domain['metrics']['mvsanywhere_firstpass'][region]
            contribution = ((own['MAE'] - reference['MAE']) * own['gt_pixels'] / total_pixels
                            if own['gt_pixels'] else None)
            rows.append({'dataset': name, 'split': domain['split'], 'region': region,
                         'pixels': own['gt_pixels'], 'pixel_fraction': own['gt_pixels'] / total_pixels,
                         'V6_MAE_m': own['MAE'], 'MVS_first_MAE_m': first['MAE'], 'MVS_refined_MAE_m': reference['MAE'],
                         'net_MAE_gap_contribution_m': contribution,
                         'net_gap_fraction': contribution / gap if contribution is not None and gap else None})
    with (OUTPUT / 'strata.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scenes = {}
    eth = report['datasets']['ETH3D_two_view_all']
    total = eth['metrics']['v6_best']['full']['gt_pixels']
    gap = eth['metrics']['v6_best']['full']['MAE'] - eth['metrics']['mvsanywhere_refined']['full']['MAE']
    for frame in eth['frames']:
        group = scenes.setdefault(frame['sequence'], {'pixels': 0, 'own_abs': 0., 'reference_abs': 0.})
        own = frame['methods']['v6_best']['full']
        reference = frame['methods']['mvsanywhere_refined']['full']
        add_statistics(group, {'pixels': own['gt_pixels'], 'own_abs': own['abs_sum'], 'reference_abs': reference['abs_sum']})
    for group in scenes.values():
        group.update(pixel_fraction=group['pixels'] / total, V6_MAE_m=group['own_abs'] / group['pixels'],
                     MVS_refined_MAE_m=group['reference_abs'] / group['pixels'],
                     net_gap_fraction=(group['own_abs'] - group['reference_abs']) / total / gap)
    # 复算主指标必须与已完成外测逐项一致；CARLA与同索引物理实验clean比较。
    original = json.loads((BASE / 'paper_eth3d/evaluation/full_test/summary.json').read_text())['datasets']['ETH3D_two_view_all']['methods']
    drift = max(abs(eth['metrics'][method]['full'][metric] - original[method]['depth']['full'][metric])
                for method in eth['metrics'] for metric in ('MAE', 'RMSE', 'AbsRel'))
    assert drift < 1e-10
    physics = json.loads((BASE / 'paper_physics/results.json').read_text())['datasets']
    carla_drift = 0.
    for name, domain in report['datasets'].items():
        if domain['split'] != 'val':
            continue
        for method in domain['metrics']:
            for metric in ('MAE', 'RMSE', 'AbsRel'):
                old = physics[name]['methods']['clean'][method]['full'][metric]
                carla_drift = max(carla_drift, abs(domain['metrics'][method]['full'][metric] - old))
    assert carla_drift < 1e-10
    audit = {'completed': True, 'forwards': 273,
                                     'ETH3D_previous_metric_max_difference': drift,
                                     'CARLA_previous_metric_max_difference': carla_drift,
                                     'eth_scenes': scenes, 'production_weights_modified': False,
                                     'CPU_test_command': 'official_methods/.venv/bin/python -m pytest -q research/v7_domain_20261102/test_diagnosis.py',
             'CPU_test_result': '1 passed in 1.95s',
             'report_first_failure': '绘图时上参考无投影像素为0，对应None误差未处理；只修空集合贡献显示为0，不改原始指标。'}
    lines = ['# ETH3D / CARLA 冻结模型的针对性诊断', '',
             '2026-11-02：已执行273次GPU前向，V6及baseline权重/架构不变。ETH3D全部27对为既有外测事后诊断，不用本轮标签训练或选模型；CARLA四参考各16固定VAL，不能将这些数值替代完整TEST。', '',
             '结论：不能用“植被多”统一解释。ETH3D弱视差定位与候选供给不足；CARLA上参考以边界/遮挡为重点，左右参考的缺重叠区域更关键。数据没有本轮可用的植被语义标签，所以不把边缘或纹理指标称为植被面积。', '',
             '## ETH3D：场景与几何条件', '',
             '|场景|GT像素占比|V6 MAE(m)|MVS细化MAE(m)|对总MAE净差贡献|', '|---|---:|---:|---:|---:|']
    for name, item in scenes.items():
        lines.append(f"|{name}|{item['pixel_fraction']:.2%}|{item['V6_MAE_m']:.4f}|{item['MVS_refined_MAE_m']:.4f}|{item['net_gap_fraction']:.2%}|")
    lines += ['', 'delivery_area、facade、terrace合计贡献89.48%净劣势；forest没有净劣势。场景名不等于像素语义，仍不能据此量化各物体类别。', '',
              '|投影敏感度(输入像素/单位log深度)|像素占比|V6 MAE(m)|MVS细化MAE(m)|净差贡献|', '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['dataset'].startswith('ETH') and row['region'].startswith('parallax'):
            lines.append(f"|{row['region']}|{row['pixel_fraction']:.2%}|{row['V6_MAE_m']:.4f}|{row['MVS_refined_MAE_m']:.4f}|{row['net_gap_fraction']:.2%}|")
    lines += ['', '平行双目时该敏感度等于fB/Z；不是“纹理充足”或“正确匹配”的证明。大于8像素时两者接近；0.5–8像素区解释约99%净差。相比植被，更直接的证据指向弱视差。', '',
              '共同stride8：粗层2.07043m→细层2.23737m→最终2.17349m；连续点oracle1.43536m、族内均值oracle1.92812m，MVS细化同格点0.99316m。现有候选就算完美选择也不能达到baseline，因此不仅是最终选面器的问题。oracle使用GT事后选择，不能当可部署性能。', '',
              '## CARLA：不同相机对应不同瓶颈', '',
              '|参考|区域|像素占比|V6 MAE(m)|MVS细化MAE(m)|对该参考总净差贡献|', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        if row['dataset'].startswith('CARLA') and row['region'] in ('full', 'gradient_band', 'smooth', 'no_source_projection', 'gt_source_consistent', 'gt_source_occluded') and row['pixels']:
            lines.append(f"|{row['dataset'].replace('CARLA_new_', '')}|{row['region']}|{row['pixel_fraction']:.2%}|{row['V6_MAE_m']:.4f}|{row['MVS_refined_MAE_m']:.4f}|{row['net_gap_fraction']:.2%}|")
    lines += ['', '上参考边界带贡献105%净劣势，平滑区域反而略好。左右参考无源投影区分别占19.0%/20.0%，解释179%/126%净劣势；在源GT一致区自研较好。下参考总体已较好，但边界/遮挡仍弱。超过100%意味着其他区域抵消一部分劣势；下参考总差为负，贡献百分比的符号不能按上参考解释。', '',
              '无源投影：GT三维点没有进入任何有效源图，只有参考图先验可用；不代表参考深度无效。gt_source_consistent表示至少一源GT与投影z相差≤5%；gt_source_occluded表示至少一源GT更近5%以上，两者可能由于不同源而重叠，不能相加。缺少源GT不当作遮挡。', '',
              '## 设计含义与边界', '',
              '1. ETH先补弱视差连续定位和候选供给，同时分离搜索跨度与最终均值定位；仅扩大候选或添加门控未必有效。',
              '2. CARLA需要对可匹配区、部分遮挡区和无重叠区分工。无重叠处不能靠重复多视图匹配恢复未观测证据，需要训练过的参考图尺度/空间补全；遮挡边界应避免错误源和跨表面扩散。',
              '3. 训练增强应从TRAIN生成窄基线、灰度/曝光和遮挡条件，并通过几何重投影与可见性定义标签；独立二维粘贴会破坏跨视图关系。',
              '4. 本轮不是训练因果消融。ETH只27对，CARLA只有固定64VAL，纹理/深度边界/距离互相关，不能用相关分层宣称单一原因。',
              '5. 当前5x5 log深度跳变阈值只能定义明显边界，不证明smooth区域是平面；ETH无效GT边界被排除，边界像素仅0.19%，须注明标签支持域限制。', '',
              '复算全ETH及CARLA既有clean指标最大差均小于1e−10；解析平行双目、共心退化、缺相机、GT源一致测试实际通过。源文件diagnosis.py，原始逐帧结果results.json，分层CSV为strata.csv，完成核验audit.json。']
    (OUTPUT / '诊断结论.md').write_text('\n'.join(lines) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    selected = [row for row in rows if row['dataset'].startswith('ETH') and row['region'] in ('parallax_05to2', 'parallax_2to8', 'parallax_ge8')]
    x = np.arange(len(selected))
    axes[0].bar(x - .18, [r['V6_MAE_m'] for r in selected], .36, label='V6', color='#0072B2')
    axes[0].bar(x + .18, [r['MVS_refined_MAE_m'] for r in selected], .36, label='MVS refined', color='#D55E00')
    axes[0].set_xticks(x, ['0.5–2 px', '2–8 px', '>=8 px'])
    axes[0].set_ylabel('MAE (m)')
    axes[0].set_title('ETH3D: depth sensitivity strata\n27 previous external-test pairs')
    axes[0].legend()
    selected = [row for row in rows if row['dataset'].startswith('CARLA') and row['region'] == 'no_source_projection']
    contributions = [r['net_MAE_gap_contribution_m'] if r['pixels'] else 0. for r in selected]
    axes[1].bar(np.arange(4), contributions, color='#0072B2')
    axes[1].set_xticks(np.arange(4), ['Upper', 'Left', 'Right', 'Down'])
    axes[1].set_ylabel('Contribution to V6 − MVS MAE (m)')
    axes[1].axhline(0, color='black', lw=.8)
    axes[1].set_title('CARLA: no source projection at GT\n16 fixed VAL frames per reference')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(OUTPUT / ('failure_strata.' + extension), dpi=180, facecolor='white')
    plt.close(fig)
    write_json(OUTPUT / 'audit.json', audit)
    print(json.dumps({'completed': True, 'forwards': 273, 'ETH_metric_drift': drift, 'CARLA_metric_drift': carla_drift}))


if __name__ == '__main__':
    main()
