"""第二轮固定末步完整VAL报告；只读取已完成的评测结果，不访问TEST。"""

import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from research.v7_domain_20261102.fullval_report import (
    MEASURES, OLD, REGIONS, STAGES, percent, read,
)

RUN = ROOT / 'runs/active/v7_domain_20261102/full_validation_round2'
OUTPUT = RUN / 'analysis'
MODELS = ('frozen_v6', 'fixed_new_only', 'fixed_joint_keep', 'coordinate_new_only',
          'coordinate_joint_keep', 'synthetic_new_only', 'synthetic_joint_keep')
MATCHED = {
    'coordinate_new_only': 'fixed_new_only',
    'coordinate_joint_keep': 'fixed_joint_keep',
    'synthetic_new_only': 'coordinate_new_only',
    'synthetic_joint_keep': 'coordinate_joint_keep',
}
LABELS = ('V6固定', '固定槽/仅新模块', '固定槽/联合保持', '坐标码/仅新模块',
          '坐标码/联合保持', '合成/仅新模块', '合成/联合保持')


def check_frozen(reports):
    """同时核对历史固定V6和新模型的清单、分母与指标。"""
    maximum = {measure: 0. for measure in MEASURES}
    checks = []
    for phase, report in reports.items():
        group = 'public_main' if phase == 'public' else 'carla_new_main'
        previous = read(OLD / group / 'full_val/summary.json')
        previous_audit = read(OLD / group / 'full_val/audit.json')
        frozen = report['models']['frozen_v6']
        assert frozen['sha256'] == previous_audit['weights_sha256']['v6_best']
        for name, domain in frozen['datasets'].items():
            old = previous['datasets'][name]['methods']['v6_best']
            assert domain['frames'] == old['frames']
            assert domain['manifest_sha256'] == previous_audit['manifest_sha256'][name]
            for stage in STAGES:
                compared = old['depth'] if stage == 'depth' else {
                    'full': old['stages_stride8'][stage.removesuffix('_stride8')]}
                for region, original in compared.items():
                    current = domain['metrics'][stage][region]
                    for key in ('gt_pixels', 'predicted_pixels'):
                        assert current[key] == original[key]
                    for measure in MEASURES:
                        a, b = current[measure], original[measure]
                        if a is None or b is None:
                            assert a == b
                            continue
                        difference = abs(a - b)
                        maximum[measure] = max(maximum[measure], difference)
                        assert difference < max(1e-7, abs(b) * 1e-9)
                        checks.append({'phase': phase, 'dataset': name, 'stage': stage,
                                       'region': region, 'metric': measure, 'difference': difference})
    return maximum, checks


def collect_rows(reports):
    rows = []
    for phase, report in reports.items():
        frozen = report['models']['frozen_v6']['datasets']
        for name, base in frozen.items():
            for model in MODELS:
                current = report['models'][model]['datasets'][name]
                assert current['frames'] == base['frames']
                assert current['manifest_sha256'] == base['manifest_sha256']
                for stage in STAGES:
                    for region in REGIONS:
                        values = current['metrics'][stage][region]
                        original = base['metrics'][stage][region]
                        assert values['gt_pixels'] == original['gt_pixels']
                        paired = (report['models'][MATCHED[model]]['datasets'][name]['metrics'][stage][region]
                                  if model in MATCHED else None)
                        row = {'phase': phase, 'dataset': name, 'units': current['units'],
                               'frames': current['frames'], 'model': model, 'stage': stage,
                               'region': region, 'matched_model': MATCHED.get(model), **values}
                        for measure in MEASURES:
                            row[measure + '_vs_frozen_pct'] = percent(values[measure], original[measure])
                            row[measure + '_vs_matched_pct'] = percent(values[measure], paired[measure]) if paired else None
                        rows.append(row)
    return rows


def summaries(rows):
    summary = {}
    for phase in ('public', 'carla'):
        summary[phase] = {}
        for model in MODELS[1:]:
            selected = [row for row in rows if row['phase'] == phase and row['model'] == model]
            summary[phase][model] = {}
            for stage in STAGES:
                summary[phase][model][stage] = {}
                for region in REGIONS:
                    values = [row for row in selected if row['stage'] == stage and row['region'] == region]
                    metrics = {}
                    for measure in MEASURES:
                        changes = [row[measure + '_vs_frozen_pct'] for row in values
                                   if row[measure + '_vs_frozen_pct'] is not None]
                        paired = [row[measure + '_vs_matched_pct'] for row in values
                                  if row[measure + '_vs_matched_pct'] is not None]
                        metrics[measure] = {
                            'domains': len(changes), 'better': sum(value < 0 for value in changes),
                            'worse_gt_01pct': sum(value > .1 for value in changes),
                            'worse_gt_1pct': sum(value > 1 for value in changes),
                            'worst_change_pct': max(changes) if changes else None,
                            'best_change_pct': min(changes) if changes else None,
                            'better_than_matched': sum(value < 0 for value in paired) if paired else None,
                        }
                    summary[phase][model][stage][region] = metrics
    return summary


def text_report(reports, rows, audit):
    summary = audit['summary']
    lookup = {(r['phase'], r['dataset'], r['model'], r['stage'], r['region']): r for r in rows}
    lines = ['# 第二轮七模型完整VAL对比', '',
             '本轮检验固定候选槽先验、连续深度坐标码，以及训练期真实深度重投影合成。仅新模块与联合解码器保持是两种训练条件，后者同时改变解冻范围与保持约束，不能单独归因于保持损失。公共与CARLA分别使用各自固定短训末步；未访问ETH或TEST，也未按本表回选权重。', '',
             f"公共{audit['frames']['public']}组，CARLA四参考共{audit['frames']['carla']}组；实际{len(MODELS) * sum(audit['frames'].values())}次模型预测；{sum(audit['bitwise_checks'].values())}次完整forward逐位核对。冻结V6的权重、清单hash、帧数与GT分母与历史完整VAL一致；MAE/RMSE/AbsRel最大绝对漂移：{audit['frozen_previous_max_difference']}。", '',
             '完整输出256×448与共同stride8格点属于不同统计分母；阶段改善只能在共同格点上比较。Synth、Blended、Dynamic的scene_unit不是米，禁止跨域平均原始MAE。', '',
             '匹配对照：坐标码对固定槽；合成对未合成坐标码；均在相同训练条件下比较。所有模型另与冻结V6比较，后者反映整体保持情况。', '']
    for phase, report in reports.items():
        lines += [f'## {phase} 完整输出指标', '', '|数据集/指标|' + '|'.join(LABELS) + '|',
                  '|---|' + '|'.join(['---:'] * len(MODELS)) + '|']
        for name, domain in report['models']['frozen_v6']['datasets'].items():
            for measure in MEASURES:
                unit = '无量纲' if measure == 'AbsRel' else domain['units']
                values = [lookup[phase, name, model, 'depth', 'full'][measure] for model in MODELS]
                lines.append(f'|{name}/{measure} ({unit})|' + '|'.join(f'{value:.6f}' for value in values) + '|')
        lines += ['', f'## {phase} 新机制相对匹配对照的MAE变化', '',
                  '|数据集|机制/条件|粗层%|细层%|同格点末端%|完整输出%|近距%|中距%|远距%|平滑%|深度梯度带%|',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        coordinates = [('coarse_stride8', 'full'), ('fine_stride8', 'full'), ('final_stride8', 'full'),
                       ('depth', 'full'), ('depth', 'near'), ('depth', 'mid'), ('depth', 'far'),
                       ('depth', 'smooth'), ('depth', 'gradient_band')]
        for name in report['models']['frozen_v6']['datasets']:
            for model in MATCHED:
                values = [lookup[phase, name, model, stage, region]['MAE_vs_matched_pct']
                          for stage, region in coordinates]
                lines.append(f'|{name}|{model}|' + '|'.join(f'{v:+.3f}' if v is not None else 'N/A' for v in values) + '|')
        lines += ['', f'## {phase} 相对V6保持情况', '',
                  '|模型|全图MAE胜域|RMSE胜域|AbsRel胜域|近距MAE胜域|中距MAE胜域|远距MAE胜域|最大MAE退化%|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for model in MODELS[1:]:
            current = summary[phase][model]['depth']
            values = []
            for region, measure in [('full', 'MAE'), ('full', 'RMSE'), ('full', 'AbsRel'),
                                    ('near', 'MAE'), ('mid', 'MAE'), ('far', 'MAE')]:
                metric = current[region][measure]
                values.append(f"{metric['better']}/{metric['domains']}")
            values.append(f"{current['full']['MAE']['worst_change_pct']:+.3f}")
            lines.append('|' + model + '|' + '|'.join(values) + '|')
        lines.append('')
    lines += ['## 解释限制', '',
              '“不下降”应逐域、逐距离、逐指标判断；单seed短训的小数差异没有统计显著性保证。胜域数是描述性计数，不是新的训练选择指标。训练约束只能约束已见训练样本的风险，不能理论保证未见域每像素保持。', '',
              '深度梯度带包含遮挡边缘、细结构与连续倾斜面等，不是植被语义。合成只发生在TRAIN，VAL始终原始RGB与真实标定；二次渲染不等同于新增真实相机观测。ETH并未进入该训练/验证协议，不能据此宣称ETH改善。', '',
              '本表比较固定末步短预算实现，不代表从零完整训练上限。速度与数值重排另见inference_round2_probe/results.json；该速度使用2步smoke权重，不与本表拼接为最终训练模型正式速度。', '',
              '机器可读结果：all_metrics.csv包含全部阶段、区域和指标；audit.json保留历史一致性、胜域计数、严格全域保持判定及证据hash。']
    (OUTPUT / '完整VAL对比.md').write_text('\n'.join(lines) + '\n')


def figure(rows):
    os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    domains = list(dict.fromkeys((row['phase'], row['dataset']) for row in rows))
    matrix = np.array([[next(row['MAE_vs_frozen_pct'] for row in rows
                            if (row['phase'], row['dataset']) == domain and row['model'] == model
                            and row['stage'] == 'depth' and row['region'] == 'full')
                        for model in MODELS[1:]] for domain in domains])
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    limit = max(.5, float(np.max(np.abs(matrix))))
    image = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(6), ['Fixed/new', 'Fixed/joint', 'Coordinate/new', 'Coordinate/joint', 'Synthetic/new', 'Synthetic/joint'],
                  rotation=30, ha='right')
    ax.set_yticks(range(len(domains)), [name for _, name in domains])
    for row in range(len(domains)):
        for column in range(6):
            value = matrix[row, column]
            ax.text(column, row, f'{value:+.2f}%', ha='center', va='center',
                    color='white' if abs(value) > .55 * limit else 'black')
    ax.set_title('Round 2 full VAL: final MAE relative to frozen V6\nNegative is better; fixed short-training endpoints, no TEST')
    fig.colorbar(image, ax=ax, label='MAE change (%)')
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(OUTPUT / ('retention_fullval.' + extension), dpi=180, facecolor='white')
    plt.close(fig)


def main():
    assert read(RUN / 'completed.json')['completed']
    reports = {phase: read(RUN / (phase + '.json')) for phase in ('public', 'carla')}
    for report in reports.values():
        assert report['completed'] and report['split'] == 'full_val' and not report['test_accessed']
        assert set(report['models']) == set(MODELS)
    OUTPUT.mkdir(exist_ok=True)
    maximum, checks = check_frozen(reports)
    rows = collect_rows(reports)
    with (OUTPUT / 'all_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    retention = {}
    for model in MODELS[1:]:
        failures = [{'dataset': row['dataset'], 'region': row['region'], 'metric': measure,
                     'change_pct': row[measure + '_vs_frozen_pct']}
                    for row in rows if row['model'] == model and row['stage'] == 'depth'
                    and row['region'] in ('full', 'near', 'mid', 'far')
                    for measure in MEASURES if row[measure + '_vs_frozen_pct'] is not None
                    and row[measure + '_vs_frozen_pct'] > 1e-7]
        retention[model] = {'all_domains_all_depth_regions_retained': not failures,
                            'worse_combinations': failures}
    audit = {'completed': True, 'test_accessed': False, 'split': 'full_val',
             'frames': {phase: sum(d['frames'] for d in report['models']['frozen_v6']['datasets'].values())
                        for phase, report in reports.items()},
             'bitwise_checks': {phase: report['bitwise_checks'] for phase, report in reports.items()},
             'source_sha256': {str(RUN / (phase + '.json')): digest(RUN / (phase + '.json')) for phase in reports},
             'frozen_previous_max_difference': maximum, 'frozen_previous_checks': checks,
             'summary': summaries(rows), 'retention': retention, 'matched_controls': MATCHED}
    write_json(OUTPUT / 'audit.json', audit)
    text_report(reports, rows, audit)
    figure(rows)
    print(json.dumps({'completed': True, 'frames': audit['frames'], 'previous_max_difference': maximum,
                      'all_retained': {key: value['all_domains_all_depth_regions_retained'] for key, value in retention.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
