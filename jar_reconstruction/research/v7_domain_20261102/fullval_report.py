"""完整VAL七模型的独立结果核验与报告；不读取TEST、不训练。"""

import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json

RUN = ROOT / 'runs/active/v7_domain_20261102/full_validation_round1'
OUTPUT = RUN / 'analysis'
OLD = ROOT / 'runs/active/rulermvs_v6_20261101/full_training/evaluation_recovery'
MODELS = ('frozen_v6', 'control_original', 'control_augment', 'projective_original',
          'projective_augment', 'prior_original', 'prior_augment')
LABELS = ('V6固定', 'Control原增强', 'Control新增强', 'Center原增强', 'Center新增强', 'Prior原增强', 'Prior新增强')
STAGES = ('coarse_stride8', 'fine_stride8', 'final_stride8', 'depth')
REGIONS = ('full', 'near', 'mid', 'far', 'smooth', 'gradient_band')
MEASURES = ('MAE', 'RMSE', 'AbsRel')


def read(path):
    return json.loads(Path(path).read_text())


def percent(current, reference):
    return 100 * (current / reference - 1) if current is not None and reference is not None and reference else None


def matched_control(name):
    return 'control_' + name.rsplit('_', 1)[1] if name.startswith(('prior_', 'projective_')) else None


def main():
    done = read(RUN / 'completed.json')
    assert done['completed']
    reports = {phase: read(RUN / (phase + '.json')) for phase in ('public', 'carla')}
    for report in reports.values():
        assert report['completed'] and report['split'] == 'full_val' and not report['test_accessed']
        assert set(report['models']) == set(MODELS)
    OUTPUT.mkdir(exist_ok=True)
    checks, maximum, rows = [], {}, []
    for phase, report in reports.items():
        previous_group = 'public_main' if phase == 'public' else 'carla_new_main'
        previous = read(OLD / previous_group / 'full_val/summary.json')
        previous_audit = read(OLD / previous_group / 'full_val/audit.json')
        frozen = report['models']['frozen_v6']
        expected_hash = previous_audit['weights_sha256']['v6_best']
        assert frozen['sha256'] == expected_hash
        for name, domain in frozen['datasets'].items():
            old = previous['datasets'][name]['methods']['v6_best']
            assert domain['frames'] == old['frames']
            assert domain['manifest_sha256'] == previous_audit['manifest_sha256'][name]
            for stage in STAGES:
                compared = old['depth'] if stage == 'depth' else {'full': old['stages_stride8'][stage.removesuffix('_stride8')]}
                for region, old_metrics in compared.items():
                    new = domain['metrics'][stage][region]
                    for count in ('gt_pixels', 'predicted_pixels'):
                        assert new[count] == old_metrics[count], (phase, name, stage, region, count)
                    for measure in MEASURES:
                        a, b = new[measure], old_metrics[measure]
                        if a is None or b is None:
                            assert a == b
                            continue
                        delta = abs(a - b)
                        maximum[measure] = max(maximum.get(measure, 0.), delta)
                        checks.append({'phase': phase, 'dataset': name, 'stage': stage,
                                       'region': region, 'metric': measure, 'absolute_difference': delta})
                        assert delta < max(1e-7, abs(b) * 1e-9), (phase, name, stage, region, measure, a, b)
            for model in MODELS:
                values = report['models'][model]['datasets'][name]
                assert values['frames'] == domain['frames'] and values['manifest_sha256'] == domain['manifest_sha256']
                for stage in STAGES:
                    for region in REGIONS:
                        metric = values['metrics'][stage][region]
                        base = domain['metrics'][stage][region]
                        assert metric['gt_pixels'] == base['gt_pixels']
                        control_name = matched_control(model)
                        control = report['models'][control_name]['datasets'][name]['metrics'][stage][region] if control_name else None
                        row = {'phase': phase, 'dataset': name, 'units': values['units'], 'frames': values['frames'],
                               'model': model, 'stage': stage, 'region': region, **metric}
                        for measure in MEASURES:
                            row[measure + '_vs_frozen_pct'] = percent(metric[measure], base[measure])
                            row[measure + '_vs_matched_control_pct'] = percent(metric[measure], control[measure]) if control else None
                        rows.append(row)
    with (OUTPUT / 'all_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = {}
    for phase in reports:
        comparisons[phase] = {}
        for model in MODELS[1:]:
            local = [row for row in rows if row['phase'] == phase and row['model'] == model]
            summary = {}
            for stage in STAGES:
                summary[stage] = {}
                for region in REGIONS:
                    selected = [row for row in local if row['stage'] == stage and row['region'] == region]
                    summary[stage][region] = {}
                    for measure in MEASURES:
                        vals = [r[measure + '_vs_frozen_pct'] for r in selected if r[measure + '_vs_frozen_pct'] is not None]
                        matched = [r[measure + '_vs_matched_control_pct'] for r in selected if r[measure + '_vs_matched_control_pct'] is not None]
                        summary[stage][region][measure] = {
                            'eligible_domains': len(vals), 'better_than_frozen': sum(v < 0 for v in vals),
                            'worse_than_frozen_gt_01pct': sum(v > .1 for v in vals),
                            'worse_than_frozen_gt_1pct': sum(v > 1 for v in vals),
                            'worst_change_vs_frozen_pct': max(vals) if vals else None,
                            'best_change_vs_frozen_pct': min(vals) if vals else None,
                            'better_than_matched_control': sum(v < 0 for v in matched) if matched else None,
                        }
            comparisons[phase][model] = summary
    audit = {'completed': True, 'split': 'full_val', 'test_accessed': False,
             'source_sha256': {str(RUN / (phase + '.json')): digest(RUN / (phase + '.json')) for phase in reports},
             'frozen_previous_comparison_maximum_absolute_difference': maximum,
             'frozen_previous_comparison': checks, 'comparisons': comparisons,
             'frames': {phase: sum(d['frames'] for d in report['models']['frozen_v6']['datasets'].values())
                        for phase, report in reports.items()},
             'bitwise_checks': {phase: report['bitwise_checks'] for phase, report in reports.items()},
             'limits': ['完整VAL，非TEST；单seed固定短训末步；冻结编码器，不能宣称全预算训练上限。',
                        '后验胜域数不代表显著性，近零差异不作为大幅架构收益。',
                        '单位逐域保留，不将scene_unit作为米或跨域平均原始MAE。']}
    write_json(OUTPUT / 'audit.json', audit)
    lines = ['# 第一轮七模型完整VAL对比', '',
             '冻结V6、同预算Control原/新增强、投影中心原/新增强、上下文Prior原/新增强。所有模型使用固定短训末步，公共与CARLA分别加载各自权重。本报告仅完整VAL，不是新TEST精度。', '',
             f"公共{audit['frames']['public']}参考，CARLA四参考合计{audit['frames']['carla']}；共{7 * sum(audit['frames'].values())}模型预测。编码共享的独立完整forward逐位检查共{sum(audit['bitwise_checks'].values())}次。冻结V6的权重hash、清单hash、样本数与GT分母与上一轮完整VAL一致；指标最大绝对差：{maximum}。", '',
             'full/depth是完整256×448输出；coarse/fine/final_stride8是共同stride8格点，不能直接把二者数值相减声称阶段增益。', '']
    for phase, report in reports.items():
        lines += [f'## {phase} 完整输出', '', '|数据集/指标|' + '|'.join(LABELS) + '|',
                  '|---|' + '|'.join(['---:'] * len(MODELS)) + '|']
        for name, domain in report['models']['frozen_v6']['datasets'].items():
            for measure in MEASURES:
                values = [report['models'][model]['datasets'][name]['metrics']['depth']['full'][measure] for model in MODELS]
                units = '无量纲' if measure == 'AbsRel' else domain['units']
                lines.append('|' + name + ' / ' + measure + f" ({units})|" + '|'.join(f'{v:.5f}' for v in values) + '|')
        lines += ['', f'## {phase} 新增模块相对匹配Control的净MAE变化', '',
                  '|数据集|模块/增强|粗层%|细层%|同格点末端%|完整输出%|近距%|中距%|远距%|平滑区%|边界带%|',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        coordinates = [('coarse_stride8', 'full'), ('fine_stride8', 'full'), ('final_stride8', 'full'),
                       ('depth', 'full'), ('depth', 'near'), ('depth', 'mid'), ('depth', 'far'),
                       ('depth', 'smooth'), ('depth', 'gradient_band')]
        for name in report['models']['frozen_v6']['datasets']:
            for model in ('projective_original', 'projective_augment', 'prior_original', 'prior_augment'):
                changes = []
                for stage, region in coordinates:
                    item = next(row for row in rows if row['phase'] == phase and row['dataset'] == name and row['model'] == model
                                and row['stage'] == stage and row['region'] == region)
                    value = item['MAE_vs_matched_control_pct']
                    changes.append(f'{value:+.3f}' if value is not None else 'N/A')
                lines.append('|' + name + '|' + model + '|' + '|'.join(changes) + '|')
        lines += ['', f'## {phase} 相对固定V6的保持情况', '',
                  '|模型|完整MAE胜域|完整RMSE胜域|完整AbsRel胜域|近距MAE胜域|中距MAE胜域|远距MAE胜域|最大完整MAE退化%|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for model in MODELS[1:]:
            summaries = comparisons[phase][model]['depth']
            values = []
            for region, measure in [('full', 'MAE'), ('full', 'RMSE'), ('full', 'AbsRel'),
                                    ('near', 'MAE'), ('mid', 'MAE'), ('far', 'MAE')]:
                metric = summaries[region][measure]
                values.append(f"{metric['better_than_frozen']}/{metric['eligible_domains']}")
            values.append(f"{summaries['full']['MAE']['worst_change_vs_frozen_pct']:+.3f}")
            lines.append('|' + model + '|' + '|'.join(values) + '|')
        lines.append('')
    speed = read(ROOT / 'runs/active/v7_domain_20261102/speed_probe_initial/results.json')
    timing = speed['datasets']['CARLA_new_upper']['methods']
    costs = '/'.join(f"{timing['fast/' + name]['median_ms']:.3f}" for name in ('control', 'projective', 'prior'))
    increments = '/'.join(f"{timing['fast/' + name]['increment_vs_control_ms']:.3f}" for name in ('projective', 'prior'))
    lines += ['## 结构开销与决策边界', '',
              f'先前同卡零初始化研究测速（实际90次/配置，其他卡训练，非最终训练权重正式部署测速）：四目fast V6/Center/Prior为{costs}ms；新增开销约{increments}ms。不能用129参数就断言Center几乎免费，其额外精确投影与小算子调度是实际代价。', '',
              'Prior相Control的净收益应在相同增强条件下比较。Control新增强相原增强才隔离数据干预；相固定V6还含继续训练效应。Center会通过联合训练影响粗层，不能将训练后的粗层差异误称为推理时直接改变粗候选。', '',
              '本表只检验实际短预算实现，不否定充分训练的新方法原理；但是当前是否满足其他域、近远距与RMSE保持，须读取全表，不能只报目标域MAE。旧V6不替换、TEST不用于本轮选方案。', '',
              '完整机器可读数据：all_metrics.csv（全部阶段/区域/指标、相固定V6与匹配Control变化）、audit.json（原始一致性及胜域/退化计数）。']
    (OUTPUT / '完整VAL对比.md').write_text('\n'.join(lines) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    labels = [(phase, name) for phase, report in reports.items() for name in report['models']['frozen_v6']['datasets']]
    matrix = np.array([[next(row['MAE_vs_frozen_pct'] for row in rows
                            if row['phase'] == phase and row['dataset'] == name and row['model'] == model
                            and row['stage'] == 'depth' and row['region'] == 'full')
                        for model in MODELS[1:]] for phase, name in labels])
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    limit = max(1., float(np.max(np.abs(matrix))))
    image = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(6), ['Control/original', 'Control/augmented', 'Center/original', 'Center/augmented', 'Prior/original', 'Prior/augmented'], rotation=30, ha='right')
    ax.set_yticks(range(len(labels)), [name for _, name in labels])
    for row in range(len(labels)):
        for column in range(6):
            value = matrix[row, column]
            ax.text(column, row, f'{value:+.2f}%', ha='center', va='center',
                    color='white' if abs(value) > .55 * limit else 'black')
    ax.set_title('Full VAL: final depth MAE change relative to frozen V6\nNegative is better; one seed, matched short adaptation, no TEST')
    fig.colorbar(image, ax=ax, label='MAE change (%)')
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(OUTPUT / ('retention_fullval.' + extension), dpi=180, facecolor='white')
    plt.close(fig)
    print(json.dumps({'completed': True, 'frames': audit['frames'], 'previous_max_difference': maximum}, ensure_ascii=False))


if __name__ == '__main__':
    main()
