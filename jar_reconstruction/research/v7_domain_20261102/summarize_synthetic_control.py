"""核对同预算合成训练的V6减法对照，汇总固定每域16VAL。"""

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from research.v7_domain_20261102.fullval_report import MEASURES, REGIONS, percent, read

RUN = ROOT / 'runs/active/v7_domain_20261102/pilots_synthetic_control'
PRIOR = ROOT / 'runs/active/v7_domain_20261102/pilots_synthetic/coordinate_joint_keep'


def exposure_rows(folder):
    rows = [json.loads(line) for path in sorted((folder / 'synthetic_exposures').glob('*.jsonl'))
            for line in path.read_text().splitlines() if line]
    return sorted(rows, key=lambda row: (row['epoch'], row['domain_index'], row['index']))


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def flatten(metrics):
    result = {'depth': metrics['depth']}
    result.update({stage + '_stride8': values for stage, values in metrics['stages_stride8'].items()})
    return result


def main():
    output = RUN / 'analysis'
    output.mkdir(exist_ok=True)
    audits, rows = {}, []
    for phase in ('public', 'carla'):
        current, previous = RUN / phase, PRIOR / phase
        done, old_done = read(current / 'completed.json'), read(previous / 'completed.json')
        audit = read(current / 'synthetic_control_audit.json')
        old_audit = read(previous / 'synthetic_audit.json')
        assert done['completed'] and audit['completed'] and old_done['completed'] and old_audit['completed']
        config, old_config = read(current / 'settings.json'), read(previous / 'settings.json')
        differences = [key for key in set(config) | set(old_config) if config.get(key) != old_config.get(key)]
        assert set(differences) == {'variant', 'device', 'output'}
        plan, old_plan = read(current / 'plan.json'), read(previous / 'plan.json')
        assert plan['checkpoint_sha256'] == old_plan['checkpoint_sha256']
        assert plan['schedule'] == old_plan['schedule'] and plan['validation'] == old_plan['validation']
        for key in ('steps', 'exposures', 'schedule_sha256', 'actual_exposures',
                    'grayscale_exposures', 'actual_view_counts'):
            assert done[key] == old_done[key], (phase, key)
        actual, original = exposure_rows(current), exposure_rows(previous)
        assert actual == original, '逐样本合成曝光或几何覆盖不同'
        initial, old_initial = read(current / 'initial.json'), read(previous / 'initial.json')
        final, old_final = read(current / 'final.json'), read(previous / 'final.json')
        initial_difference = 0.
        for name, base in initial['datasets'].items():
            reports = {'frozen_v6': base, 'control_synthetic': final['datasets'][name],
                       'coordinate_synthetic': old_final['datasets'][name]}
            for report in reports.values():
                assert report['indices'] == base['indices'] and len(report['indices']) == 16
                assert report['manifest_sha256'] == base['manifest_sha256']
            reference = flatten(base['metrics'])
            old_reference = flatten(old_initial['datasets'][name]['metrics'])
            comparison = flatten(old_final['datasets'][name]['metrics'])
            for model, report in reports.items():
                for stage, sections in flatten(report['metrics']).items():
                    for region, values in sections.items():
                        if region not in REGIONS:
                            continue
                        baseline, paired = reference[stage][region], comparison[stage][region]
                        assert values['gt_pixels'] == baseline['gt_pixels'] == paired['gt_pixels']
                        row = {'phase': phase, 'dataset': name, 'units': base['units'], 'model': model,
                               'stage': stage, 'region': region, **values}
                        for measure in MEASURES:
                            a, b = baseline[measure], old_reference[stage][region][measure]
                            if a is not None and b is not None:
                                initial_difference = max(initial_difference, abs(a - b))
                                assert abs(a - b) < max(1e-7, abs(b) * 1e-9)
                            row[measure + '_vs_frozen_pct'] = percent(values[measure], baseline[measure])
                            row[measure + '_vs_coordinate_pct'] = percent(values[measure], paired[measure])
                        rows.append(row)
        audits[phase] = {
            'config_only_differences': sorted(differences), 'checkpoint_start_sha256': plan['checkpoint_sha256'],
            'checkpoint_final_sha256': audit['checkpoint_sha256'], 'coordinate_final_sha256': digest(previous / 'final.pt'),
            'schedule_sha256': done['schedule_sha256'], 'exact_exposure_sha256': canonical_digest(actual),
            'exposures_identical': True, 'steps': done['steps'], 'exposures': done['actual_exposures'],
            'synthetic_exposures': audit['exposure']['synthetic'], 'grayscale_exposures': done['grayscale_exposures'],
            'view_counts': done['actual_view_counts'], 'initial_max_metric_difference': initial_difference,
            'checkpoint_checks': done['checkpoint_checks'], 'no_added_prior': audit['no_added_prior'],
            'optimizer_lrs': audit['optimizer_lrs'], 'frozen_parameters_unchanged': done['frozen_parameters_unchanged'],
            'maximum_gradient_norm': done['maximum_gradient_norm'],
        }
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'audit.json', {'completed': True, 'test_accessed': False, 'phases': audits})
    lines = ['# 删除新增先验的同预算合成训练对照', '',
             '仅每域固定16VAL，公共7域、CARLA4参考；未访问TEST，不是完整VAL。两个分支分别从对应V6起点训练，数据/seed/增强/联合保持/schedule一致，只删除新增坐标先验。没有新增参数组，原解码器学习率3e−6，V6原结构保留。', '',
             '逐样本合成记录（含索引、epoch、抽签、有效覆盖、基线和保留视图）、训练顺序、灰度曝光数、实际视图数均与含先验对照一致；起点权重和清单hash一致，初始指标在浮点容差内相同。有限梯度、冻结参数、严格重载和targets隔离实际通过。', '',
             '|数据集/指标|冻结V6|V6＋同合成训练|坐标先验＋同合成训练|删先验相含先验变化%|',
             '|---|---:|---:|---:|---:|']
    selected = [row for row in rows if row['stage'] == 'depth' and row['region'] == 'full']
    for phase, name in dict.fromkeys((row['phase'], row['dataset']) for row in selected):
        records = {row['model']: row for row in selected if row['phase'] == phase and row['dataset'] == name}
        for measure in MEASURES:
            values = [records[model][measure] for model in ('frozen_v6', 'control_synthetic', 'coordinate_synthetic')]
            difference = records['control_synthetic'][measure + '_vs_coordinate_pct']
            unit = '无量纲' if measure == 'AbsRel' else records['frozen_v6']['units']
            lines.append(f'|{name}/{measure} ({unit})|' + '|'.join(f'{v:.6f}' for v in values) + f'|{difference:+.3f}|')
    lines += ['', '完整阶段、近中远、平滑/深度梯度带指标见metrics.csv。正数表示删除先验后误差上升。合成相对冻结V6还包含继续训练和保持约束效应，不能把所有收益都归因于合成；删除模块的净效应可由相同合成训练对照直接判断。本轮单seed、短预算，不代表完整训练上限。']
    (output / '减法对照.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'completed': True, 'audits': audits}, ensure_ascii=False))


if __name__ == '__main__':
    main()
