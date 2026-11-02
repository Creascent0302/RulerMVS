"""完整VAL减法对照：复用既有三模型结果，新增V6合成训练结果。"""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from research.v7_domain_20261102.fullval_report import MEASURES, REGIONS, STAGES, percent, read

BASE = ROOT / 'runs/active/v7_domain_20261102'
RUN = BASE / 'full_validation_control'
OUTPUT = RUN / 'analysis'
MODELS = ('frozen_v6', 'coordinate_joint_keep', 'synthetic_joint_keep', 'control_synthetic')
LABELS = ('冻结V6', '坐标先验/未合成', '坐标先验/合成', 'V6原结构/合成')


def main():
    assert read(RUN / 'completed.json')['completed']
    assert read(BASE / 'full_validation_round2/completed.json')['completed']
    plan = read(ROOT / 'research/v7_domain_20261102/configs_fullval_control/plan.json')
    assert all(digest(path) == signature for path, signature in plan['source_sha256'].items())
    OUTPUT.mkdir(exist_ok=True)
    reports, rows, sources = {}, [], {}
    frame_counts, checks = {}, {}
    for phase in ('public', 'carla'):
        path = RUN / (phase + '.json')
        old_path = BASE / 'full_validation_round2' / (phase + '.json')
        current, old = read(path), read(old_path)
        for report in (current, old):
            assert report['completed'] and report['split'] == 'full_val' and not report['test_accessed']
        assert set(current['models']) == {'control_synthetic'}
        new_model = current['models']['control_synthetic']
        trained = BASE / 'pilots_synthetic_control' / phase
        train_audit = read(trained / 'synthetic_control_audit.json')
        assert new_model['sha256'] == train_audit['checkpoint_sha256']
        reports[phase] = {model: old['models'][model] for model in MODELS[:-1]}
        reports[phase]['control_synthetic'] = new_model
        frame_counts[phase] = sum(domain['frames'] for domain in new_model['datasets'].values())
        checks[phase] = current['bitwise_checks']
        sources.update({str(path): digest(path), str(old_path): digest(old_path)})
        for name, base in reports[phase]['frozen_v6']['datasets'].items():
            prior = reports[phase]['synthetic_joint_keep']['datasets'][name]
            for model, record in reports[phase].items():
                domain = record['datasets'][name]
                assert domain['frames'] == base['frames'] and domain['units'] == base['units']
                assert domain['manifest_sha256'] == base['manifest_sha256']
                for stage in STAGES:
                    for region in REGIONS:
                        metric = domain['metrics'][stage][region]
                        original, paired = base['metrics'][stage][region], prior['metrics'][stage][region]
                        assert metric['gt_pixels'] == original['gt_pixels'] == paired['gt_pixels']
                        assert metric['predicted_pixels'] == original['predicted_pixels']
                        row = {'phase': phase, 'dataset': name, 'units': domain['units'],
                               'frames': domain['frames'], 'model': model, 'stage': stage,
                               'region': region, **metric}
                        for measure in MEASURES:
                            row[measure + '_vs_frozen_pct'] = percent(metric[measure], original[measure])
                            row[measure + '_vs_coordinate_synthetic_pct'] = percent(metric[measure], paired[measure])
                        rows.append(row)
    with (OUTPUT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for phase in reports:
        summary[phase] = {}
        for stage in STAGES:
            summary[phase][stage] = {}
            for region in REGIONS:
                current = [row for row in rows if row['model'] == 'control_synthetic' and row['phase'] == phase
                           and row['stage'] == stage and row['region'] == region]
                summary[phase][stage][region] = {}
                for measure in MEASURES:
                    valid = [row for row in current if row[measure] is not None]
                    values = [row[measure + '_vs_frozen_pct'] for row in valid]
                    paired = [row[measure + '_vs_coordinate_synthetic_pct'] for row in valid]
                    summary[phase][stage][region][measure] = {
                        'domains': len(valid), 'better_than_frozen': sum(value < 0 for value in values),
                        'better_than_coordinate_synthetic': sum(value < 0 for value in paired),
                        'worst_vs_frozen_pct': max(values) if values else None,
                        'worst_vs_coordinate_synthetic_pct': max(paired) if paired else None,
                    }
    audit = {'completed': True, 'test_accessed': False, 'split': 'full_val', 'new_predictions': sum(frame_counts.values()),
             'frames': frame_counts, 'bitwise_checks': checks, 'source_sha256': sources,
             'evaluation_sources_unchanged': True, 'summary': summary,
             'training_pair_audit': str(BASE / 'pilots_synthetic_control/analysis/audit.json'),
             'scope': '仅control重新推理；其余三模型引用已完成同口径完整VAL，不冒充新GPU执行',
             'causal_limit': '删除prior相含prior合成组是匹配对照；相未合成coordinate同时改变模块和数据，不能单独归因合成'}
    write_json(OUTPUT / 'audit.json', audit)
    lookup = {(row['phase'], row['dataset'], row['model'], row['stage'], row['region']): row for row in rows}
    lines = ['# V6原结构合成训练：完整VAL减法对照', '',
             f"新增公共{frame_counts['public']}＋CARLA{frame_counts['carla']}＝{sum(frame_counts.values())}次预测，{sum(checks.values())}次完整forward逐位核验。其余三模型直接引用已完成第二轮完整VAL，清单、帧数、GT及预测分母一致。此报告不读取TEST/ETH、不据结果更换固定训练末步。", '',
             '仅删除prior的训练对照已核对：起点SHA相同，训练顺序、逐样本合成覆盖/基线/抽签、灰度曝光与实际视图数相同，原解码器学习率3e−6。删去先验后网络仍是原始V6，不新增推理操作；尚未将这份末步权重重新正式测速。', '',
             'full/depth为256×448输出；coarse/fine/final_stride8在共同格点比较。不同单位不平均；scene_unit不是米。', '']
    for phase, report in reports.items():
        lines += [f'## {phase} 完整输出', '', '|数据集/指标|' + '|'.join(LABELS) + '|删除prior相含prior变化%|',
                  '|---|---:|---:|---:|---:|---:|']
        for name, domain in report['frozen_v6']['datasets'].items():
            for measure in MEASURES:
                values = [lookup[phase, name, model, 'depth', 'full'][measure] for model in MODELS]
                change = lookup[phase, name, 'control_synthetic', 'depth', 'full'][measure + '_vs_coordinate_synthetic_pct']
                units = '无量纲' if measure == 'AbsRel' else domain['units']
                lines.append(f'|{name}/{measure} ({units})|' + '|'.join(f'{value:.6f}' for value in values) + f'|{change:+.3f}|')
        for target, key in [('冻结V6', '_vs_frozen_pct'), ('同合成训练含prior', '_vs_coordinate_synthetic_pct')]:
            lines += ['', f'### 删除prior后的分阶段MAE相{target}变化', '',
                      '|数据集|粗层%|细层%|同格点末端%|完整输出%|近距%|中距%|远距%|平滑%|梯度带%|',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            coordinates = [('coarse_stride8', 'full'), ('fine_stride8', 'full'), ('final_stride8', 'full'),
                           ('depth', 'full'), ('depth', 'near'), ('depth', 'mid'), ('depth', 'far'),
                           ('depth', 'smooth'), ('depth', 'gradient_band')]
            for name in report['frozen_v6']['datasets']:
                changes = [lookup[phase, name, 'control_synthetic', stage, region]['MAE' + key]
                           for stage, region in coordinates]
                lines.append(f'|{name}|' + '|'.join(f'{v:+.3f}' if v is not None else 'N/A' for v in changes) + '|')
        lines.append('')
    lines += ['## 结论边界', '',
              '判断新增先验的必要性应看同合成训练的匹配对照，不只看相冻结V6的总收益。相冻结V6包含训练预算、数据增强、合成与保持约束的共同效果；相未合成coordinate又同时改变先验和合成，不能单独归因。', '',
              '最终全图变好不代表全部阶段或距离变好；完整区域MAE较低也不代表RMSE/AbsRel保持。边界由深度梯度定义，不能当作植被语义结论。单seed短预算的细小差异不作显著性宣称；V6旧权重及默认不改变。', '',
              '完整机器可读指标见metrics.csv，源文件/输入一致性与胜域数见audit.json。']
    (OUTPUT / '完整减法对照.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'completed': True, 'frames': frame_counts, 'bitwise_checks': checks,
                      'full_summary': {phase: summary[phase]['depth']['full'] for phase in summary}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
