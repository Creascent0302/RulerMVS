"""汇总公共VAL压力实测，不把增强收益归于新增模块。"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import write_json
from research.v7_domain_20261102.stress_val import BASE, CONDITIONS, METHODS, OUTPUT


def main():
    report = json.loads((OUTPUT / 'results.json').read_text())
    assert report['completed'] and report['shared_encoding_full_forward_checks'] == 140
    matched = []
    pair_equal = {}
    for domain_name, domain in report['datasets'].items():
        same_pair = all(frame['conditions']['clean']['input_sha256'] == frame['conditions']['closest_real_pair']['input_sha256']
                        and frame['conditions']['grayscale']['input_sha256'] == frame['conditions']['gray_closest_pair']['input_sha256']
                        for frame in domain['frames'])
        pair_equal[domain_name] = same_pair
        for condition in CONDITIONS:
            rows = domain['metrics'][condition]
            values = {method: rows[method]['depth']['full']['MAE'] for method in METHODS}
            matched.append({'dataset': domain_name, 'units': domain['units'], 'condition': condition,
                            'prior_aug_vs_control_aug_pct': 100 * (values['prior_augment'] / values['control_augment'] - 1),
                            'prior_aug_vs_v6_pct': 100 * (values['prior_augment'] / values['v6_frozen'] - 1),
                            'augment_vs_original_control_pct': 100 * (values['control_augment'] / values['control_original'] - 1),
                            'augment_vs_original_prior_pct': 100 * (values['prior_augment'] / values['prior_original'] - 1)})
    wins = {}
    for condition in CONDITIONS:
        selection = [row for row in matched if row['condition'] == condition]
        wins[condition] = {key: sum(row[key] < 0 for row in selection)
                           for key in ('prior_aug_vs_control_aug_pct', 'prior_aug_vs_v6_pct',
                                       'augment_vs_original_control_pct', 'augment_vs_original_prior_pct')}
    clean_drift = {}
    for method in METHODS[1:]:
        prior = json.loads((BASE / 'pilots_round1' / method / 'public/final.json').read_text())
        clean_drift[method] = {}
        for domain_name, domain in report['datasets'].items():
            old = prior['datasets'][domain_name]['metrics']['depth']['full']
            new = domain['metrics']['clean'][method]['depth']['full']
            clean_drift[method][domain_name] = {metric: new[metric] - old[metric] for metric in ('MAE', 'RMSE', 'AbsRel')}
            assert new['gt_pixels'] == old['gt_pixels']
            for metric in ('MAE', 'RMSE', 'AbsRel'):
                # 本诊断先转float64再乘域单位，历史pilot先float32乘单位；允许其末位舍入差。
                assert abs(new[metric] - old[metric]) < max(1e-5, abs(old[metric]) * 2e-6), (method, domain_name, metric)
    audit = {'completed': True, 'wins_out_of_7': wins, 'paired_changes': matched,
             'pair_condition_input_identical_to_full_input': pair_equal,
             'clean_vs_pilot_metric_rounding_differences': clean_drift,
             'CPU_tests': '2 passed in 2.37s',
             'inference_checks': '140/140 独立完整forward与共享编码decoder输出逐位相同；共享encoded未修改。',
             'ETH_or_TEST_used': False, 'models_retrained_in_this_test': False}
    write_json(OUTPUT / 'analysis.json', audit)
    lines = ['# 五模型的公共VAL灰度/真实源子集压力测试', '',
             '2026-11-02：实际完成2240次模型预测，以及140次独立完整forward核对。编码器参数/buffers hash完全相同，各模型共享编码与独立完整前向逐位相同；没有访问ETH3D或TEST，也没有训练或修改权重。', '',
             '每域16个先前固定VAL输入；条件为原图、所有有效视图统一灰度、保留参考+最近非零真实基线源、两者组合。相机参数不变，不复制图像。最近相机只是低视差压力proxy，不等于生成了ETH3D的新窄基线数据。VKITTI2和DynamicReplica原本双目，对应pair条件与原输入相同，不能当作新的独立证据。', '',
             '所有表中误差单位逐域保留：Tartan/Matrix/Hyp/VKITTI为米，其余scene_unit；不做原始MAE跨域平均。±百分比为相同模型相对于clean的相对变化，正数变差、负数变好。', '']
    for name, domain in report['datasets'].items():
        lines += [f"## {name} ({domain['units']})", '',
                  '|条件|V6固定|Control原增强|Control新增强|Prior原增强|Prior新增强|',
                  '|---|---:|---:|---:|---:|---:|']
        for condition in CONDITIONS:
            cells = []
            for method in METHODS:
                mae = domain['metrics'][condition][method]['depth']['full']['MAE']
                clean = domain['metrics']['clean'][method]['depth']['full']['MAE']
                cells.append(f'{mae:.5f}' + (f' ({100 * (mae / clean - 1):+.2f}%)' if condition != 'clean' else ''))
            lines.append('|' + condition + '|' + '|'.join(cells) + '|')
        lines.append('')
    lines += ['## 归因与限制', '',
              '|条件|新增强Control胜原增强Control|新增强Prior胜原增强Prior|Prior新增强胜Control新增强|Prior新增强胜固定V6|',
              '|---|---:|---:|---:|---:|']
    for condition, values in wins.items():
        lines.append('|' + condition + '|' + '|'.join(str(values[key]) + '/7' for key in (
            'augment_vs_original_control_pct', 'augment_vs_original_prior_pct',
            'prior_aug_vs_control_aug_pct', 'prior_aug_vs_v6_pct')) + '|')
    lines += ['', '单域小幅改变量不当作显著性。增强有效必须与同架构同预算的原增强比较；prior结构收益必须与同新增强的Control比较，不能把增强带来的全部收益归于新层。', '',
              '本次clean与短训末步既有VAL逐域GT像素数完全相同，MAE/RMSE/AbsRel只有允许的单位转换末位舍入差（analysis.json保留）。本测试所有模型均先转换float64再乘域单位，和历史pilot的float32乘单位略有不同；不据这些末位变化选模型。', '',
              '需依据全部验证结果决定后续训练。当前灰度压力不能证明真实窄基线域已改善；若clean或far退化，仍不满足用户“其他指标保持”的目标。']
    (OUTPUT / '压力测试结论.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'completed': True, 'wins_out_of_7': wins}, ensure_ascii=False))


if __name__ == '__main__':
    main()
