"""对完成的固定四视图结果做误差分解，不重新选择模型或评价范围。"""
import json
from pathlib import Path
import sys

PROJECT = Path('/home/liuzonghao/3Dscene/jar_reconstruction')
sys.path.insert(0, str(PROJECT))
from jar.pretraining.comparison import add_statistics
from research.near_mid_evaluation_20261028.metrics import depth_metrics

ROOT = PROJECT / 'runs/active/v7_zero_shot_20261103/analysis'


def main():
    result = json.loads((ROOT / 'results.json').read_text())
    output = {}
    for dataset in ['ETH3D_DSLR', 'TanksAndTemples']:
        rows = [row for row in result['records'].values()
                if row['domain']['dataset'] == dataset and row['domain']['num_views'] == 4]
        totals = {}
        for row in rows:
            for method, regions in row['sums'].items():
                for region, stats in regions.items():
                    add_statistics(totals.setdefault(method, {}).setdefault(region, {}), stats)
        count = totals['v6']['full']['gt_pixels']
        region_report = {}
        for region in ['full', 'near', 'mid', 'far']:
            a, b = totals['v6'][region], totals['v7'][region]
            assert a['gt_pixels'] == a['predicted_pixels'] == b['gt_pixels'] == b['predicted_pixels']
            region_report[region] = dict(
                gt_pixels=a['gt_pixels'],
                methods={m: depth_metrics(r[region]) for m, r in totals.items()},
                v7_minus_v6_error_contribution_per_all_gt=(b['abs_sum'] - a['abs_sum']) / count,
                overprediction_contribution={m: (r[region]['abs_sum'] + r[region]['signed_sum']) / (2 * count)
                                             for m, r in totals.items()},
                underprediction_contribution={m: (r[region]['abs_sum'] - r[region]['signed_sum']) / (2 * count)
                                              for m, r in totals.items()})
        scene_deltas = {row['domain']['scene']: {
            metric: row['metrics']['v7']['full'][metric] - row['metrics']['v6']['full'][metric]
            for metric in ['MAE', 'RMSE', 'AbsRel']} for row in rows}
        output[dataset] = dict(scene_deltas=scene_deltas, regions=region_report,
            scenes_improved={metric: sum(d[metric] < 0 for d in scene_deltas.values())
                             for metric in ['MAE', 'RMSE', 'AbsRel']}, total_scenes=len(rows),
            interpretation='事后误差分解，不证明架构因果。区域贡献按全部GT像素归一；主表仍采用场景等权。')
    (ROOT / 'diagnosis.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    report = ROOT / '零样本泛化结果.md'
    marker = '\n## 实测结论与后续诊断\n'
    lines = [marker, '',
        'V7保留了四视图单参考输出的速度优势，但本轮两个真实数据域的总体精度均未达到共同预算重训MVSAnywhere。',
        'V7相V6在ETH3D的13场景中有10个MAE改善；facade等场景的退化抬高了总体米制误差。此前未分析的六个场景组MAE均改善，但该子集与完整ETH结果重叠，不能当作另一个独立数据集。',
        'ETH整体AbsRel改善而MAE退化并不矛盾：前者按各像素真值深度归一，近距误差的相对权重更高。两个全数据域的场景bootstrap区间均跨零，不能据当前单次训练宣称V7稳定优于V6。', '',
        '以下是距离段内全部GT像素池化的MAE（米），仅用于误差定位。它不同于上面主表的场景等权平均，不能相互替换。近距<5米，中距5–20米，远距≥20米。', '',
        '| 数据集/距离段 | V6 | V7 | 重训MVS | 官方MVS |', '|---|---|---|---|---|']
    for name, item in output.items():
        for region in ['near', 'mid', 'far']:
            values = item['regions'][region]['methods']
            lines.append('| ' + name + '/' + region + ' | ' + ' | '.join(
                f'{values[m]["MAE"]:.4f}' if values[m]['MAE'] is not None else '无GT'
                for m in ['v6', 'v7', 'mvs_retrained', 'mvs_hero']) + ' |')
    lines += ['',
        '误差方向分解显示，V7新增误差总量主要来自过浅预测。ETH近距有改善，中远距却退化；Tanks的新增误差主要来自远距。这里说的是误差量，不是错误像素比例，也不能仅凭最终深度断言问题发生在粗层。',
        '下一轮宜先在训练/验证数据上对初始候选、族选择、细化逐阶段诊断，检查哪些条件触发过浅选面；当前证据不支持继续无差别增加模块。facade和Courthouse的建筑图像也说明，植被不能解释全部失败。', '']
    extent_path = ROOT / 'depth_extent_audit.json'
    if extent_path.exists():
        extents = json.loads(extent_path.read_text())
        outside = sum(v['below_input_near'] + v['above_input_far'] for v in extents.values())
        pixels = sum(v['valid_pixels'] for v in extents.values())
        lines.append(f'推理完成后独立扫描原始评价标签：共同0.1–100米输入范围外有{outside}个有效GT像素，占{100 * outside / pixels:.6f}%。这些像素始终保留在原主表中；未据此改范围或重测。')
    lines += ['', '## 图表与复核文件', '',
        '![视图数量泛化](zero_shot_view_generalization.png)', '',
        '![四视图全部场景差异](zero_shot_scene_differences.png)', '',
        '两图同时保存PDF/SVG。逐场景与距离段数据见per_scene.csv；误差分解见diagnosis.json，范围核查见depth_extent_audit.json。',
        '上一级verification.json记录实际2076次预测、权重/源码冻结、全部coverage=1以及服务器eno5直连下载核验。']
    report.write_text(report.read_text().split(marker)[0] + '\n'.join(lines) + '\n')
    for name, d in output.items():
        print(name, d['scenes_improved'])
        print({r: round(v['v7_minus_v6_error_contribution_per_all_gt'], 6) for r, v in d['regions'].items()})


if __name__ == '__main__':
    main()
