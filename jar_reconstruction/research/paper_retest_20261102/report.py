"""从本次重测和已完成论文实验生成一份简洁报告，不补造缺失结果。"""

import csv
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import add_statistics, digest, write_json
from research.near_mid_evaluation_20261028.metrics import depth_metrics, surface_metrics

BASE = ROOT / 'runs/active/paper_retest_20261102'
OLD = ROOT / 'runs/active/rulermvs_v6_20261101/full_training'
DOC = ROOT / 'docs/paper_results_consolidated_20261102.md'
LABELS = {
    'v6_best': 'V6 普通', 'v6_fast': 'V6 快速',
    'mvsanywhere_firstpass': 'MVSAnywhere 首遍',
    'mvsanywhere_refined': 'MVSAnywhere 细化',
    'fast_mvsnet_dtu': 'Fast-MVSNet（DTU）',
    'patchmatchnet_best': 'PatchmatchNet（适配）',
    'diffmvs_best': 'DiffMVS（先导训练）',
    'simplerecon_carla_old': 'SimpleRecon（旧CARLA）',
}
ORDER = ['v6_fast', 'mvsanywhere_refined', 'fast_mvsnet_dtu',
         'patchmatchnet_best', 'diffmvs_best', 'simplerecon_carla_old']
SHORT = {
    'TartanAir_forest': 'TartanAir', 'MatrixCity_street': 'MatrixCity',
    'Hypersim_subset': 'Hypersim', 'MVS_Synth': 'MVS-Synth',
    'DynamicReplica_valid': 'DynamicReplica', 'CARLA_new_upper': 'CARLA 上',
    'CARLA_new_stereo_left': 'CARLA 左', 'CARLA_new_stereo_right': 'CARLA 右',
    'CARLA_new_stereo_down': 'CARLA 下', 'ETH3D_two_view_all': 'ETH3D ALL',
    'ETH3D_two_view_nocc': 'ETH3D 非遮挡',
}
DISCUSSION = {
    'TartanAir_forest': '该域主要衡量已覆盖森林环境中的跨轨迹迁移，而非跨地图泛化。MAE接近时还需比较RMSE与远距误差，不能把很小均值差异解释为稳定优势；官方训练名单与本地测试轨迹重合，也限制了零样本叙述。',
    'MatrixCity_street': '静态城市、多位置观察和较规则结构提供了可利用的几何证据，结果与显式候选推理适应该类条件的假设相容。官方训练使用big_city、本地使用small_city，城市内容并不一致；这一相关性尚不能证明场景规则性是唯一原因。',
    'Hypersim_subset': '室内域的结果需要结合复杂材质、弱纹理以及训练覆盖共同解释。本轮权重训练时该域仅597组训练配组，既有实验也未证明所有误差都集中在遮挡边缘，因此不能仅通过增加边界模块或将问题归因于植被解释差距。',
    'MVS_Synth': '原始场景尺度使绝对误差数值较大，必须结合无单位AbsRel理解，不能把该域数字直接标为米或与CARLA相加平均。官方训练名单覆盖部分测试帧，故此处优势支持当前数据条件下的系统能力，不足以证明普遍未见域泛化。',
    'VKITTI2': '同步双目的固定基线在远处产生较弱深度敏感性。较低近距相对误差可以与较高远距尾部误差并存，从而出现AbsRel较优、RMSE较差的组合；判断整体优势应保留分段结果，而非只选择MAE这一列。',
    'BlendedMVS': '该域具有多样场景尺度与观察几何，既有同预算消融中去掉源几何条件带来较明显退化，支持融合阶段利用标定的设计。绝对误差仍采用原场景单位，公开baseline配组文件的不完整也使预训练曝光无法完全认证。',
    'DynamicReplica_valid': '同步双目减少了跨时间物体运动的不一致，但较小基线与有限训练序列仍会增加歧义。既有关闭搜索扩宽的实验改善了族内均值oracle而降低离散候选覆盖，说明候选跨度与连续读出可能比单纯增加候选数更关键；这属于机制证据，尚不是已部署升级。',
    'CARLA_new_upper': '上参考的跨轨迹结果说明V6并未全面继承V5在这一相机上的优势。历史同格点分析发现退化先出现在粗层，细层反而挽回部分误差，因此不能将全部问题归咎于后端交接。',
    'CARLA_new_stereo_left': '左参考与上参考分布不同，自研训练包含参考轮换，而PatchmatchNet和DiffMVS当前适配主要监督上参考。该表能够评价实际跨参考迁移，但不能据此将差距全部归因于架构。',
    'CARLA_new_stereo_right': '右参考同样检验相机角色迁移。像素深度与表面覆盖应联合解读：更好的双向最近邻均值并不必然意味着同射线定位更精确，后文保留Accuracy、Completeness和F-score以区分两类行为。',
    'CARLA_new_stereo_down': '下参考是检验方法适用性的重要补充，因为它具有与上参考不同的可见区域和遮挡关系。若深度、两个方向表面距离及F-score同时改善，证据比单一Chamfer改善更一致；该优势仍不意味着其他相机同样获益。',
    'ETH3D_two_view_all': '真实灰度、约6cm基线与合成预训练域存在明显差异。此前按场景诊断显示森林并非统一失败来源，建筑场景的弱视差是更明确的瓶颈之一。该结果约束了真实环境泛化主张，但不能替代原论文的DSLR多视图ETH3D评测。',
    'ETH3D_two_view_nocc': '非遮挡评价使用同一批图像和不同GT掩码，并非另一批独立测试样本。若差距在非遮挡区域仍存在，就不能将整体失败仅解释为遮挡；候选定位、纹理证据及真实域适应仍需分别检验。',
}


def read(path):
    return json.loads(Path(path).read_text())


def number(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows] + ['']


def highlight(rows, lower=(), higher=(), group_column=None):
    """在可比较的行之间逐指标加粗；显示精度相同者视为并列。"""
    groups = {}
    for row in rows:
        key = row[group_column] if group_column is not None else None
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        for column, choose in [(i, min) for i in lower] + [(i, max) for i in higher]:
            values = []
            for row in group:
                match = re.match(r'^[+-]?\d+(?:\.\d+)?', str(row[column]))
                if match:
                    values.append((row, float(match.group())))
            if not values:
                continue
            best = choose(value for _, value in values)
            for row, value in values:
                if value == best:
                    row[column] = f'**{row[column]}**'
    return rows


def main_matrix(names, domains, speed, pooled, four):
    rows = []
    for name in [*names, 'CARLA_four_references']:
        joint = name == 'CARLA_four_references'
        methods = pooled if joint else domains[name]['methods']
        timings = four['summary'] if joint else speed[name]['summary']
        unit = 'm' if joint or domains[name]['units'] == 'meters' else 'scene_unit'
        metrics = []
        for method in ORDER:
            if method in methods:
                depth = methods[method]['depth']['full']
                metrics.append([method, number(depth['MAE']), number(depth['RMSE']),
                                number(timings[method]['depth_output_ms']['median_ms'], 2)])
        highlight(metrics, lower=(1, 2, 3))
        cells = {row[0]: f'MAE {row[1]}<br>RMSE {row[2]}<br>{row[3]} ms' for row in metrics}
        label = 'CARLA 四参考联合' if joint else SHORT.get(name, name)
        rows.append([f'{label} ({unit})', *[cells.get(method, '—') for method in ORDER]])
    return table(['数据集 / 评价分区', *[LABELS[m] for m in ORDER]], rows)


def csv_write(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def historical_table(text, marker):
    start = text.index(marker)
    lines = text[start:].splitlines()
    selected = []
    for line in lines:
        if not line.startswith('|'):
            break
        selected.append(line)
    rows = [[cell.strip() for cell in line.strip('|').split('|')] for line in selected[2:]]
    if marker.startswith('| 完整VAL分区'):
        for row in rows:
            candidates = [[i, row[i]] for i in range(2, len(row))]
            highlight(candidates, lower=(1,))
            for i, value in candidates:
                row[i] = value
    elif marker.startswith('| 分区 | 共同粗好'):
        for row in rows:
            for column, maximize in ((1, False), (2, True)):
                parts = [value.strip() for value in row[column].split('→')]
                values = [[i, part] for i, part in enumerate(parts)]
                highlight(values, lower=() if maximize else (1,), higher=(1,) if maximize else ())
                row[column] = ' → '.join(value for _, value in values)
            if row[3].startswith(('-', '−')):
                row[3] = f'**{row[3]}**'
    elif marker.startswith('| 分区 | 关源几何'):
        for row in rows:
            for column in range(1, len(row)):
                if row[column].startswith('+'):
                    row[column] = f'**{row[column]}**'
    return selected[:2] + ['| ' + ' | '.join(row) + ' |' for row in rows] + ['']


def collect(plan):
    domains, speed, configs, sources = {}, {}, {}, {}
    depth_rows, surface_rows, stage_rows, speed_rows = [], [], [], []
    for group, settings in plan['groups'].items():
        folder = Path(settings['output']) / 'full_test'
        summary = read(folder / 'summary.json')
        if not summary['completed']:
            raise RuntimeError('完整精度测试尚未完成：' + group)
        config = read(settings['config'])
        sources[str(folder / 'summary.json')] = digest(folder / 'summary.json')
        for domain in config['datasets']:
            name = domain['name']
            row = summary['datasets'][name]
            assert set(row['methods']) == set(config['methods'])
            assert all(v['frames'] == settings['frames'][name] for v in row['methods'].values())
            domains[name], configs[name] = row, domain
            filename = BASE / 'speed' / (name + '.json')
            speed[name] = read(filename)
            assert speed[name]['completed']
            assert speed[name]['test_manifest_sha256'] == digest(domain['manifests']['test'])
            sources[str(filename)] = digest(filename)
            for method, result in row['methods'].items():
                common = {'dataset': name, 'method': method, 'units': row['units'],
                          'views': row['num_views'], 'frames': result['frames']}
                time = speed[name]['summary'][method]
                resource = speed[name]['isolated_resources'][method]
                timing = {f'{field}_{key}': value for field, dist in time.items() for key, value in dist.items()}
                speed_rows.append({**common, 'references_per_call': 1, **timing, **resource})
                for region, values in result['depth'].items():
                    depth_rows.append({**common, 'region': region, **values,
                                       **(timing if region == 'full' else {})})
                surface_rows.append({**common, **result['surface']})
                for stage, values in result['stages_stride8'].items():
                    stage_rows.append({**common, 'stage': stage, **values})
    for filename, expected in {**plan['weights'], **plan['manifests']}.items():
        if digest(filename) != expected:
            raise RuntimeError('冻结权重或测试清单在重测中改变：' + filename)
    return domains, speed, configs, sources, depth_rows, surface_rows, stage_rows, speed_rows


def pooled_carla(plan):
    root = Path(plan['groups']['carla']['output']) / 'full_test'
    totals = {}
    for file in sorted(root.glob('CARLA*/part_*.json')):
        for name, row in read(file)['totals'].items():
            target = totals.setdefault(name, {'depth': {}, 'surface': {}, 'thresholds': row['thresholds']})
            for region, values in row['depth'].items():
                add_statistics(target['depth'].setdefault(region, {}), values)
            add_statistics(target['surface'], row['surface'])
    return {name: {'depth': {region: depth_metrics(values) for region, values in row['depth'].items()},
                   'surface': surface_metrics(row['surface'], row['thresholds'])}
            for name, row in totals.items()}


def plots(domains, speed, output):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(BASE / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names = [name for name in domains if not name.startswith('CARLA') and not name.endswith('nocc')]
    names.append('CARLA_new_upper')
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout='constrained')
    labels = ['V6 fast', 'MVSA refined', 'Fast-MVSNet', 'PatchmatchNet', 'DiffMVS', 'SimpleRecon']
    colors = ['#005e63', '#bd5735', '#6b709e', '#5980c2', '#9c6b99', '#777777']
    for ax, name in zip(axes.flat, names):
        for method, label, color in zip(ORDER, labels, colors):
            if method not in domains[name]['methods']:
                continue
            value = domains[name]['methods'][method]['depth']['full']
            latency = speed[name]['summary'][method]['depth_output_ms']['median_ms']
            ax.scatter(latency, value['AbsRel'], label=label, color=color,
                       marker='*' if method == 'v6_fast' else 'o', s=85 if method == 'v6_fast' else 32)
        title = {'CARLA_new_upper': 'CARLA upper', 'ETH3D_two_view_all': 'ETH3D stereo'}.get(name, SHORT.get(name, name))
        ax.set(title=title, xlabel='Single-reference latency (ms)', ylabel='AbsRel')
        ax.set_yscale('log')
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'accuracy_latency.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    plan = read(BASE / 'plan.json')
    domains, speed, configs, sources, depth_rows, surface_rows, stage_rows, speed_rows = collect(plan)
    historical_sources = {
        OLD / 'paper_handoff_ablation/summary.json': 'completed',
        OLD / 'paper_geometry_ablation/analysis/summary.json': 'complete_full_val',
        OLD / 'paper_robustness/results.json': 'completed',
        OLD / 'paper_mechanism/results.json': 'completed',
    }
    for file, complete_field in historical_sources.items():
        if not read(file).get(complete_field):
            raise RuntimeError('引用的历史实验未完成：' + str(file))
        sources[str(file)] = digest(file)
    output = BASE / 'analysis'
    output.mkdir(exist_ok=True)
    four_path = BASE / 'speed/CARLA_new_upper_four.json'
    four = read(four_path)
    assert four['completed']
    sources[str(four_path)] = digest(four_path)
    for method, time in four['summary'].items():
        timing = {f'{field}_{key}': value for field, dist in time.items()
                  for key, value in dist.items()}
        speed_rows.append({
            'dataset': 'CARLA_four_references', 'method': method, 'units': 'meters',
            'views': 4, 'frames': domains['CARLA_new_upper']['methods'][method]['frames'],
            'references_per_call': 4, **timing, **four['isolated_resources'][method],
        })
    for file, rows in [('depth_all_metrics.csv', depth_rows), ('surface_all_metrics.csv', surface_rows),
                       ('stages_all_metrics.csv', stage_rows), ('latency_all_metrics.csv', speed_rows)]:
        csv_write(output / file, rows)
    pooled = pooled_carla(plan)
    write_json(output / 'carla_pooled.json', pooled)
    plots(domains, speed, output)
    historical = (ROOT / 'docs/v6_paper_results_20261101.md').read_text()
    public = [name for name in domains if not name.startswith(('CARLA', 'ETH3D'))]
    wins = sum(domains[name]['methods']['v6_fast']['depth']['full']['MAE'] <
               domains[name]['methods']['mvsanywhere_refined']['depth']['full']['MAE']
               for name in public)
    faster = sum(speed[name]['summary']['v6_fast']['depth_output_ms']['median_ms'] <
                 speed[name]['summary']['mvsanywhere_refined']['depth_output_ms']['median_ms']
                 for name in public)
    lines = ['# RulerMVS-V6：主实验重测与论文实验汇总', '',
        '更新于2026年11月2日。本轮主实验重新执行完整TEST推理，并补齐逐数据集耗时；后半部分汇总此前实际完成的机制与消融实验，明确保留VAL、短预算训练和测试集的区别。所有数字由原始JSON生成，不按本次TEST更换权重。', '',
        f'**主要结论。** V6快速版在七个公共域中有{wins}个域的MAE低于MVSAnywhere细化版，{faster}个域的单参考深度耗时更低。优势随场景变化，尚未全面胜出；ETH3D、Hypersim及CARLA前三参考仍是明确短板。历史同预算消融支持候选交接保护正确粗面，但同时显示纠错能力的取舍。', '',
        '**实验口径。** 七个公共域11970组、CARLA174帧×四参考、ETH3D27对×两种GT掩码；ETH两种掩码不增加独立图像数量。所有方法接收相同256×448 RGB信息、实际N=2/4及已知标定，batch1、FP32、TF32关闭，不使用tele或GT确定输入范围。公共域保留全部原有效GT，CARLA评价2–80m。自研公共/ETH使用公共VAL-best，CARLA使用独立适配VAL-best，不能声称单权重覆盖全部域。', '',
        f"耗时在同一RTX5090上测量，PyTorch {four['torch']}。每域预先按清单等间隔取16组TEST输入，不读标签；首组预热10次，每组随机交替重复10次。主表为网络、输入适配和共同深度格点输出的中位耗时，排除IO、传输、首次编译及点云融合；Hz为1000除以该中位耗时，只表示单参考深度频率。P95、平均值、原始样本和热文件缓存下含IO/H2D/D2H的数据管线耗时另存完整表。", '',
        '显存另行逐方法独立驻留，在每域首个计时输入上测量allocated峰值，不含其他模型，也不是整个测试集的最坏情况上界。精度推理采用八卡分片，计时在其全部结束后独占单卡进行。同期数据盘仍有独立下载整理任务，含IO列仅作为热缓存离线管线诊断，不代表独占磁盘性能。', '',
        'MVSAnywhere启用现有快速cost-volume实现及位置编码缓存；V6快速版使用既有代数复用和局部torch.compile。编译策略并非完全相同，因此加速属于所测部署实现，不能全部归因于架构创新。', '',
        'MAE、RMSE、AbsRel分别为绝对误差均值、平方误差均值的平方根及相对绝对误差均值；RMSE对大误差更敏感。δ1/2/3为预测与GT的双向比值小于1.25、1.25²、1.25³的像素比例，缺预测计为失败。表内覆盖率表示有效预测占有效GT的比例；不足100%时误差是条件误差，不能只凭较低MAE排名。米制与原场景单位分别报告，不跨单位平均。', '',
        '**基线状态。** MVSAnywhere使用官方Hero的一次范围细化结果；Fast-MVSNet使用DTU发布权重；PatchmatchNet为DTU初始化后的新CARLA小规模适配；DiffMVS为未获任务权重的CARLA先导训练；SimpleRecon为旧CARLA四图训练权重，只参加四图数据。不同预训练及预算使本表属于当前系统比较，而非同预算架构上限比较。MVSFormer++缺少任务权重，SimpleRecon四图权重不兼容双目，均不填假指标。占据方法没有共同占据标签，不混入深度排名。', '',
        '公开数据审计还表明，VKITTI2、部分TartanAir和MVS-Synth测试图像标识出现在官方MVSAnywhere训练名单；本地ETH3D为灰度双目，非论文RMVD的DSLR多视图。因此本表不能称全部域的双方零样本比较或原论文指标复现，详见[数据审计](mvsanywhere_dataset_comparison_20261102.md)。', '',
        '## 主实验：完整测试精度与逐域速度', '',
        '本文部署结果只列V6快速版和MVSAnywhere细化版及其余基线。V6的深度、阶段和表面指标均来自快速路径的实际输出。附带δ1/2/3、SqRel、对数误差、偏差、分段及全部表面阈值的[完整深度表](../runs/active/paper_retest_20261102/analysis/depth_all_metrics.csv)、[表面表](../runs/active/paper_retest_20261102/analysis/surface_all_metrics.csv)及[耗时表](../runs/active/paper_retest_20261102/analysis/latency_all_metrics.csv)与正文同源。', '']
    ordered_domains = ['TartanAir_forest', 'MatrixCity_street', 'Hypersim_subset', 'MVS_Synth', 'VKITTI2',
                       'BlendedMVS', 'DynamicReplica_valid', 'CARLA_new_upper', 'CARLA_new_stereo_left',
                       'CARLA_new_stereo_right', 'CARLA_new_stereo_down', 'ETH3D_two_view_all', 'ETH3D_two_view_nocc']
    lines += ['**主表：数据集 × 方法。** 每格依次为MAE、RMSE、运行时间（ms），三项均越低越好。“RMAE”按既有评测中的RMSE（均方根误差）理解。除最后一行CARLA四参考联合外，时间均为单参考完整深度输出的中位耗时。', '',
              '同一数据分区的每项最优值分别加粗，按显示精度并列时均加粗；加粗不表示统计显著性。后续详细表也按同一分区、同一指标标注最优值，消融表按均值比较。未支持组合用“—”，不会补值。覆盖率不足100%的误差仍是条件误差，须结合详细表的覆盖率解读。', '']
    lines += main_matrix(ordered_domains, domains, speed, pooled, four)
    lines += ['### 分数据集详细指标与分析', '']
    for name in ordered_domains:
        row = domains[name]
        unit = 'm' if row['units'] == 'meters' else 'scene_unit'
        frames = row['methods']['v6_fast']['frames']
        lines += [f"**{SHORT.get(name, name)}：{frames}组，{row['num_views']}图输入，误差单位{unit}。**", '']
        rows = []
        for method in ORDER:
            if method not in row['methods']:
                continue
            metrics = row['methods'][method]['depth']['full']
            timing = speed[name]['summary'][method]['depth_output_ms']
            rows.append([LABELS[method], number(metrics['MAE']), number(metrics['RMSE']),
                         number(metrics['AbsRel']),
                         *[number(100 * metrics[f'delta_{i}'], 2) for i in (1, 2, 3)],
                         number(100 * metrics['coverage'], 3),
                         number(timing['median_ms'], 2), number(timing['hz'], 2)])
        lines += table(['方法', 'MAE↓', 'RMSE↓', 'AbsRel↓', 'δ1%↑', 'δ2%↑', 'δ3%↑', '覆盖率%', 'ms↓', 'Hz↑'],
                       highlight(rows, lower=(1, 2, 3, 8), higher=(4, 5, 6, 9)))
        a = row['methods']['v6_fast']['depth']['full']
        b = row['methods']['mvsanywhere_refined']['depth']['full']
        t = speed[name]['summary']
        change = 100 * (a['MAE'] / b['MAE'] - 1)
        latency = 100 * (t['v6_fast']['depth_output_ms']['median_ms'] / t['mvsanywhere_refined']['depth_output_ms']['median_ms'] - 1)
        lines += [f"V6快速版相MVSAnywhere细化版的MAE变化为{change:+.2f}%，单参考耗时变化为{latency:+.2f}%。"
                  + DISCUSSION[name], '']
    lines += ['![各域精度与耗时，纵轴为对数AbsRel](../runs/active/paper_retest_20261102/analysis/accuracy_latency.png)', '',
              '**CARLA四参考联合结果。** 深度从四个参考的全部有效像素累加计算，RMSE不作相机均值；速度实际执行四参考，V6共享编码，baseline轮换原生参考入口。', '']
    rows = []
    for method in ORDER:
        m = pooled[method]['depth']['full']
        t = four['summary'][method]['depth_output_ms']
        rows.append([LABELS[method], number(m['MAE']), number(m['RMSE']), number(m['AbsRel']),
                     number(t['median_ms'], 2), number(t['hz'], 2)])
    lines += table(['方法', 'MAE/m↓', 'RMSE/m↓', 'AbsRel↓', '四参考ms↓', '四参考Hz↑'],
                   highlight(rows, lower=(1, 2, 3, 4), higher=(5,)))
    lines += ['四参考计时是四张深度图的推理成本，不含跨参考点云融合或车辆感知系统其他模块，不能直接等同于车端完整三维系统频率。快速执行包含局部编译和代数复用，其收益不应全部作为网络结构创新。', '',
              '## 距离分区、可见表面与阶段定位', '',
              '下表保留V6快速版与MVSAnywhere细化版的完整分域几何对照。近/中/远分界在米制域为10/40m，非米制域为5/20倍固定depth_scale。F采用各域预登记阈值列表的第三档，具体阈值随行注明；Chamfer为双向最近邻距离均值，不是平方距离。', '']
    rows = []
    for name in ordered_domains:
        for method in ('v6_fast', 'mvsanywhere_refined'):
            v = domains[name]['methods'][method]
            s = v['surface']
            threshold = s['thresholds'][2]
            rows.append([SHORT.get(name, name), LABELS[method],
                         *[number(v['depth'][r]['MAE']) for r in ('near', 'mid', 'far')],
                         number(s['Accuracy']), number(s['Completeness']), number(s['Chamfer_L1']),
                         f"{number(s[f'Fscore_{threshold:g}'] * 100, 2)}@{threshold:g}"])
    lines += table(['数据', '方法', '近MAE', '中MAE', '远MAE', 'Acc↓', 'Compl↓', 'Chamfer↓', 'F%↑@阈值'],
                   highlight(rows, lower=(2, 3, 4, 5, 6, 7), higher=(8,), group_column=0))
    lines += ['可见表面评价只对同一参考有效区域反投影，每侧固定1024个查询点，对另一侧完整点集计算最近邻，不进行ICP或真值尺度配准。像素深度比较同一射线，最近邻距离允许相邻射线之间配对，因此二者并不等价。较小Completeness也可能伴随预测分散或定位偏差，必须结合Accuracy与F-score解释，不能仅据Chamfer证明完整场景重建更优。其余基线的全部表面指标和各阈值F-score保留在同源完整表中。', '',
              'V6的粗层、细层和最终输出在共同stride8格点比较，以避免采样分辨率混杂。MVSAnywhere输出不能直接当作其原生粗候选，故本表仅定位自研内部误差演化，不把两者非同义中间量强行排名。', '']
    rows = []
    for name in ordered_domains:
        v = domains[name]['methods']['v6_fast']['stages_stride8']
        rows.append([SHORT.get(name, name), *[number(v[s]['MAE']) for s in ('coarse', 'fine', 'final')]])
    for row in rows:
        values = [[i, row[i]] for i in (1, 2, 3)]
        highlight(values, lower=(1,))
        for i, value in values:
            row[i] = value
    lines += table(['数据', '粗层MAE', '细层MAE', '最终同格点MAE'], rows)
    lines += ['阶段指标应分别解释候选生成、候选利用与稠密传播。粗层改善而最终改善不足时，问题可能出现在细化选面或上采样；粗层已经退化时，不能只归因于后端交接。候选中存在较好深度也不保证实际读出正确，离线oracle仅衡量候选集合的潜力，并非可部署预测。', '',
              '**边界代理评价。** 用GT相邻像素log深度差大于0.1提取边缘，再扩张为5×5邻域带；与剩余平滑区域分开汇总。它刻画深度不连续区域，不是有真实实例标签的小障碍检测。', '']
    rows = []
    for name in ordered_domains:
        m = domains[name]['methods']
        rows.append([SHORT.get(name, name),
                     *[number(m[method]['depth'][region]['MAE'])
                       for region in ('gradient_band', 'smooth')
                       for method in ('v6_fast', 'mvsanywhere_refined')]])
    for row in rows:
        for pair in ((1, 2), (3, 4)):
            values = [[i, row[i]] for i in pair]
            highlight(values, lower=(1,))
            for i, value in values:
                row[i] = value
    lines += table(['数据', '边界V6', '边界MVS细化', '平滑V6', '平滑MVS细化'], rows)
    lines += ['若误差差距在平滑区域同样明显，就不足以支持“主要由细碎边界造成”的单一解释；反之，边界收益也不能直接推出小物体召回提高，因为这里未按物体尺寸或实例独立统计。此前预登记固定CARLA帧的[四参考可视化](../runs/active/rulermvs_v6_20261101/full_training/paper_qualitative/CARLA_fixed_test_four_references.pdf)保留成功与失败区域，属于历史同权重定性证据，未按本轮改善大小重新挑选图像。', '',
              '**场景级稳定性。** 历史场景重采样分析表明，MVS-Synth的优势比单轨迹结果有更广的场景支持，BlendedMVS的轻微均值优势则不够稳定。VKITTI2多种变体共享Scene20，CARLA四参考也不增加独立轨迹数；不能用连续帧数量代替场景多样性。相关原始统计留在历史报告，本版不将其数值转标为快速部署结果。', '',
              '## 已完成的论文消融与机制实验（历史结果）', '',
              '以下实验均来自此前已完成的原始记录，本轮未重新训练。三种子消融使用共同V5起点和相同样本顺序，公共每次896更新、CARLA256更新，冻结编码器、匹配骨干及CNN桥；全部公共7957组或CARLA411参考VAL评估。表为MAE均值±样本标准差，既不是TEST，也不是从零全预算上限。', '']
    lines += historical_table(historical, '| 完整VAL分区 | 单位 | 同预算V5 |')
    lines += ['候选族与自适应交接相同预算V5在10/11分区降低全图MAE，说明有限预算下存在可重复的适配收益；然而自适应门仅在6/11分区优于常数门，不能将复杂条件化描述为普遍必要。该实验通过匹配继续训练预算，降低了“新增模块只是获得额外训练”这一解释的影响，但仍不能代替充分联合训练的因果对照。', '',
              '为区分保留与纠错，进一步在双方粗层共同正确或共同错误的像素集合上计算转移率，正确阈值固定为5%相对误差。箭头两侧加粗较优值，配对MAE差为负时加粗。使用共同支持集合可避免不同模型粗层正确点数量不同引起的分母混杂。', '']
    lines += historical_table(historical, '| 分区 | 共同粗好上的破坏率：')
    lines += ['交接在11/11分区减少已正确粗面的破坏，却在9/11分区降低错误粗面的纠正率。因而其主要作用是保护已有证据，并非同时增强所有纠错行为；论文应将这种保守性作为机制与边界共同呈现，而非只报告破坏率下降。', '',
              '**标尺各用途。** 四种配置分别进行公共与CARLA同预算短训，共8次训练、单seed、完整VAL。精确K/T投影始终保留。下表正数表示关闭该项后误差增加，负数表示改善；加粗正值表示完整方案相应指标占优。', '']
    lines += historical_table(historical, '| 分区 | 关源几何：MAE变化 |')
    lines += ['关闭源融合几何后9/11分区MAE变差，支持标定导数与投影尺度参与证据融合的价值。固定尺度读取的变化较小且方向不一致，当前证据不足以把路由机制描述为主要精度来源。关闭Jacobian扩宽虽使8/11分区MAE改善，却使8/11分区RMSE以及7/9个有效远距分区MAE变差，说明搜索覆盖与族内连续定位存在取舍。弱视差条件下过宽候选族会放大评分扰动，未来设计应区分搜索方向、覆盖范围与读出不确定性。该结论解释失败并提出可检验方向，不代表已经完成新结构升级。', '',
              '**缺失视图、尺度与可靠性。** 历史压力实验发现，深度均值改善不等于缺图鲁棒性或置信度排序改善；单位换算一致也不等于物理尺度等变。标定误差会显著损害跨视图取证，范围细化则可能压缩可表达的远距深度。为避免将历史执行路径的数值混作快速版结果，本版保留这些机制结论，具体旧数值参见原始记录。', '',
              '**CNN桥与后续专项的取舍。** V5的桥接短预算同参考实验中，粗层近距MAE由0.76581降至0.36577m，最终近距由0.55901降至0.40521m，十二候选5%覆盖率由81.223%升至91.959%，但选择遗憾增加。该结果支持局部CNN信息补充ViT token具有候选供给潜力，而不是证明后端充分利用了改善。此后V6的完整训练包含额外热启动预算，其与V5的全部差异不能只归因于新增129参数交接门。', '',
              '11月2日ETH3D/CARLA专项共26次短训，最有希望的合成联合方案使CARLA上VAL MAE下降7.30%，固定TEST却增加3.13%；ETH粗层改善1.22%，最终MAE增加0.28%。删去新增先验仍保留主要VAL收益，因而没有将新模块加入正式V6。该负结果提示训练分布变化和后端定位均可能限制收益，不能凭VAL或中间层改善宣称最终泛化提升。V6权重和默认架构继续保留。', '',
              '**补充真实外域：ETH3D courtyard。** 既有单场景外测揭示了明显的真实域泛化缺口，增加输入视图未能消除失配。该项尚无快速版的独立精度与正式计时，故本版不列其普通执行数值，不与27对灰度双目主表合并；完整原始记录保留在此前报告中。', '',
              '## 论文结论与证据边界', '',
              '现有证据支持以“标定引导证据读取、候选族保留多峰、跨阶段交接保护正确粗面”为研究主线。源几何融合与共同预算交接实验提供了具体支持；尺度路由、自适应门以及搜索扩宽的收益具有条件性，应呈现其代价与失效区域。公共合成域的优势不能直接外推到真实灰度窄基线双目，当前ETH3D结果和CARLA跨参考差异应进入讨论，而不应以平均指标掩盖。', '',
              '当前测试集已多次用于研究分析，不能称为从未查看过的最终盲测。等预训练曝光、所有baseline共同全预算训练、具有实例标签的小障碍召回、完整占据与未知空间评价，以及实际车端含融合的端到端实时性仍未完成。这些内容不填数值，也不作为已验证贡献。', '',
              '**数据与图表。** [本次精度—延迟图](../runs/active/paper_retest_20261102/analysis/accuracy_latency.pdf)；[全部深度指标](../runs/active/paper_retest_20261102/analysis/depth_all_metrics.csv)；[全部表面指标](../runs/active/paper_retest_20261102/analysis/surface_all_metrics.csv)；[全部阶段指标](../runs/active/paper_retest_20261102/analysis/stages_all_metrics.csv)；[完整耗时与显存](../runs/active/paper_retest_20261102/analysis/latency_all_metrics.csv)。历史[三种子原始结果](../runs/active/rulermvs_v6_20261101/full_training/paper_handoff_ablation/summary.json)、[标尺消融](../runs/active/rulermvs_v6_20261101/full_training/paper_geometry_ablation/analysis/summary.json)、[物理与可靠性](../runs/active/rulermvs_v6_20261101/full_training/paper_physics/analysis/结果说明.md)保留原协议；正文历史表来自已完成报告，不是本轮重新训练。', '']
    DOC.write_text('\n'.join(lines))
    sources[str(ROOT / 'docs/v6_paper_results_20261101.md')] = digest(ROOT / 'docs/v6_paper_results_20261101.md')
    write_json(BASE / 'delivery.json', {
        'completed': True, 'new_accuracy_rows': len(depth_rows), 'new_surface_rows': len(surface_rows),
        'new_timing_rows': len(speed_rows), 'datasets': list(domains), 'document': str(DOC),
        'source_sha256': sources, 'document_sha256': digest(DOC),
        'new_training': False, 'historical_experiments_retrained': False,
        'unavailable': plan['unavailable'],
    })
    print(json.dumps({'completed': True, 'document': str(DOC)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
