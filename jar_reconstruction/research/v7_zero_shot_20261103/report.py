"""按事前定义的场景分组报告零样本误差；配对区间按场景重采样。"""

import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import numpy as np

from jar.pretraining.comparison import add_statistics
from research.near_mid_evaluation_20261028.metrics import depth_metrics
from research.v7_zero_shot_20261103.prepare import DATA, SCENES, INDOOR, write, sha
from research.v7_zero_shot_20261103.run import TANKS, FRESH, domain

METHODS = ['v6', 'v7', 'mvs_retrained', 'mvs_hero']
METRICS = ['MAE', 'RMSE', 'AbsRel', 'delta_1', 'coverage']
GROUPS = {
    'ETH3D_DSLR_13场景': SCENES,
    'ETH3D_新增拍摄12场景': [s for s in SCENES if s != 'courtyard'],
    'ETH3D_此前未分析6场景组': sorted(FRESH),
    'ETH3D_室内6场景': sorted(INDOOR),
    'ETH3D_室外7场景': [s for s in SCENES if s not in INDOOR],
    'TanksAndTemples_4场景': TANKS,
}


def summarize_scenes(records, scenes, views):
    result = {}
    for method in METHODS:
        values = [records[f'{s}_{views}views']['metrics'][method]['full'] for s in scenes]
        pooled = {}
        for scene in scenes:
            add_statistics(pooled, records[f'{scene}_{views}views']['sums'][method]['full'])
        result[method] = dict(scene_macro={key: float(np.mean([x[key] for x in values])) for key in METRICS},
                              pixel_pooled=depth_metrics(pooled))
    rng = np.random.default_rng(20261103)
    # relief/relief_2可能属于相关采集区域，保守合并重采样，保留原场景等权点估计。
    clusters = {}
    for scene in scenes:
        clusters.setdefault('relief_family' if scene.startswith('relief') else scene, []).append(scene)
    keys = list(clusters)
    draws = rng.integers(0, len(keys), (20000, len(keys)))
    intervals = {}
    for metric in ['MAE', 'RMSE', 'AbsRel']:
        sums, sizes = [], []
        for key in keys:
            diffs = [records[f'{s}_{views}views']['metrics']['v7']['full'][metric] -
                     records[f'{s}_{views}views']['metrics']['v6']['full'][metric] for s in clusters[key]]
            sums.append(sum(diffs))
            sizes.append(len(diffs))
        bootstrap = np.asarray(sums)[draws].sum(1) / np.asarray(sizes)[draws].sum(1)
        intervals[metric] = dict(v7_minus_v6=float(sum(sums) / sum(sizes)),
            ci95=np.quantile(bootstrap, [.025, .975]).tolist(), clusters=len(keys),
            interpretation='负值有利V7；区间只覆盖场景抽样，不覆盖训练随机种子')
    return dict(scenes=scenes, methods=result, paired_bootstrap=intervals)


def supplemental_previous():
    """复用251同权重TEST的场景留出部分；不冒称重新运行或新数据域。"""
    previous = PROJECT / 'runs/active/overnight_v6_v7_mvs_20261103/comparison'
    current_weights = json.loads((DATA / 'results/plan.json').read_text())['weights_sha256']
    old_plan = json.loads((previous / 'plan.json').read_text())
    assert old_plan['weights_sha256'] == current_weights
    config = json.loads((previous / 'config.json').read_text())
    registry = json.loads(Path('/home/liuzonghao/3Dscene/data/public_expansion_20261102/heldout_scene_registry.json').read_text())
    output = {}
    for heldout in registry['datasets']:
        name = heldout['name']
        reference = next(d for d in config['datasets'] if d['name'] == name)
        rows = [json.loads(line) for line in Path(reference['manifests']['test']).read_text().splitlines()]
        selected = [json.loads(line) for line in Path(heldout['manifests']['test']).read_text().splitlines()]
        key = lambda row: (row['sequence'], str(row['frame_id']))
        wanted = {key(row) for row in selected}
        indices = {i for i, row in enumerate(rows) if key(row) in wanted}
        assert len(indices) == len(selected) == len(wanted)
        records = [f for p in (previous / 'accuracy/full_test' / name).glob('part_*.json')
                   for f in json.loads(p.read_text())['frames'] if f['index'] in indices]
        assert len(records) == len(indices)
        sums = {}
        for frame in records:
            for method, value in frame['methods'].items():
                add_statistics(sums.setdefault(method, {}), value)
        output[name] = dict(groups=len(records), units=reference['geometry_units'],
            methods={m: depth_metrics(v) for m, v in sums.items()}, source=str(previous),
            scope='同训练数据家族内场景留出；重用251结果，不计入新域零样本主表')
    return output


def main():
    records = {}
    for views in [2, 4, 8]:
        for scene in SCENES + TANKS:
            name = f'{scene}_{views}views'
            source = DATA / 'results/accuracy/full_test' / name / 'part_0000.json'
            item = json.loads(source.read_text())
            assert item['completed']
            records[name] = dict(domain=domain(scene, views), frames=len(item['frames']), sums=item['totals'],
                metrics={m: {r: depth_metrics(s) for r, s in values.items()} for m, values in item['totals'].items()})
    grouped = {f'{name}/{views}views': summarize_scenes(records, scenes, views)
               for views in [2, 4, 8] for name, scenes in GROUPS.items()}
    speed = {str(v): json.loads((DATA / 'results/speed' / f'{v}views.json').read_text()) for v in [2, 4, 8]}
    output = DATA / 'results/analysis'
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'results.json', dict(completed=True, records=records, groups=grouped))
    write(output / 'supplemental_heldout_scenes.json', supplemental_previous())
    with (output / 'per_scene.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scene', 'dataset', 'views', 'method', 'region', *METRICS])
        writer.writeheader()
        for record in records.values():
            d = record['domain']
            for method, regions in record['metrics'].items():
                for region, metric in regions.items():
                    writer.writerow(dict(scene=d['scene'], dataset=d['dataset'], views=d['num_views'],
                                         method=method, region=region, **{k: metric[k] for k in METRICS}))
    lines = ['# V7零样本泛化测试', '',
        '固定2026-11-03共同预算重训公共best；没有外测微调或尺度对齐。全部输入256×448，FP32/TF32off。',
        '主表是四视图；每格为场景等权平均MAE / RMSE（米）/ AbsRel，均越低越好。RMSE为各场景RMSE的平均。', '',
        '| 测试组 | V6 | V7 | MVSAnywhere重训·细化 | MVSAnywhere官方·细化 |', '|---|---|---|---|---|']
    for name in GROUPS:
        group = grouped[f'{name}/4views']
        cells = [' / '.join(f'{group["methods"][m]["scene_macro"][k]:.4f}' for k in ['MAE', 'RMSE', 'AbsRel']) for m in METHODS]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 相机数量变化', '', '| 数据集与视图数 | V6 AbsRel | V7 AbsRel | 重训MVS AbsRel | 官方MVS AbsRel |', '|---|---|---|---|---|']
    for name in ['ETH3D_DSLR_13场景', 'TanksAndTemples_4场景']:
        for views in [2, 4, 8]:
            group = grouped[f'{name}/{views}views']
            lines.append('| ' + f'{name}/{views}' + ' | ' + ' | '.join(f'{group["methods"][m]["scene_macro"]["AbsRel"]:.4f}' for m in METHODS) + ' |')
    lines += ['', '## 场景差异的配对区间', '', 'V7减V6，负值有利V7；场景簇bootstrap 20000次，relief两组共同抽样。不是多训练seed区间。', '']
    for name in ['ETH3D_DSLR_13场景', 'ETH3D_此前未分析6场景组', 'TanksAndTemples_4场景']:
        entry = grouped[f'{name}/4views']['paired_bootstrap']['AbsRel']
        lines.append(f'- {name}：AbsRel差 {entry["v7_minus_v6"]:.5f}，95%区间 {entry["ci95"]}。')
    lines += ['', '## 速度', '', '同一5090、17个场景各一组、随机交替10轮、预热10轮；batch1完整深度输出，排除IO/编译。', '',
              '| 输入视图数 | V6 ms | V7 ms | 重训MVS ms | 官方MVS ms |', '|---|---|---|---|---|']
    for views in [2, 4, 8]:
        lines.append('| ' + str(views) + ' | ' + ' | '.join(f'{speed[str(views)]["summary"][m]["depth_output_ms"]["median_ms"]:.2f}' for m in METHODS) + ' |')
    lines += ['', '## 解释边界', '',
        '- 任务训练64198组中未包含ETH3D/Tanks and Temples；通用预训练资产曝光无法完整认证。',
        '- ETH3D本次为全部13个官方场景104组的共同输入协议；其中courtyard历史访问，另外六组与历史双目场景同名，另六组单列。',
        '- Tanks and Temples为官方四场景69组；公开称为其training split，但本项目仅用于外测，未进入任务训练。',
        '- 使用官方配组及标定，固定前2/4/8图和0.1–100米输入范围；不用官方可能源自GT的depth_range。',
        '- 原尺寸全部有限正GT计分，不过滤范围外GT；coverage单独保留，不能删除失败预测美化结果。',
        '- 不等同原论文分辨率、原生可变源图设置或完整五域RMVD排行榜；官方Hero训练预算不同。',
        '- 既有CARLA和ETH双目结果见251；同家族留出1667组另存，不与本轮新数据域混合。', '',
        '来源：https://github.com/lmb-freiburg/robustmvd/tree/master/rmvd/data ，https://www.eth3d.net/datasets 。']
    (output / '零样本泛化结果.md').write_text('\n'.join(lines) + '\n')
    write(DATA / 'results/delivery.json', dict(completed=True, predictions=sum(r['frames'] for r in records.values()) * 4,
                                             report=str(output / '零样本泛化结果.md')))


if __name__ == '__main__':
    main()
