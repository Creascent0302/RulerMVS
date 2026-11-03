"""冻结旧TEST清单、新TRAIN范围与四套权重；不根据测试结果改协议。"""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jar.pretraining.comparison import digest, write_json

OUTPUT = ROOT / 'runs/active/overnight_v6_v7_mvs_20261103/comparison'
TRAIN = OUTPUT.parent


def main():
    if (OUTPUT / 'config.json').exists():
        raise FileExistsError('不覆盖已经冻结的评测配置')
    if not OUTPUT.resolve().is_relative_to(Path('/data')):
        raise ValueError('结果必须位于数据盘')
    previous = ROOT / 'runs/active/paper_retest_20261102/configs'
    training = json.loads((TRAIN / 'v6/config.json').read_text())
    train_domains = {d['name']: d for d in training['datasets']}
    domains = []
    for group in ('public4', 'public2', 'carla', 'eth3d'):
        old = json.loads((previous / (group + '.json')).read_text())
        for source in old['datasets']:
            domain = copy.deepcopy(source)
            if domain['name'] in train_domains:
                new = train_domains[domain['name']]
                for key in ('near', 'far', 'depth_scale'):
                    domain[key] = new[key]
            domains.append(domain)
    # 与先前主表相同的行序；新模型均只使用公共权重，CARLA不额外微调。
    order = ['TartanAir_forest', 'MatrixCity_street', 'Hypersim_subset', 'MVS_Synth',
             'VKITTI2', 'BlendedMVS', 'DynamicReplica_valid']
    domains.sort(key=lambda d: order.index(d['name']) if d['name'] in order else 20)
    methods = {
        'v6': {'kind': 'own', 'retrained': True, 'weights': str(TRAIN / 'v6/best.pt'),
               'input_mode': 'equal_pixels'},
        'v7': {'kind': 'own', 'retrained': True, 'weights': str(TRAIN / 'v7/best.pt'),
               'input_mode': 'equal_pixels'},
        'mvs_retrained': {'kind': 'mvs_retrained', 'retrained': True,
                         'weights': str(TRAIN / 'mvsanywhere/best.pt'), 'refinements': 1},
        'mvs_hero': {'kind': 'mvsanywhere',
                    'weights': str(ROOT / 'runs/carla_research_20261011/cache/mvsanywhere_hero.ckpt'),
                    'refinements': 1, 'shared_train_range': True},
    }
    config = dict(image_size=[256, 448], matching_image_size=[512, 896],
                  datasets=domains, methods=methods, own_weights=methods['v6']['weights'],
                  output=str(OUTPUT / 'accuracy'), evaluation_split='test', gpus=list(range(8)),
                  cpu_threads=2, loader_workers=1, shard_size=64, seed=20261103, tf32=False,
                  position_cache=True, fast_cost_volume=True,
                  fast=dict(compile_source=True, compile_fusion=True, compile_threads=1,
                            cache_dir=str(OUTPUT / 'compile_cache')),
                  timing=dict(gpu=0, test_samples_per_domain=16, warmup=10, repeats=10),
                  protocol_notes=[
                      '原主表TEST清单与GT掩码不变，公共域范围和归一尺度取本次TRAIN；全方法共用。',
                      '本次V6/V7/MVS共同数据预算best；官方Hero单列、预训练曝光不同。',
                      '新V6/V7在CARLA使用公共权重，不与历史CARLA适配数字混用。',
                      '主表仅MAE/RMSE/完整深度输出中位ms，另存coverage和原始累计量。',
                      '图横轴sum(abs(error))/sum(valid_GT)，不跨单位平均原始MAE。',
                      '图只用CARLA上参考、ETH ALL避免重复展示；逐域判断Pareto，不连跨域曲线。'])
    config['extension_source_sha256'] = {str(p.relative_to(ROOT)): digest(p)
                                       for p in Path(__file__).parent.glob('*.py')}
    plan = dict(completed=False, weights_sha256={k: digest(v['weights']) for k, v in methods.items()},
                test_manifest_sha256={d['name']: digest(d['manifests']['test']) for d in domains},
                frame_counts={d['name']: sum(bool(x.strip()) for x in Path(d['manifests']['test']).read_text().splitlines())
                              for d in domains}, timing=config['timing'])
    write_json(OUTPUT / 'config.json', config)
    write_json(OUTPUT / 'plan.json', plan)
    print(json.dumps(plan['frame_counts']))


if __name__ == '__main__':
    main()
