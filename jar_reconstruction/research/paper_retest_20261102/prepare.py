"""冻结本轮重测配置；不改变既有权重、划分、指标或默认模型。"""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json

OUTPUT = ROOT / 'runs/active/paper_retest_20261102'
PRIOR = ROOT / 'runs/active/rulermvs_v6_20261101/full_training'


def prepare():
    if not OUTPUT.resolve().is_relative_to(Path('/data')):
        raise ValueError('结果必须通过既有软链接保存至数据盘')
    sources = {
        'public': PRIOR / 'evaluation_recovery/configs/public_main_test.json',
        'carla': PRIOR / 'evaluation_recovery/configs/carla_new_main_test.json',
        'eth3d': PRIOR / 'paper_eth3d/evaluation.json',
    }
    plan = {'groups': {}, 'sources': {}, 'weights': {}, 'manifests': {}, 'completed': False}
    for phase, filename in sources.items():
        config = json.loads(filename.read_text())
        keep = ('v6_best', 'mvsanywhere_refined', 'patchmatchnet_best', 'diffmvs_best')
        config['methods'] = {key: config['methods'][key] for key in keep}
        first = copy.deepcopy(config['methods']['mvsanywhere_refined'])
        first['refinements'] = 0
        config['methods']['mvsanywhere_firstpass'] = first
        fast = copy.deepcopy(config['methods']['v6_best'])
        fast['deployment_fast'] = True
        config['methods']['v6_fast'] = fast
        config['methods']['fast_mvsnet_dtu'] = {
            'kind': 'fast_mvsnet',
            'weights': str(ROOT / 'official_methods/methods/fast_mvsnet/upstream/outputs/pretrained.pth'),
            'drop_missing_views': True,
        }
        config.update(
            evaluation_split='test', gpus=list(range(8)), loader_workers=1,
            cpu_threads=2, shard_size=64,
            fast={'compile_source': True, 'compile_fusion': True,
                  'cache_dir': str(OUTPUT / 'compile_cache'), 'compile_threads': 1},
        )
        groups = {phase: config}
        if phase == 'public':
            groups = {}
            for views in (4, 2):
                item = copy.deepcopy(config)
                item['datasets'] = [d for d in item['datasets'] if d['num_views'] == views]
                groups[f'public{views}'] = item
        for name, item in groups.items():
            if all(d['num_views'] == 4 for d in item['datasets']):
                item['methods']['simplerecon_carla_old'] = {
                    'kind': 'simplerecon', 'num_views': 4,
                    'weights': str(ROOT / 'runs/carla_research_20261011/baselines/simplerecon/step3000.pt'),
                }
            item['output'] = str(OUTPUT / 'accuracy' / name)
            item['protocol_notes'] = [
                '本轮完整TEST重新推理，不复用历史预测；所有权重固定，不按TEST选择。',
                '256x448共同RGB、真实N=2/4、FP32/TF32关闭、同GT掩码及TRAIN范围。',
                '同时测V6普通/快速路径及可用baseline；快速版独立计算所有深度和表面指标。',
                'Fast-MVSNet为DTU发布权重；SimpleRecon为旧CARLA四目训练，非八目Hero迁移。',
                'Patchmatch/Diff保留已有新CARLA适配权重与逐帧随机流；不同训练预算需披露。',
                '双目不复制图像凑四目；不兼容的SimpleRecon四目权重不参加双目排名。',
            ]
            path = OUTPUT / 'configs' / f'{name}.json'
            write_json(path, item)
            frames = {d['name']: sum(bool(line.strip()) for line in
                                    Path(d['manifests']['test']).read_text().splitlines())
                      for d in item['datasets']}
            plan['groups'][name] = {'config': str(path), 'output': item['output'], 'frames': frames}
            for settings in item['methods'].values():
                path = settings['weights']
                plan['weights'][path] = digest(path)
            for domain in item['datasets']:
                path = domain['manifests']['test']
                plan['manifests'][path] = digest(path)
        plan['sources'][str(filename)] = digest(filename)
    plan['unavailable'] = {
        'MVSFormer++': '未获得任务权重，不将随机初始化放入精度排名。',
        'SimpleRecon N=2': '现有任务权重固定四图；不复制输入或改MLP冒充原模型。',
        'OctreeOcc/GaussianFormer-2': '当前无对应占据语义共同标签及完整可比协议，不混入深度主表。',
    }
    plan['timing'] = {
        'gpu': 0, 'test_samples_per_domain': 16, 'warmup': 10, 'repeats': 10,
        'scope': '固定TEST等间隔16组（不足则全部），无标签，GPU常驻；同卡交替，batch1。',
    }
    write_json(OUTPUT / 'plan.json', plan)
    print(json.dumps({k: sum(v['frames'].values()) for k, v in plan['groups'].items()}))


if __name__ == '__main__':
    prepare()
