"""V6及历史模型的薄评测注册；数据、指标和基线网络复用已有实现。"""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining import comparison
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.model import PretrainingModel
from research.baseline_full_comparison_20261101.evaluate import Models as AccuracyModels
from research.v5_comparison_20261101.evaluate import Models as NativeModels
from research.rulermvs_v5_20261031.model import V5Model
from research.rulermvs_v6_20261101.model import V6Model

OUTPUT = ROOT / 'runs/active/rulermvs_v6_20261101/full_validation'
PILOTS = ROOT / 'runs/active/rulermvs_v6_20261101/pilots'
REFERENCE = ROOT / 'runs/active/baseline_full_comparison_20261101'


def model_factory(config, initialize=True):
    architecture = config.get('architecture')
    if architecture == 'rulermvs_v6':
        return V6Model(config, initialize)
    if architecture == 'rulermvs_v5':
        return V5Model(config, initialize)
    if architecture == 'ruler_flow_v2':
        from research.ruler_flow_v2_20261031.model import EvidenceFlowModel

        return EvidenceFlowModel(config, initialize)
    return PretrainingModel(config, initialize)


# 只在显式运行此入口及其spawn工作进程时生效。
comparison.PretrainingModel = model_factory


class Models(AccuracyModels):
    """PatchmatchNet只读取存在的相机；计时不含逐帧CPU数据哈希。"""

    def __init__(self, config, device, fixed_random_stream=True):
        super().__init__(config, device)
        self.fixed_random_stream = fixed_random_stream

    def forward(self, method, domain, inputs):
        reference = domain['reference_index']
        kind = self.config['methods'][method]['kind']
        if kind != 'own' and (not bool(inputs['view_mask'][0, reference])
                              or int(inputs['view_mask'][0].sum()) < 2):
            # 不支持单目的原生MVS保留GT分母，以零覆盖报告，不补图或删除样本。
            depth = inputs['rgb'].new_zeros((1, 1, *inputs['rgb'].shape[-2:]))
            return {'depth': depth, 'valid': depth.bool(), 'reference_indices': [reference],
                    'unsupported_reason': '原生多视图方法需要有效参考和至少一个真实源相机'}
        compacted = kind == 'patchmatchnet' and not bool(inputs['view_mask'].all())
        if compacted:
            inputs, local_reference = comparison.compact_valid_views(inputs, reference)
            domain = dict(domain, num_views=inputs['rgb'].shape[1], reference_index=local_reference)
        forward = AccuracyModels.forward if self.fixed_random_stream else NativeModels.forward
        prediction = forward(self, method, domain, inputs)
        if compacted:
            prediction = dict(prediction, reference_indices=[reference])
        return prediction


# spawn会以__mp_main__重新导入入口，注册必须在模块层执行。
# 否则worker仍使用baseline入口先注册的类，丢失缺相机压紧逻辑。
from research.near_mid_evaluation_20261028 import evaluate

evaluate.Models = Models


def prepare():
    sources = [
        Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).with_name('handoff.py'),
        ROOT / 'research/rulermvs_v5_20261031/model.py',
        ROOT / 'research/rulermvs_v5_20261031/bridge.py',
        ROOT / 'jar/models/ruler_proposals.py',
    ]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    plan = {'completed': False, 'split': 'val', 'test_accessed': False, 'groups': {},
            'selection': '开发分区预先选择parent均值版本；不依据完整VAL重新选择其他读出',
            'extension_source_sha256': hashes}
    for phase, source in [('public', 'public_carla_transfer'), ('carla', 'carla_four_references')]:
        audit = json.loads((REFERENCE / source / 'full_test/audit.json').read_text())
        config = copy.deepcopy(audit['config'])
        frozen = config['own_weights']
        config['methods'] = {
            name: {'kind': 'own', 'weights': str(weights), 'input_mode': 'equal_pixels',
                   'inference': {'spatial_layout': True}}
            for name, weights in (
                ('v5_frozen', frozen),
                ('v5_control', PILOTS / (phase + '_v5') / 'final.pt'),
                ('v6_parent', PILOTS / (phase + '_parent') / 'final.pt'),
            )
        }
        config.update(output=str(OUTPUT / phase), evaluation_split='val',
                      own_weights=config['methods']['v6_parent']['weights'], gpus=list(range(8)),
                      cpu_threads=2, loader_workers=1, shard_size=128 if phase == 'public' else 32,
                      extension_source_sha256=hashes)
        config.pop('split', None)
        config['protocol_notes'] = [
            '固定正式V5、同预算V5续训、开发分区选出的V6 parent均值三个checkpoint。',
            '全部VAL、相同输入像素/GT/范围/表面阈值、FP32/TF32off；不访问TEST。',
            'V6只改候选族交接/读出与对应监督，编码器及粗层结构保持。',
            '公共与CARLA使用各自独立权重，不宣称同一checkpoint跨域泛化。',
            'scene_unit不称米；双目不复制为四目；本阶段不计时。',
        ]
        for domain in config['datasets']:
            domain['manifests'] = {key: value for key, value in domain['manifests'].items()
                                   if key in ('train', 'val')}
        frames = {domain['name']: sum(bool(line.strip()) for line in
                                     Path(domain['manifests']['val']).read_text().splitlines())
                  for domain in config['datasets']}
        filename = OUTPUT / 'configs' / (phase + '.json')
        write_json(filename, config)
        plan['groups'][phase] = {'config': str(filename), 'frames': frames,
                                 'output': config['output'],
                                 'weights_sha256': {name: digest(Path(settings['weights']))
                                                    for name, settings in config['methods'].items()}}
    write_json(OUTPUT / 'plan.json', plan)
    print(json.dumps({name: sum(group['frames'].values()) for name, group in plan['groups'].items()}))


if __name__ == '__main__':
    if sys.argv[1:] == ['--prepare']:
        prepare()
    else:
        if '--speed' in sys.argv:
            from functools import partial
            from tools import benchmark_public_speed

            sys.argv.remove('--speed')
            # 网络原生随机采样保留；seed由公共计时入口设置，输入哈希不计入网络耗时。
            benchmark_public_speed.Models = partial(Models, fixed_random_stream=False)
            benchmark_public_speed.main()
        else:
            evaluate.main()
