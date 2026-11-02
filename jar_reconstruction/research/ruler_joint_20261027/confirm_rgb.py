"""固定 RGB 训练末步的 405 组历史 VAL 确认；不选检查点、不访问 TEST。"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from research.ruler_joint_20261027.model import make_decoder
from research.ruler_joint_20261027.run import compare, evaluate


RUN = ROOT / 'runs/active/ruler_research_20261027'
OUTPUT = RUN / 'confirmation_rgb_last'
NAMES = ('rgb_linear', 'rgb_relation', 'rgb_relation_joint', 'rgb_control')
INITIAL = ROOT / 'runs/active/public_refinement_20261025/recommended/pretrained.pt'
OLD_INDEX = ROOT / 'runs/active/public_refinement_20261025/features/index.json'
DEV_INDEX = ROOT / 'runs/active/architecture_screen_20261026/features/index.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare():
    """先固定模型、步数和已使用的 405 个确认样本，不读取本轮预测。"""
    previous = RUN / 'confirmation_round12'
    selection = read_json(previous / 'selection.json')
    old_index = read_json(OLD_INDEX)
    protocol = read_json(previous / 'protocol.json')
    assert digest(OLD_INDEX) == protocol['cache_index_sha256']
    assert digest(DEV_INDEX) == protocol['dev_index_sha256']
    index = {'domains': old_index['domains'], 'splits': {'confirmation': {}}}
    for domain in old_index['domains']:
        name = domain['name']
        available = {
            int(Path(path).stem): path
            for path in old_index['splits']['confirmation'][name]
        }
        fixed = selection[name]
        assert not fixed['shared_rgb']
        assert digest(domain['manifests']['val']) == fixed['manifest_sha256']
        paths = [available[number] for number in fixed['indices']]
        assert len(paths) == len(set(paths)) == fixed['count']
        assert all((ROOT / path).exists() for path in paths)
        index['splits']['confirmation'][name] = paths
    assert sum(len(paths) for paths in index['splits']['confirmation'].values()) == 405
    plan = {
        'models': ['original', *NAMES],
        'checkpoint_selection': '固定每组 last.pt，必须 step=1792；不查看 best.pt、不据确认调参',
        'step': 1792,
        'device': 'cuda:4',
        'precision': 'FP32 TF32 off',
        'training': False,
        'test_accessed': False,
        'historical_validation': True,
        'selection_note': '与 confirmation_round12 相同的 405 组历史 VAL；不是新盲测',
        'metrics': ['full/MAE', 'full/RMSE', 'full/AbsRel',
                    'far25/MAE', 'far25/RMSE', 'far25/AbsRel'],
        'comparison_targets': ['original', 'rgb_control'],
        'pixel_protocol': '完整合法 GT，不按候选范围删除；far25 取每帧完整合法 GT 的 75% 分位数',
        'selection_sha256': digest(previous / 'selection.json'),
        'cache_index_sha256': digest(OLD_INDEX),
        'dev_index_sha256': digest(DEV_INDEX),
        'previous_protocol_sha256': digest(previous / 'protocol.json'),
        'previous_original_sha256': digest(previous / 'original.json'),
        'script_sha256': digest(__file__),
        'frame_counts': {name: entry['count'] for name, entry in selection.items()},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'plan.json').exists():
        assert read_json(OUTPUT / 'plan.json') == plan, '已登记协议改变'
        assert read_json(OUTPUT / 'index.json') == index, '已登记样本改变'
    else:
        write_json(OUTPUT / 'plan.json', plan)
        write_json(OUTPUT / 'index.json', index)
        shutil.copy2(previous / 'selection.json', OUTPUT / 'selection.json')
    return plan, index, protocol


def verify_original(result, previous):
    differences = {}

    def walk(a, b, path):
        if isinstance(a, dict):
            assert set(a) == set(b), path
            for key in a:
                walk(a[key], b[key], path + '/' + key)
        elif isinstance(a, (float, int)):
            if isinstance(a, int):
                assert a == b, path
            else:
                assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6), (path, a, b)
                differences[path] = abs(a - b)
        else:
            assert a == b, path

    walk(result, previous, '')
    return {'numeric_fields': len(differences),
            'maximum_absolute_difference': max(differences.values()),
            'differences': differences}


def run():
    os.chdir(ROOT)
    plan, index, previous_protocol = prepare()
    missing = [name for name in NAMES if not (RUN / name / 'complete.json').exists()]
    if missing:
        raise RuntimeError('等待四组训练完成后再运行：' + ', '.join(missing))
    if (OUTPUT / 'original.json').exists():
        raise FileExistsError('结果目录已有预测，禁止重跑覆盖')

    torch.set_num_threads(2)
    torch.manual_seed(20261027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(plan['device'])
    torch.cuda.set_device(device)
    started = time.monotonic()
    dev_index = read_json(DEV_INDEX)
    initial_hash = digest(INITIAL)
    assert initial_hash == dev_index['checkpoint_sha256']
    assert initial_hash == previous_protocol['models']['original']['sha256']
    old_cache_source = Path(previous_protocol['old_cache_source_checkpoint'])
    assert digest(old_cache_source) == previous_protocol['old_cache_source_sha256']
    initial = torch.load(INITIAL, map_location='cpu', weights_only=False, mmap=True)
    records = {}
    models = {'original': None}
    schedules = set()
    source_paths = [Path(__file__).with_name('model.py'), Path(__file__).with_name('run.py')]
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    for name in NAMES:
        folder = RUN / name
        completion = read_json(folder / 'complete.json')
        assert completion['steps'] == 1792 and completion['exposures'] == 3584
        assert completion['full_pass_completed'] and not completion['smoke']
        assert completion['frozen_encoder_and_coarse_verified']
        assert not completion['test_accessed']
        audit = read_json(folder / 'audit.json')
        assert audit['checkpoint_sha256'] == initial_hash
        assert audit['encoder_sha256'] == dev_index['encoder_sha256']
        assert audit['cache_index_sha256'] == plan['dev_index_sha256']
        assert all(audit['sources'][path] == value for path, value in source_hashes.items())
        checkpoint = folder / 'last.pt'
        record = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert record['step'] == 1792
        assert record['encoder_sha256'] == dev_index['encoder_sha256']
        assert digest(record['encoder_checkpoint']) == initial_hash
        assert record['schedule_sha256'] == audit['schedule_sha256']
        schedules.add(record['schedule_sha256'])
        models[name] = record
        records[name] = {'path': str(checkpoint), 'sha256': digest(checkpoint),
                         'step': record['step'], 'plan': record['plan'],
                         'completion': completion, 'audit_sha256': digest(folder / 'audit.json')}
    assert len(schedules) == 1, '四组训练采样调度必须相同'
    records['original'] = {'path': str(INITIAL), 'sha256': initial_hash, 'step': 'original'}
    write_json(OUTPUT / 'checkpoint_audit.json', records)
    print('固定405样本、四组1792末步、编码器/调度摘要检查通过', flush=True)

    results = {}
    for name, record in models.items():
        decoder, _ = make_decoder(initial, record['plan'] if record else {'relation': False}, device)
        if record:
            decoder.load_state_dict(record['model'], strict=True)
        results[name] = evaluate(decoder, index, device, 'confirmation')
        write_json(OUTPUT / (name + '.json'), results[name])
        if name == 'original':
            parity = verify_original(results[name], read_json(RUN / 'confirmation_round12/original.json'))
            write_json(OUTPUT / 'original_parity.json', parity)
        print(name, '405组完成', flush=True)
        del decoder
    for result in results.values():
        for domain in plan['frame_counts']:
            for region in ('full', 'far25', 'near75'):
                assert result[domain][region]['count'] == results['original'][domain][region]['count']
    comparisons = {
        target: {name: compare(result, results[target]) for name, result in results.items() if name != target}
        for target in ('original', 'rgb_control')
    }
    write_json(OUTPUT / 'comparisons.json', comparisons)
    assert source_hashes == {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    assert all(digest(records[name]['path']) == records[name]['sha256'] for name in models)
    protocol = {
        **plan, 'models': records, 'source_sha256': source_hashes,
        'schedule_sha256': next(iter(schedules)), 'decoder_forwards': 2025,
        'original_matches_round12': True,
        'cache_equivalence': '复用本轮已完成的325个编码器张量及7域RGB/GT逐张量一致性验证；本次只校原始checkpoint/index/manifest摘要及重算original结果',
        'seconds': time.monotonic() - started, 'network_latency_measured': False,
        'decision': '固定末步诊断；不依据确认结果继续调参，不发布新默认',
    }
    write_json(OUTPUT / 'protocol.json', protocol)
    shutil.copy2(__file__, OUTPUT / Path(__file__).name)
    write_json(OUTPUT / 'complete.json', {'decoder_forwards': 2025,
                                         'seconds': protocol['seconds'], 'test_accessed': False})
    print(json.dumps({target: {name: {'risk': row['normalized_risk_change'], 'failures': row['failures']}
                              for name, row in value.items()}
                      for target, value in comparisons.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='只保存固定协议，不读取末步权重、不使用GPU')
    arguments = parser.parse_args()
    if arguments.prepare:
        os.chdir(ROOT)
        prepare()
        print('405组固定末步确认计划已保存；尚未执行模型前向', flush=True)
    else:
        run()
