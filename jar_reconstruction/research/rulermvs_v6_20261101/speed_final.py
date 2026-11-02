"""冻结V5/V6最终候选：同卡交替单参考/真实四参考计时，另进程独占显存。"""

import argparse
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.pretraining.inference import prepare_inference
from jar.runtime import move
from research.rulermvs_v6_20261101.inference import prepare_fast_inference
from research.rulermvs_v6_20261101.model import build_from_checkpoint
from research.rulermvs_v6_20261101.speed_options import differences
from tools.smoke_outdoor_datasets import fingerprint, offline_only


METHODS = ('v5_eager', 'v5_fast', 'v6_eager', 'v6_fast')


def initialize(settings):
    offline_only()
    torch.set_num_threads(2)
    torch.cuda.set_device(settings.get('gpu', 0))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load(settings, name):
    version = name.split('_')[0]
    weights = Path(settings[version + '_checkpoint'])
    saved = torch.load(weights, map_location='cpu', weights_only=False, mmap=True)
    if saved['architecture'] != 'rulermvs_' + version:
        raise ValueError(f'{name}必须加载对应版本完整权重，实际为{saved["architecture"]}')
    with torch.inference_mode(False), torch.no_grad():
        model, _ = build_from_checkpoint(saved, 'v5' if version == 'v5' else 'parent')
        model.requires_grad_(False).to('cuda:' + str(settings.get('gpu', 0))).eval()
        install_position_cache(model.foundation.model)
        if name.endswith('_fast'):
            prepare_fast_inference(model, settings.get('fast', {}))
        else:
            prepare_inference(model, spatial_layout=True, projective_ruler=False)
    return model


def make_calls(model, inputs, domain, four):
    calls = {'single': lambda: model(equal_pixel_inputs(inputs), domain, domain['reference_index'])}
    if four:
        calls['four'] = lambda: model.forward_references(equal_pixel_inputs(inputs), domain, range(4))
    return calls


def distribution(samples):
    median = float(np.median(samples))
    return {'median_ms': median, 'mean_ms': float(np.mean(samples)),
            'p95_ms': float(np.percentile(samples, 95)), 'hz': 1000 / median, 'calls': len(samples)}


def data(settings):
    config = json.loads(Path(settings['data_config']).read_text())
    domain = next(d for d in config['datasets'] if d['name'] == settings.get('dataset', 'CARLA_new_upper'))
    if domain['num_views'] != 4:
        raise ValueError('这套正式测速协议使用真实完整四目')
    dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=False)
    return config, domain, dataset


def memory_worker(settings, name):
    initialize(settings)
    _, domain, dataset = data(settings)
    index = len(dataset) // 2
    inputs = move(collate_frames([dataset[index]])['inputs'], 'cuda:' + str(settings.get('gpu', 0)))
    model = load(settings, name)
    result = {'method': name, 'single_model_process': True, 'load_targets': False, 'index': index}
    with torch.inference_mode():
        for scope, call in make_calls(model, inputs, domain, settings.get('four_references', True)).items():
            for _ in range(10):
                call()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            prediction = call()
            torch.cuda.synchronize()
            result[scope] = {'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
                             'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30}
            del prediction
    write_json(Path(settings['output']) / ('memory_' + name + '.json'), result)


def run(settings, config_path):
    initialize(settings)
    output = Path(settings['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'正式测速拒绝覆盖已有结果：{output}')
    output.mkdir(parents=True, exist_ok=True)
    config, domain, dataset = data(settings)
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    models = {name: load(settings, name) for name in METHODS}
    durations, differences_by_frame, equality, inputs_sha, warmup_seconds = {}, [], [], {}, {}
    rng = random.Random(20261101)
    with torch.inference_mode():
        for index in indices:
            cpu = collate_frames([dataset[index]])['inputs']
            if not bool(cpu['view_mask'].all()):
                raise RuntimeError('预登记测速帧不完整；不得悄悄换帧')
            inputs_sha[str(index)] = fingerprint(cpu)
            inputs = move(cpu, 'cuda:' + str(settings.get('gpu', 0)))
            calls = {name + '_' + scope: call for name, model in models.items()
                     for scope, call in make_calls(model, inputs, domain, settings.get('four_references', True)).items()}
            for name, call in calls.items():
                start = time.perf_counter()
                for _ in range(settings.get('warmup', 10)):
                    call()
                torch.cuda.synchronize()
                warmup_seconds[f'{index}/{name}'] = time.perf_counter() - start
            for _ in range(settings.get('repeats', 40)):
                order = list(calls)
                rng.shuffle(order)
                for name in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    prediction = calls[name]()
                    torch.cuda.synchronize()
                    durations.setdefault(name, []).append((time.perf_counter() - start) * 1000)
                    del prediction
            for version in ('v5', 'v6'):
                eager, fast = calls[version + '_eager_single'](), calls[version + '_fast_single']()
                differences_by_frame.append({'index': index, 'version': version, **differences(fast, eager)})
                del eager, fast
            if settings.get('four_references', True):
                for name, model in models.items():
                    shared = calls[name + '_four']()
                    row = {'index': index, 'method': name, 'depth_exact': True,
                           'valid_exact': True, 'max_depth_difference': 0.}
                    for reference in range(4):
                        serial = model(equal_pixel_inputs(inputs), domain, reference)
                        row['depth_exact'] &= torch.equal(shared[reference]['depth'], serial['depth'])
                        row['valid_exact'] &= torch.equal(shared[reference]['valid'], serial['valid'])
                        row['max_depth_difference'] = max(row['max_depth_difference'],
                            float((shared[reference]['depth'] - serial['depth']).abs().max()))
                        del serial
                    equality.append(row)
                    del shared
            print(json.dumps({'frame_done': index, 'methods': len(calls)}), flush=True)
            del inputs, cpu, calls
    report = {
        'completed': False, 'settings': settings, 'data_config': config,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'precision': 'FP32/TF32off',
        'split': 'val', 'load_targets': False, 'indices': indices, 'input_sha256': inputs_sha,
        'methods': {name: distribution(values) for name, values in durations.items()},
        'samples_ms': durations, 'fast_vs_eager': differences_by_frame, 'shared_vs_serial': equality,
        'warmup_including_compile_seconds': warmup_seconds,
        'checkpoint_sha256': {v: digest(settings[v + '_checkpoint']) for v in ('v5', 'v6')},
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), ROOT / 'research/rulermvs_v6_20261101/inference.py',
            ROOT / 'research/rulermvs_v6_20261101/model.py', ROOT / 'research/rulermvs_v6_20261101/handoff.py')},
        'scope': '含输入适配、完整网络或真实共享四参考；不含IO/H2D/点云融合；四参考不是单次乘四。',
        'limits': '编译改变浮点归约，三帧一致性不代替全VAL精度；稳态速度不含首次编译。',
    }
    write_json(output / 'results.json', report)
    # 用新进程测各模型allocated峰值，避免共驻四模型或编译缓存抬高显存结果。
    model = None
    call = None
    models.clear()
    del models
    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    for name in METHODS:
        subprocess.run([sys.executable, str(Path(__file__)), '--config', str(config_path),
                        '--memory-worker', name], check=True)
    report['memory'] = {name: json.loads((output / ('memory_' + name + '.json')).read_text()) for name in METHODS}
    report['completed'] = True
    write_json(output / 'results.json', report)
    print(json.dumps({'completed': True, 'methods': report['methods']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--memory-worker', choices=METHODS)
    arguments = parser.parse_args()
    path = Path(arguments.config).resolve()
    settings = json.loads(path.read_text())
    if arguments.memory_worker:
        memory_worker(settings, arguments.memory_worker)
    else:
        run(settings, path)
