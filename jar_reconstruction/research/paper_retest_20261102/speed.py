"""逐数据域同卡测速：固定 TEST 无标签输入，完整深度输出，随机交替。"""

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
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.paper_retest_20261102.evaluate import Models
from research.rulermvs_v6_20261101.speed_final import distribution
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import fingerprint, offline_only


def read(path):
    return json.loads(Path(path).read_text())


def clear(models):
    models.loaded.clear()
    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()


def call(models, method, domain, inputs, four=False):
    if not four:
        return models.forward(method, domain, inputs)
    settings = models.config['methods'][method]
    if settings['kind'] == 'own':
        outputs = models.get(method, domain).forward_references(
            equal_pixel_inputs(inputs), domain, range(4))
        return {ref: {'depth': out['depth'] * out['unit_scale'], 'valid': out['valid'],
                      'reference_indices': [ref]} for ref, out in outputs.items()}
    return {ref: models.forward(method, dict(domain, reference_index=ref), inputs)
            for ref in range(4)}


def convert(models, result, inputs, four):
    if four:
        return {ref: models.convert(value, inputs) for ref, value in result.items()}
    return models.convert(result, inputs)


def resources(config, domain, cpu, device, four):
    """各方法独立驻留，显存不包含其他模型，也不以共驻峰值排名。"""
    records = {}
    for method in config['methods']:
        inputs = move(cpu, device)
        models = Models(config, device, fixed_random_stream=False)
        models.get(method, domain)
        with torch.inference_mode():
            for _ in range(3):
                result = call(models, method, domain, inputs, four)
                converted = convert(models, result, inputs, four)
                del result, converted
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            result = call(models, method, domain, inputs, four)
            converted = convert(models, result, inputs, four)
            torch.cuda.synchronize(device)
            records[method] = {
                'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
                'parameters': sum(p.numel() for m in models.loaded.values() for p in m.parameters()),
            }
            del result, converted
        del inputs
        clear(models)
        del models
    return records


def benchmark(config, domain, settings, output, four=False):
    device = configure(config, settings['gpu'])
    dataset = build_dataset(dataset_config(domain, config), 'test', load_targets=False)
    indices = np.linspace(0, len(dataset) - 1, min(len(dataset), settings['test_samples_per_domain']))
    indices = sorted(set(indices.round().astype(int).tolist()))
    first = collate_frames([dataset[indices[0]]])['inputs']
    memory = resources(config, domain, first, device, four)
    models = Models(config, device, fixed_random_stream=False)
    for name in config['methods']:
        models.get(name, domain)
    rng = random.Random(20261102)
    samples = {name: {'network_ms': [], 'depth_output_ms': [], 'pipeline_ms': []}
               for name in config['methods']}
    hashes, warmup_seconds, shared_checks = {}, {}, {}
    with torch.inference_mode():
        for position, index in enumerate(indices):
            cpu = collate_frames([dataset[index]])['inputs']
            hashes[str(index)] = fingerprint(cpu)
            inputs = move(cpu, device)
            for name in config['methods']:
                started = time.perf_counter()
                for _ in range(settings['warmup'] if position == 0 else 1):
                    result = call(models, name, domain, inputs, four)
                    converted = convert(models, result, inputs, four)
                    del result, converted
                torch.cuda.synchronize(device)
                warmup_seconds[f'{index}/{name}'] = time.perf_counter() - started
            if four and position == 0:
                for name, method in config['methods'].items():
                    if method['kind'] != 'own':
                        continue
                    shared = call(models, name, domain, inputs, True)
                    exact = True
                    for ref in range(4):
                        serial = models.forward(name, dict(domain, reference_index=ref), inputs)
                        exact &= all(torch.equal(serial[key], shared[ref][key]) for key in ('depth', 'valid'))
                        del serial
                    if not exact:
                        raise RuntimeError('共享四参考与串行输出不一致：' + name)
                    shared_checks[name] = {'index': index, 'depth_valid_exact': True}
                    del shared
            for _ in range(settings['repeats']):
                order = list(config['methods'])
                rng.shuffle(order)
                for name in order:
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    result = call(models, name, domain, inputs, four)
                    torch.cuda.synchronize(device)
                    network_done = time.perf_counter()
                    converted = convert(models, result, inputs, four)
                    torch.cuda.synchronize(device)
                    done = time.perf_counter()
                    samples[name]['network_ms'].append((network_done - start) * 1000)
                    samples[name]['depth_output_ms'].append((done - start) * 1000)
                    del result, converted
            del inputs, cpu
            # 独立计量单进程、热文件缓存的部署数据管线，不混入GT或指标计算。
            order = list(config['methods'])
            rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                cpu = collate_frames([dataset[index]])['inputs']
                inputs = move(cpu, device)
                result = call(models, name, domain, inputs, four)
                converted = convert(models, result, inputs, four)
                returned = move(converted, 'cpu')
                torch.cuda.synchronize(device)
                samples[name]['pipeline_ms'].append((time.perf_counter() - start) * 1000)
                del returned, converted, result, inputs, cpu
            print(json.dumps({'timing': domain['name'], 'four': four, 'frame': index}), flush=True)
    result = {
        'completed': True, 'domain': domain, 'four_references': four,
        'indices': indices, 'input_sha256': hashes, 'samples_ms': samples,
        'summary': {name: {key: distribution(value) for key, value in fields.items()}
                    for name, fields in samples.items()},
        'isolated_resources': memory, 'warmup_seconds': warmup_seconds,
        'shared_vs_serial': shared_checks,
        'weights_sha256': {name: digest(item['weights']) for name, item in config['methods'].items()},
        'test_manifest_sha256': digest(domain['manifests']['test']),
        'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
        'precision': 'FP32/TF32off', 'load_targets': False,
        'scope': 'batch1；network含输入适配，depth_output加共同格点转换；GPU常驻；不含首次编译。',
        'pipeline_scope': '热文件缓存，单进程读取/图像解码/输入准备/H2D/网络/输出转换/D2H；不含GT、指标、点云融合。',
        'memory_scope': '单模型驻留的输入+参数+激活allocated峰值；非nvidia-smi驱动显存。',
        'stochastic': '保留原生随机候选；计时去掉评测专用逐帧CPU哈希，精度评测仍固定逐帧随机流。',
    }
    write_json(output, result)
    clear(models)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--domain', help='用于逐域独立进程，避免跨域编译缓存干扰')
    parser.add_argument('--four', action='store_true')
    args = parser.parse_args()
    offline_only()
    config = read(args.config)
    base = ROOT / 'runs/active/paper_retest_20261102'
    settings = read(base / 'plan.json')['timing']
    if args.domain:
        domain = next(d for d in config['datasets'] if d['name'] == args.domain)
        suffix = '_four' if args.four else ''
        target = base / 'speed' / (domain['name'] + suffix + '.json')
        if target.exists() and read(target).get('completed'):
            return
        benchmark(config, domain, settings, target, args.four)
        return
    active = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('其他GPU任务尚未退出，不能开始独占计时：' + active)
    for domain in config['datasets']:
        command = [sys.executable, __file__, '--config', args.config, '--domain', domain['name']]
        subprocess.run(command, check=True, cwd=ROOT)
        if domain['name'] == 'CARLA_new_upper':
            subprocess.run(command + ['--four'], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
