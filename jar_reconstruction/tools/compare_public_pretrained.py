"""同协议评测公共预训练模型与本地baseline；八卡分片，完成后单卡交替计时。"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.data import collate_frames
from jar.runtime import move, seed_all
from jar.pretraining.comparison import (
    Models, add_statistics, digest, measure, metrics, test_dataset, write_json,
)
from tools.smoke_outdoor_datasets import fingerprint, offline_only


def configure(config, gpu):
    torch.set_num_threads(config['cpu_threads'])
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = config['tf32']
    torch.backends.cudnn.allow_tf32 = config['tf32']
    torch.backends.cudnn.benchmark = False
    seed_all(config['seed'])
    return torch.device('cuda', gpu)


def execute_task(config, output, task, models, datasets, device, protocol_hash, smoke):
    domain_index, method, part, indices = task
    domain = config['datasets'][domain_index]
    destination = output / domain['name'] / method / f'part_{part:04d}.json'
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous.get('completed') and previous.get('protocol_hash') == protocol_hash:
            return {'event': 'cached', 'dataset': domain['name'], 'method': method, 'part': part}
    if domain_index not in datasets:
        datasets[domain_index] = test_dataset(config, domain)
    dataset = datasets[domain_index]
    models.get(method, domain)
    loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate_frames,
                        num_workers=0 if smoke else config['loader_workers'], pin_memory=True)
    sums, records = {}, []
    begin = time.monotonic()
    for index, cpu_batch in zip(indices, loader):
        batch = move(cpu_batch, device)
        inputs = batch['inputs']
        with torch.inference_mode():
            prediction = models.forward(method, domain, inputs)
            depth, valid = models.convert(prediction, inputs)
            item = measure(depth, valid, batch, domain['reference_index'], domain.get('eval_range'))
        add_statistics(sums, item)
        record = {'index': index, 'statistics': item,
                  'valid_views': int(inputs['view_mask'][0].sum())}
        if domain.get('distance_bands'):
            record['bands'] = {}
            for near, far in domain['distance_bands']:
                upper = far if far == domain['eval_range'][1] else math.nextafter(far, -math.inf)
                record['bands'][f'{near:g}-{far:g}'] = measure(
                    depth, valid, batch, domain['reference_index'], [near, upper])
        # 只记录首中尾输入校验，避免每帧回传大图像作无意义哈希。
        if index in (0, len(dataset) // 2, len(dataset) - 1):
            record['input_sha256'] = fingerprint(cpu_batch['inputs'])
        records.append(record)
        del prediction, depth, valid, inputs, batch, cpu_batch
    report = {'completed': True, 'protocol_hash': protocol_hash, 'method': method,
              'dataset': domain['name'], 'part': part, 'geometry_units': domain['geometry_units'],
              'reference_index': domain['reference_index'], 'num_views': domain['num_views'],
              'statistics': sums, 'metrics': metrics(sums, domain['geometry_units']),
              'frames': records, 'wall_seconds': time.monotonic() - begin}
    write_json(destination, report)
    return {'event': 'part_complete', 'dataset': domain['name'], 'method': method,
            'part': part, 'frames': len(indices), 'seconds': report['wall_seconds']}


def worker(config, output, gpu, queue, results, protocol_hash, smoke):
    offline_only()
    device = configure(config, gpu)
    models, datasets = Models(config, device), {}
    while True:
        task = queue.get()
        if task is None:
            break
        try:
            result = execute_task(config, Path(output), task, models, datasets,
                                  device, protocol_hash, smoke)
        except Exception:
            result = {'event': 'failed', 'dataset': config['datasets'][task[0]]['name'],
                      'method': task[1], 'part': task[2], 'error': traceback.format_exc()}
            write_json(Path(output) / 'failures' / f'{result["dataset"]}_{task[1]}_{task[2]}.json', result)
        results.put(dict(result, gpu=gpu))


def summarize(config, output, tasks, blocked, protocol_hash):
    result = {'completed': True, 'protocol_hash': protocol_hash, 'datasets': {},
              'unsupported': blocked, 'not_applicable': config['not_applicable'],
              'protocol': config['protocol_notes'], 'config': config}
    for domain in config['datasets']:
        methods = {}
        expected_gt = None
        input_hashes = {}
        for method in config['methods']:
            selected = [t for t in tasks if config['datasets'][t[0]]['name'] == domain['name'] and t[1] == method]
            if not selected:
                continue
            total, indices, hashes, bands, by_views = {}, [], {}, {}, {}
            for _, _, part, _ in selected:
                path = output / domain['name'] / method / f'part_{part:04d}.json'
                if not path.exists():
                    result['completed'] = False
                    break
                record = json.loads(path.read_text())
                if record['protocol_hash'] != protocol_hash:
                    raise ValueError('不能合并不同协议的评测')
                add_statistics(total, record['statistics'])
                for frame in record['frames']:
                    indices.append(frame['index'])
                    if 'valid_views' in frame:
                        add_statistics(by_views.setdefault(str(frame['valid_views']), {}), frame['statistics'])
                    for band, values in frame.get('bands', {}).items():
                        add_statistics(bands.setdefault(band, {}), values)
                    if 'input_sha256' in frame:
                        hashes[frame['index']] = frame['input_sha256']
            else:
                intended = sorted(i for t in selected for i in t[3])
                if sorted(indices) != intended or len(set(indices)) != len(indices):
                    raise ValueError('分片重复或漏帧')
                if expected_gt is not None and total['gt_pixels'] != expected_gt:
                    raise ValueError('同数据集不同方法GT分母不一致')
                for index, value in hashes.items():
                    if index in input_hashes and input_hashes[index] != value:
                        raise ValueError('同帧不同方法实际输入不一致')
                    input_hashes[index] = value
                expected_gt = total['gt_pixels']
                methods[method] = {'frames': len(indices), 'statistics': total,
                                   'metrics': metrics(total, domain['geometry_units']),
                                   'distance_bands': {key: metrics(value, domain['geometry_units'])
                                                      for key, value in bands.items()},
                                   'by_valid_views': {key: metrics(value, domain['geometry_units'])
                                                      for key, value in by_views.items()}}
        result['datasets'][domain['name']] = {'units': domain['geometry_units'],
                                             'num_views': domain['num_views'], 'methods': methods}
    write_json(output / 'summary.json', result)
    return result


def timing(config, output):
    offline_only()
    device = configure(config, config['gpus'][0])
    models = Models(config, device)
    reports = {}
    for domain in config['datasets']:
        if domain['name'] not in config['timing']['datasets']:
            continue
        if config['timing'].get('split', 'test') == 'val':
            from jar.data import build_dataset
            from jar.pretraining.data import dataset_config

            dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=False)
        else:
            dataset = test_dataset(config, domain)
        sample = dataset[len(dataset) // 2]
        inputs = move(collate_frames([sample])['inputs'], device)
        names = [name for name, option in config['methods'].items()
                 if option.get('num_views', domain['num_views']) == domain['num_views']]
        for name in names:
            models.get(name, domain)
        durations = {name: {'forward_ms': [], 'with_resampling_ms': []} for name in names}
        with torch.inference_mode():
            for name in names:
                for _ in range(config['timing']['warmup']):
                    prediction = models.forward(name, domain, inputs)
                    converted = models.convert(prediction, inputs)
                    del prediction, converted
            torch.cuda.synchronize(device)
            for repeat in range(config['timing']['repetitions']):
                # 同一张卡交替、轮换顺序，准确率并行作业已全部退出。
                order = names[repeat % len(names):] + names[:repeat % len(names)]
                for name in order:
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    prediction = models.forward(name, domain, inputs)
                    torch.cuda.synchronize(device)
                    middle = time.perf_counter()
                    converted = models.convert(prediction, inputs)
                    torch.cuda.synchronize(device)
                    end = time.perf_counter()
                    durations[name]['forward_ms'].append((middle - begin) * 1000)
                    durations[name]['with_resampling_ms'].append((end - begin) * 1000)
                    del prediction, converted
        reports[domain['name']] = {name: {
            **{key: {'median': statistics.median(values), 'mean': statistics.mean(values),
                     'p90': sorted(values)[math_index(len(values), .9)]} for key, values in record.items()},
            'single_reference_hz': 1000 / statistics.median(record['with_resampling_ms'])}
            for name, record in durations.items()}
    write_json(output / 'latency.json', {'completed': True, 'results': reports,
               'gpu': torch.cuda.get_device_name(device), 'precision': 'fp32', 'tf32': config['tf32'],
               'scope': '单参考RGB至深度；含输入适配和可选深度格点重采样，不含读盘/传输/融合。不是四参考场景频率。',
               'timing': config['timing']})


def math_index(length, quantile):
    return min(length - 1, max(0, int(length * quantile)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke', action='store_true', help='各域首中尾；输出到独立smoke目录')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config['output']) / ('smoke' if args.smoke else 'full')
    output.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    sources = ['jar/pretraining/comparison.py', 'tools/compare_public_pretrained.py']
    audit = {'config_sha256': sha, 'source_sha256': {p: digest(p) for p in sources},
             'weights_sha256': {}, 'test_manifest_sha256': {d['name']: digest(d['manifests']['test']) for d in config['datasets']}}
    for path in {config['own_weights']} | {m['weights'] for m in config['methods'].values() if 'weights' in m}:
        audit['weights_sha256'][path] = digest(path)
    protocol_hash = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    audit['protocol_hash'] = protocol_hash
    previous_audit = output / 'audit.json'
    if previous_audit.exists() and not args.smoke:
        previous = json.loads(previous_audit.read_text())
        if previous['protocol_hash'] != protocol_hash:
            raise FileExistsError('已有其他源码/权重/协议的正式结果，请在配置中使用新output目录')
    write_json(output / 'audit.json', audit)
    write_json(output / 'config.json', config)
    tasks, blocked = [], []
    for domain_index, domain in enumerate(config['datasets']):
        length = len(test_dataset(config, domain))
        indices = [0, length // 2, length - 1] if args.smoke else list(range(length))
        for method, settings in config['methods'].items():
            if settings.get('datasets') and domain['name'] not in settings['datasets']:
                blocked.append({'dataset': domain['name'], 'method': method,
                                'reason': '配置明确限定适用数据域；不将缺相机帧伪装成完整输入。'})
                continue
            if settings.get('num_views', domain['num_views']) != domain['num_views']:
                blocked.append({'dataset': domain['name'], 'method': method,
                                'reason': '四目MLP权重不兼容真实双目；不复制图像或改层。'})
                continue
            for part, start in enumerate(range(0, len(indices), config['shard_size'])):
                tasks.append((domain_index, method, part, indices[start:start + config['shard_size']]))
    # 先调度较大分片，尾部小任务填充空闲GPU。
    tasks.sort(key=lambda t: len(t[3]), reverse=True)
    context = mp.get_context('spawn')
    queue, results = context.Queue(), context.Queue()
    gpus = config['gpus'][:2] if args.smoke else config['gpus']
    processes = [context.Process(target=worker, args=(config, str(output), gpu, queue, results, protocol_hash, args.smoke)) for gpu in gpus]
    for task in tasks:
        queue.put(task)
    for _ in processes:
        queue.put(None)
    for process in processes:
        process.start()
    failures = []
    with (output / 'progress.jsonl').open('a') as log:
        remaining = len(tasks)
        while remaining:
            try:
                result = results.get(timeout=30)
            except queue_module.Empty:
                crashed = [p.pid for p in processes if p.exitcode not in (None, 0)]
                if crashed or all(not p.is_alive() for p in processes):
                    raise RuntimeError(f'评测进程提前退出，剩余{remaining}分片，异常进程{crashed}')
                continue
            remaining -= 1
            log.write(json.dumps(result, ensure_ascii=False) + '\n')
            log.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result['event'] == 'failed':
                failures.append(result)
    for process in processes:
        process.join()
    summary = summarize(config, output, tasks, blocked, protocol_hash)
    if failures or not summary['completed']:
        raise RuntimeError(f'{len(failures)}个评测分片失败；见failures目录，未完成项不生成假指标')
    if not args.smoke:
        timing(config, output)
    print('completed', output, flush=True)


if __name__ == '__main__':
    main()
