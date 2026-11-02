"""完整 RGB 单参考计时及缓存等价核验；配置全部来自 JSON，不下载权重。

配置示例：
{
  "checkpoint": "runs/active/public_refinement_20261025/recommended/pretrained.pt",
  "decoder_checkpoints": {"relation140": "runs/active/ruler_research_20261027/relation/best.pt"},
  "baseline_weights": "runs/carla_research_20261011/cache/mvsanywhere_hero.ckpt",
  "cache": "runs/active/architecture_screen_20261026/features",
  "datasets": ["TartanAir_forest", "VKITTI2"],
  "device": "cuda:0",
  "output": "runs/active/ruler_research_20261027/timing",
  "verify_cache": true,
  "warmup": 8,
  "repetitions": 30,
  "cpu_threads": 2
}

每域使用既有 DEV 的首、中、末三帧；全部模型同卡轮换。
计时含 RGB 编码、解码及薄适配，不含读盘、H2D、点云/多参考融合。
实时数据不加载 targets；缓存核验只使用 work，不把标签传给模型。
"""

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import Models, digest, write_json
from jar.pretraining.data import dataset_config
from jar.pretraining.inference import prepare_inference
from jar.pretraining.model import PretrainingModel
from jar.runtime import move
from research.ruler_joint_20261027.model import make_decoder
from tools.smoke_outdoor_datasets import fingerprint, offline_only


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def distribution(values):
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        'calls': len(ordered), 'median_ms': median,
        'mean_ms': statistics.mean(ordered),
        'p90_ms': ordered[math.ceil(.9 * len(ordered)) - 1],
        'min_ms': ordered[0], 'max_ms': ordered[-1],
        'single_reference_hz': 1000 / median,
    }


def reference_first(inputs, reference):
    """统一物理参考相机为槽0，保持其余源次序，与冻结编码缓存一致。"""
    views = inputs['rgb'].shape[1]
    order = [reference] + [view for view in range(views) if view != reference]
    return {
        key: value[:, order] if torch.is_tensor(value) and value.ndim >= 2
        and value.shape[:2] == (1, views) else value
        for key, value in inputs.items()
    }


def build_own_models(saved, plan, device):
    models, records = {}, {}
    paths = {'original': None, **plan.get('decoder_checkpoints', {})}
    if plan.get('decoder_checkpoints', {}).get('original') is not None:
        raise ValueError('original 名称保留给未修改的推荐模型')
    for name, path in paths.items():
        model = PretrainingModel(saved['config'], initialize=False)
        model.load_state_dict(saved['model'], strict=True)
        if path is not None:
            path = project_path(path)
            record = torch.load(path, map_location='cpu', weights_only=False)
            if digest(project_path(record['plan']['checkpoint'])) != digest(project_path(plan['checkpoint'])):
                raise ValueError(f'{name} 的编码器起点与计时 checkpoint 不一致')
            decoder, settings = make_decoder(saved, record['plan'], device)
            if settings != record['settings']:
                raise ValueError(f'{name} 的模型设置与 checkpoint 不一致')
            decoder.load_state_dict(record['model'], strict=True)
            model.decoder = decoder
            records[name] = {'checkpoint': str(path), 'sha256': digest(path),
                             'step': record['step'], 'settings': settings}
        else:
            records[name] = {'checkpoint': plan['checkpoint'],
                             'sha256': digest(project_path(plan['checkpoint']))}
        model = model.to(device).eval().requires_grad_(False)
        install_position_cache(model.foundation.model)
        prepare_inference(model, spatial_layout=True)
        models[name] = model
    return models, records


@torch.inference_mode()
def verify_cached_prediction(models, inputs, domain, cache_path, device):
    """同一帧完整RGB与冻结work运行同一个已加载解码器，结果单位相同。"""
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    work = move(cached['work'], device)
    reports = {}
    for name, model in models.items():
        full = model(inputs, domain, reference=0)
        model.decoder.set_range(domain['near'] / domain['depth_scale'],
                                domain['far'] / domain['depth_scale'])
        encoded = model.decoder(work)
        torch.testing.assert_close(full['depth'], encoded['depth'], rtol=1e-5, atol=1e-6)
        if not torch.equal(full['valid'], encoded['valid']):
            raise AssertionError(f'{name} 缓存与RGB有效性不一致')
        reports[name] = {
            'depth_exact': torch.equal(full['depth'], encoded['depth']),
            'max_depth_difference_original_units': float(
                (full['depth'] - encoded['depth']).abs().max() * domain['depth_scale']),
            'valid_exact': True, 'cache': str(cache_path),
        }
    return reports


@torch.inference_mode()
def measure_frame(predict, names, inputs, domain, device, warmup, repetitions):
    shapes = {}
    for _ in range(warmup):
        for name in names:
            output = predict(name, domain, inputs)
            if not torch.isfinite(output['depth']).all():
                raise ValueError(f'{name} 输出非有限')
            shapes[name] = list(output['depth'].shape)
            del output
    torch.cuda.synchronize(device)
    samples = {name: {'wall_network_ms': [], 'cuda_network_ms': []} for name in names}
    for repeat in range(repetitions):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in order:
            begin_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            begin_event.record()
            prediction = predict(name, domain, inputs)
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000
            samples[name]['wall_network_ms'].append(elapsed)
            samples[name]['cuda_network_ms'].append(begin_event.elapsed_time(end_event))
            del prediction
    return {name: {'samples': values, 'native_depth_shape': shapes[name],
                   'summary': {key: distribution(times) for key, times in values.items()}}
            for name, values in samples.items()}


def run(plan):
    offline_only()
    device = torch.device(plan.get('device', 'cuda:0'))
    if device.type != 'cuda':
        raise ValueError('本脚本用于单卡CUDA计时；CPU静态检查不运行run')
    torch.cuda.set_device(device)
    torch.set_num_threads(plan.get('cpu_threads', 2))
    torch.manual_seed(plan.get('seed', 20261027))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = project_path(plan['output'])
    if output.exists():
        raise FileExistsError(f'拒绝覆盖已有计时：{output}')
    output.mkdir(parents=True)
    checkpoint = project_path(plan['checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cache_root = project_path(plan['cache'])
    cache_index = json.loads((cache_root / 'index.json').read_text())
    if cache_index['checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('DEV缓存必须来自当前推荐checkpoint')
    domains = {domain['name']: domain for domain in saved['config']['datasets']}
    own, records = build_own_models(saved, plan, device)
    hero_path = project_path(plan['baseline_weights'])
    baseline_options = {
        'image_size': saved['config']['image_size'],
        'position_cache': True, 'fast_cost_volume': True,
        'methods': {
            name: {'kind': 'mvsanywhere', 'weights': str(hero_path),
                   'refinements': count, 'shared_train_range': True}
            for name, count in [('mvsanywhere_shared_first', 0),
                                ('mvsanywhere_shared_refined', 1)]
        },
    }
    if set(own) & set(baseline_options['methods']):
        raise ValueError('自研候选名称与baseline名称重复')
    baselines = Models(baseline_options, device)
    names = list(own) + list(baseline_options['methods'])

    def predict(name, domain, inputs):
        if name not in own:
            return baselines.forward(name, domain, inputs)
        result = own[name](inputs, domain, reference=0)
        return {'depth': result['depth'] * result['unit_scale'],
                'valid': result['valid'], 'reference_indices': [0]}

    warmup, repetitions = plan.get('warmup', 8), plan.get('repetitions', 30)
    if warmup < 1 or repetitions < 1:
        raise ValueError('warmup及repetitions必须为正整数')
    report = {
        'completed': False, 'plan': plan, 'models': records,
        'baseline_sha256': digest(hero_path), 'cache_index_sha256': digest(cache_root / 'index.json'),
        'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'precision': 'FP32 TF32 off',
        'position_cache': True, 'fast_cost_volume': True,
        'scope': 'batch1完整RGB到单参考原生深度，含输入适配；不含读盘/H2D/点云或多参考融合。',
        'sample_scope': '每域既有DEV首中末3帧的重复计时，不是全TEST耗时分布或车端频率。',
        'pixel_budget': '主图沿用checkpoint，自研匹配沿用原生较大图；baseline原生设置，不声称同像素预算。',
        'inputs': 'load_targets=False；各方法使用同一参考相机，计时前统一到槽0。缓存核验只使用work。',
        'memory': '所有模型共驻，此处不报告方法独占峰值显存。',
        'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), Path(__file__).with_name('model.py'),
            ROOT / 'jar/pretraining/comparison.py', ROOT / 'jar/pretraining/inference.py')},
        'datasets': {},
    }
    write_json(output / 'plan.json', plan)
    for name in plan.get('datasets', ['TartanAir_forest', 'VKITTI2']):
        domain = copy.deepcopy(domains[name])
        original_reference = domain['reference_index']
        domain['reference_index'] = 0
        paths = cache_index['splits']['development'][name]
        chosen = [paths[i] for i in dict.fromkeys((0, len(paths) // 2, len(paths) - 1))]
        dataset = build_dataset(dataset_config(domains[name], saved['config']), 'val', load_targets=False)
        for method in baseline_options['methods']:
            baselines.get(method, domain)
        frames, parity = [], None
        for position, cache_path in enumerate(chosen):
            index = int(Path(cache_path).stem)
            sample = dataset[index]
            if sample.get('targets'):
                raise ValueError('无标签计时入口不应加载targets')
            cpu_inputs = reference_first(collate_frames([sample])['inputs'], original_reference)
            inputs = move(cpu_inputs, device)
            if position == 0 and plan.get('verify_cache', True):
                parity = verify_cached_prediction(own, inputs, domain, project_path(cache_path), device)
            frame = {'index': index, 'input_sha256': fingerprint(cpu_inputs),
                     'rgb_shape': list(inputs['rgb'].shape),
                     'matching_rgb_shape': list(inputs['matching_rgb'].shape),
                     'methods': measure_frame(predict, names, inputs, domain, device, warmup, repetitions)}
            frames.append(frame)
            write_json(output / name / f'frame_{index:06d}.json', frame)
            print(json.dumps({'dataset': name, 'index': index,
                              'median_ms': {method: row['summary']['wall_network_ms']['median_ms']
                                            for method, row in frame['methods'].items()}}), flush=True)
            del inputs, cpu_inputs, sample
        summary = {
            method: {field: distribution([value for frame in frames
                                          for value in frame['methods'][method]['samples'][field]])
                     for field in ('wall_network_ms', 'cuda_network_ms')}
            for method in names
        }
        report['datasets'][name] = {
            'frames': [frame['index'] for frame in frames], 'num_views': domain['num_views'],
            'physical_reference_original_slot': original_reference,
            'reference_tensor_slot': 0, 'cache_parity': parity, 'summary': summary,
            'val_manifest_sha256': digest(domain['manifests']['val']),
        }
        write_json(output / 'latency.json', report)
    report['completed'] = True
    write_json(output / 'latency.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(json.loads(project_path(parser.parse_args().config).read_text()))
