"""V7零初始化独立模块的真实GPU研究测速，不替代最终独占部署评测。"""

import gc
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.pretraining.inference import prepare_inference
from jar.runtime import move
from research.rulermvs_v6_20261101.inference import prepare_fast_inference
from research.v7_domain_20261102.model import build_model
from tools.smoke_outdoor_datasets import fingerprint, offline_only

VARIANTS = ('control', 'projective', 'prior')


def distribution(values):
    return {'median_ms': float(np.median(values)), 'mean_ms': float(np.mean(values)),
            'p95_ms': float(np.percentile(values, 95)), 'calls': len(values), 'samples_ms': values}


def main():
    settings = json.loads(Path(__file__).with_suffix('.json').read_text())
    output = ROOT / settings['output']
    if output.exists():
        raise FileExistsError(output)
    if not output.resolve().is_relative_to('/data'):
        raise ValueError('测速产物与编译缓存必须放数据盘')
    output.mkdir(parents=True)
    sources = [Path(__file__), *(Path(__file__).with_name(name) for name in ('model.py', 'center.py', 'prior.py'))]
    source_hash = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for source in sources:
        destination = output / 'source' / source.name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(source.read_bytes())
    offline_only()
    torch.set_num_threads(2)
    torch.cuda.set_device(settings['gpu'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = 'cuda:' + str(settings['gpu'])
    report = {'completed': False, 'settings': settings, 'source_sha256': source_hash,
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'datasets': {},
              'scope': '共同256x448最终深度，含equal_pixels输入适配，排除IO/H2D/冷编译/点云融合。',
              'limits': 'GPU0无其他计算，但其他卡可同时pilot；研究增量计时，不作正式整机延迟或显存结论。',
              'precision': 'FP32/TF32off', 'targets_loaded': False}
    write_json(output / 'plan.json', report)
    rng = random.Random(20261102)
    for phase in settings['phases']:
        config = json.loads(Path(phase['data_config']).read_text())
        domain = next(d for d in config['datasets'] if d['name'] == phase['dataset'])
        dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=False)
        indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
        weights = config['methods']['v6_best']['weights']
        saved = torch.load(weights, map_location='cpu', weights_only=False, mmap=True)
        record = {'num_views': domain['num_views'], 'image_size': config['image_size'],
                  'indices': indices, 'weights': weights, 'weights_sha256': digest(weights),
                  'methods': {}, 'checks': [], 'input_sha256': {}, 'warmup_seconds': {}}
        cpu_inputs = [collate_frames([dataset[index]])['inputs'] for index in indices]
        for index, data in zip(indices, cpu_inputs):
            assert data['rgb'].shape[1] == domain['num_views']
            assert data['view_mask'].all()
            record['input_sha256'][str(index)] = fingerprint(data)
        for mode in ('eager', 'fast'):
            models = {}
            for variant in VARIANTS:
                with torch.inference_mode(False), torch.no_grad():
                    model, _ = build_model(saved, variant)
                    model = model.to(device).eval().requires_grad_(False)
                    install_position_cache(model.foundation.model)
                    if mode == 'fast':
                        prepare_fast_inference(model, {'compile_source': True, 'compile_fusion': True,
                                                       'cache_dir': str(output / 'compile_cache')})
                    else:
                        prepare_inference(model, spatial_layout=True, projective_ruler=False)
                models[variant] = model
                record['methods'][mode + '/' + variant] = {
                    'parameters': sum(p.numel() for p in model.parameters())}
            durations = {v: [] for v in VARIANTS}
            with torch.inference_mode():
                for index, cpu in zip(indices, cpu_inputs):
                    inputs = move(cpu, device)
                    def call(name):
                        native = models[name](equal_pixel_inputs(inputs), domain, domain['reference_index'])
                        return {'depth': native['depth'] * native['unit_scale'], 'valid': native['valid']}
                    for name in VARIANTS:
                        start = time.perf_counter()
                        for _ in range(settings['warmup']):
                            call(name)
                        torch.cuda.synchronize()
                        record['warmup_seconds'][f'{mode}/{name}/{index}'] = time.perf_counter() - start
                    for _ in range(settings['repetitions']):
                        order = list(VARIANTS)
                        rng.shuffle(order)
                        for name in order:
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            current = call(name)
                            torch.cuda.synchronize()
                            durations[name].append((time.perf_counter() - start) * 1000)
                            assert current['depth'].shape == (1, 1, 256, 448)
                            del current
                    baseline = call('control')
                    for name in VARIANTS[1:]:
                        current = call(name)
                        delta = (current['depth'] - baseline['depth']).abs()
                        record['checks'].append({'mode': mode, 'variant': name, 'index': index,
                                                 'depth_exact': torch.equal(current['depth'], baseline['depth']),
                                                 'valid_exact': torch.equal(current['valid'], baseline['valid']),
                                                 'max_depth_difference_m': float(delta.max()),
                                                 'mean_depth_difference_m': float(delta.mean()),
                                                 'depth_finite': bool(current['depth'].isfinite().all())})
                        del current, delta
                    del baseline, inputs
                    print(json.dumps({'dataset': domain['name'], 'mode': mode, 'frame': index}), flush=True)
            control_median = float(np.median(durations['control']))
            for name in VARIANTS:
                record['methods'][mode + '/' + name].update(distribution(durations[name]))
                record['methods'][mode + '/' + name]['increment_vs_control_ms'] = float(np.median(durations[name])) - control_median
            models.clear()
            del model, models, call
            gc.collect()
            torch.cuda.empty_cache()
            report['datasets'][domain['name']] = record
            write_json(output / 'results.json', report)
        del saved, cpu_inputs
    for source in sources:
        if digest(source) != source_hash[str(source.relative_to(ROOT))]:
            raise RuntimeError('测速期间源码改变；保留记录但不能声称对应当前源码')
    report['completed'] = True
    write_json(output / 'results.json', report)
    print(json.dumps({'completed': True, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
