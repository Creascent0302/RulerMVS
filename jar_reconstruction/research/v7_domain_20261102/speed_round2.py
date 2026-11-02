"""坐标码推理减法：真实N2/N4输入、初值及两步smoke权重、同卡交替计时。"""

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
from research.v7_domain_20261102.inference_round2 import prepare_coordinate_reuse
from research.v7_domain_20261102.model_round2 import build_model
from research.v7_domain_20261102.speed_probe import distribution
from tools.smoke_outdoor_datasets import fingerprint, offline_only

BASE = ROOT / 'runs/active/v7_domain_20261102'
OUTPUT = BASE / 'inference_round2_probe'


def load(saved, variant, *, fast=False, reuse=False):
    with torch.inference_mode(False), torch.no_grad():
        model, _ = build_model(saved, variant)
        model = model.to('cuda:0').eval().requires_grad_(False)
        install_position_cache(model.foundation.model)
        if reuse:
            prepare_coordinate_reuse(model)
        if fast:
            prepare_fast_inference(model, {'compile_source': True, 'compile_fusion': True,
                                           'cache_dir': str(OUTPUT / 'compile_cache')})
        else:
            prepare_inference(model, spatial_layout=True)
    return model


def differences(a, b):
    result = {}
    for stage, first, second in [('depth', a['depth'], b['depth']),
                                 ('coarse', a['coarse']['depth'], b['coarse']['depth']),
                                 ('fine', a['fine']['depth'], b['fine']['depth'])]:
        change = (first - second).abs() * float(a['unit_scale'])
        result[stage] = {'exact': torch.equal(first, second), 'max_absolute_m': float(change.max()),
                         'mean_absolute_m': float(change.mean()), 'finite': bool(first.isfinite().all())}
    result['valid_exact'] = torch.equal(a['valid'], b['valid'])
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    offline_only()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    v6_path = ROOT / 'runs/active/rulermvs_v6_20261101/full_training/carla_new/best.pt'
    smoke_path = BASE / 'smoke_round2/coordinate_new_only/final.pt'
    original_state = torch.load(v6_path, map_location='cpu', mmap=True, weights_only=False)
    smoke_state = torch.load(smoke_path, map_location='cpu', mmap=True, weights_only=False)
    sources = [Path(__file__), *(Path(__file__).with_name(name) for name in ('inference_round2.py', 'coordinate_prior.py', 'model_round2.py'))]
    report = {'completed': False, 'datasets': {}, 'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
              'weights_sha256': {'V6': digest(v6_path), 'coordinate_smoke': digest(smoke_path)},
              'precision': 'FP32/TF32off', 'load_targets': False, 'gpu': torch.cuda.get_device_name(),
              'notes': ['数学函数相同；只在粗候选H/W stride均0时提升公共深度编码；其他情况回退原forward。',
                        '精度检查用未训练初值和仅2步smoke权重，不是最终正式权重完整VAL精度。',
                        'N2 VKITTI2与N4 CARLA各真实VAL首中末，相同256x448图像信息。',
                        '所有模型为CARLA V6起点；N2仅检查实现/成本，不当跨域精度。',
                        'GPU0独占，其他卡可能训练；研究计时包含equal_pixels适配和最终深度单位换算，排除IO/H2D/冷编译/融合。',
                        'fast与V6相同source/fusion编译；fusion已包含prior，不另外套重复的compile_prior。']}
    write_json(OUTPUT / 'plan.json', report)
    data_specs = [('public_main_val.json', 'VKITTI2'), ('carla_new_main_val.json', 'CARLA_new_upper')]
    rng = random.Random(20261102)
    for config_name, domain_name in data_specs:
        config = json.loads((ROOT / 'runs/active/rulermvs_v6_20261101/full_training/evaluation_recovery/configs' / config_name).read_text())
        domain = next(d for d in config['datasets'] if d['name'] == domain_name)
        dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=False)
        indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
        raw_inputs = [collate_frames([dataset[index]])['inputs'] for index in indices]
        inputs = [move(item, 'cuda:0') for item in raw_inputs]
        record = {'num_views': domain['num_views'], 'indices': indices,
                  'input_sha256': {str(i): fingerprint(item) for i, item in zip(indices, raw_inputs)},
                  'checks': [], 'timings': {}, 'warmup_seconds': {}, 'depth_code_input_shapes': {}}
        # 可关闭契约与初始/非零参数的真实整网前向，不读取标签。
        for checkpoint_name, saved in [('initial', original_state), ('smoke', smoke_state)]:
            model = load(saved, 'coordinate')
            keys = tuple(model.state_dict())
            with torch.inference_mode():
                for index, item in zip(indices, inputs):
                    prepared = equal_pixel_inputs(item)
                    old = model(prepared, domain, domain['reference_index'])
                    prepare_coordinate_reuse(model)
                    shapes = []
                    handle = model.decoder.context_prior.depth_code.register_forward_pre_hook(
                        lambda module, args: shapes.append(list(args[0].shape)))
                    new = model(prepared, domain, domain['reference_index'])
                    handle.remove()
                    prepare_coordinate_reuse(model, False)
                    restored = model(prepared, domain, domain['reference_index'])
                    restored_check = differences(restored, old)
                    assert all(restored_check[stage]['exact'] for stage in ('depth', 'coarse', 'fine'))
                    assert restored_check['valid_exact']
                    assert tuple(model.state_dict()) == keys
                    record['checks'].append({'mode': 'eager', 'checkpoint': checkpoint_name, 'index': index,
                                             'reuse_vs_original': differences(new, old), 'disable_exact': True})
                    record['depth_code_input_shapes'][f'{checkpoint_name}/{index}'] = shapes
                    assert shapes == [[1, 96, 1, 1, 13]]
                    del old, new, restored, prepared
            del model
            gc.collect()
            torch.cuda.empty_cache()
        models = {'v6_fast': load(original_state, 'control', fast=True),
                  'coordinate_original_fast': load(smoke_state, 'coordinate', fast=True),
                  'coordinate_reuse_fast': load(smoke_state, 'coordinate', fast=True, reuse=True)}
        durations = {name: [] for name in models}
        with torch.inference_mode():
            for index, item in zip(indices, inputs):
                def call(name):
                    native = models[name](equal_pixel_inputs(item), domain, domain['reference_index'])
                    native['metric_depth'] = native['depth'] * native['unit_scale']
                    return native
                for name in models:
                    started = time.perf_counter()
                    for _ in range(10):
                        call(name)
                    torch.cuda.synchronize()
                    record['warmup_seconds'][f'{name}/{index}'] = time.perf_counter() - started
                for _ in range(30):
                    order = list(models)
                    rng.shuffle(order)
                    for name in order:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        current = call(name)
                        torch.cuda.synchronize()
                        durations[name].append((time.perf_counter() - started) * 1000)
                        assert current['metric_depth'].shape == (1, 1, 256, 448)
                        del current
                old, new = call('coordinate_original_fast'), call('coordinate_reuse_fast')
                record['checks'].append({'mode': 'fast', 'checkpoint': 'smoke', 'index': index,
                                         'reuse_vs_original': differences(new, old)})
                del old, new
                print(json.dumps({'dataset': domain_name, 'frame': index}), flush=True)
        for name, values in durations.items():
            record['timings'][name] = distribution(values)
            record['timings'][name]['increment_vs_v6_ms'] = float(np.median(values)) - float(np.median(durations['v6_fast']))
        report['datasets'][domain_name] = record
        write_json(OUTPUT / 'results.json', report)
        models.clear()
        del models, inputs, item, call
        gc.collect()
        torch.cuda.empty_cache()
    assert all(digest(p) == report['source_sha256'][str(p.relative_to(ROOT))] for p in sources)
    report['completed'] = True
    write_json(OUTPUT / 'results.json', report)
    print(json.dumps({'completed': True, 'output': str(OUTPUT)}), flush=True)


if __name__ == '__main__':
    main()
