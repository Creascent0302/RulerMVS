"""局部融合编译与尺度读取统计；不改训练结构或checkpoint，时间标记为竞争测量。"""

import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / 'runs/active/rulermvs_v6_20261101/speed_compile/cache'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(CACHE / 'inductor')
os.environ['TRITON_CACHE_DIR'] = str(CACHE / 'triton')
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.pretraining.model import CalibratedDecoder
from jar.runtime import move
from research.rulermvs_v5_20261031.infer import load_model
from research.rulermvs_v6_20261101.speed_options import differences


def kernel_counts(call):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        call()
        torch.cuda.synchronize()
    return {
        'cuda_events': sum('CUDA' in str(event.device_type) for event in profile.events()),
        'aten_calls': sum(event.count for event in profile.key_averages() if event.key.startswith('aten::')),
    }


def main():
    config = json.loads((ROOT / 'research/baseline_full_comparison_20261101/configs/speed_data.json').read_text())
    output = CACHE.parent
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    domain = config['datasets'][0]
    dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=False)
    indices = (0, len(dataset) // 2, len(dataset) - 1)
    inputs = [move(collate_frames([dataset[index]])['inputs'], 'cuda:0') for index in indices]
    inference = {'spatial_layout': True, 'factorized_fusion': True,
                 'hoist_reference': True, 'compact_pair': True}
    models = {}
    compile_seconds = {}
    checkpoint = config['methods']['v5_best']['weights']
    for name in ('eager_algebra', 'compile_source', 'compile_fusion', 'compile_both'):
        model, _ = load_model(checkpoint, 'cuda:0', inference=inference)
        if name in ('compile_source', 'compile_both'):
            model.decoder.source.forward = torch.compile(
                model.decoder.source.forward, fullgraph=True, dynamic=False)
        if name in ('compile_fusion', 'compile_both'):
            model.decoder._tokens_from_source = torch.compile(
                model.decoder._tokens_from_source, fullgraph=True, dynamic=False)
        models[name] = model
    routing = []
    current_frame = None

    def capture(frame, event, argument):
        if event == 'return' and frame.f_code is CalibratedDecoder._observe.__code__:
            level = frame.f_locals['level'].detach().cpu()
            valid = frame.f_locals['valid'].detach().cpu().reshape_as(level)
            row = {'frame': current_frame, 'stride': frame.f_locals['stride'],
                   'candidate_source_count': level.numel(), 'valid_count': int(valid.sum()),
                   'min_level': float(level.min()), 'max_level': float(level.max()), 'layers': {}}
            for index, stride in enumerate((4, 8, 16)):
                active = (1 - (level - index).abs()).clamp_min(0) > 0
                row['layers'][stride] = {'active_fraction': float(active.float().mean()),
                                        'active_valid_fraction': float(active[valid].float().mean()),
                                        'whole_layer_unused': not bool(active.any())}
            routing.append(row)

    result = {'completed': False, 'load_targets': False, 'indices': indices, 'split': 'val',
              'checkpoint': checkpoint, 'precision': 'FP32/TF32off', 'inference': inference,
              'timing_is_contended': True, 'compile_seconds': compile_seconds,
              'routing': routing, 'checks': {}, 'timings': {}, 'kernel_counts': {}, 'failures': {}}
    with torch.inference_mode():
        for frame, item in zip(indices, inputs):
            current_frame = frame
            sys.setprofile(capture)
            try:
                models['eager_algebra'](equal_pixel_inputs(item), domain, domain['reference_index'])
            finally:
                sys.setprofile(None)
        write_json(output / 'results.json', result)
        print('routing', json.dumps(routing), flush=True)
        failed = []
        for name, model in models.items():
            start = time.perf_counter()
            try:
                for _ in range(3):
                    model(equal_pixel_inputs(inputs[0]), domain, domain['reference_index'])
                torch.cuda.synchronize()
                compile_seconds[name] = time.perf_counter() - start
                print('compiled', name, compile_seconds[name], flush=True)
            except Exception as error:
                result['failures'][name] = repr(error)
                failed.append(name)
            write_json(output / 'results.json', result)
        for name in failed:
            del models[name]
        timings = {name: [] for name in models}
        checks = {name: [] for name in models}
        rng = random.Random(20261101)
        for frame, item in zip(indices, inputs):
            for model in models.values():
                for _ in range(5):
                    model(equal_pixel_inputs(item), domain, domain['reference_index'])
            for _ in range(20):
                order = list(models)
                rng.shuffle(order)
                for name in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    current = models[name](equal_pixel_inputs(item), domain, domain['reference_index'])
                    torch.cuda.synchronize()
                    timings[name].append((time.perf_counter() - start) * 1000)
                    del current
            reference = models['eager_algebra'](equal_pixel_inputs(item), domain, domain['reference_index'])
            for name, model in models.items():
                current = model(equal_pixel_inputs(item), domain, domain['reference_index'])
                checks[name].append({'frame': frame, **differences(current, reference)})
                del current
            del reference
        for name, model in models.items():
            call = lambda: model(equal_pixel_inputs(inputs[1]), domain, domain['reference_index'])
            result['kernel_counts'][name] = kernel_counts(call)
        result.update(completed=True, checks=checks,
                      timings={name: {'median_ms': statistics.median(values), 'samples_ms': values}
                               for name, values in timings.items()})
        write_json(output / 'results.json', result)
        print(json.dumps({name: {'median_ms': statistics.median(timings[name]),
                                'checks': checks[name], 'kernels': result['kernel_counts'][name]}
                          for name in models}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
