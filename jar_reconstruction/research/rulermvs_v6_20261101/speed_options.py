"""固定V5权重比较推理代数复用与Conv-BN折叠；不替换生产默认。"""

import gc
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.rulermvs_v5_20261031.infer import load_model


def fuse_backbone(model):
    """只在eval副本内折叠ResNet的固定BN，不用于继续训练或保存原结构权重。"""
    network = model.matching.backbone
    count = 0
    for module in network.modules():
        for name in ('1', '2'):
            conv, bn = getattr(module, 'conv' + name, None), getattr(module, 'bn' + name, None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(module, 'conv' + name, fuse_conv_bn_eval(conv, bn))
                setattr(module, 'bn' + name, nn.Identity())
                count += 1
        if hasattr(module, 'downsample') and isinstance(module.downsample, nn.Sequential):
            if isinstance(module.downsample[0], nn.Conv2d) and isinstance(module.downsample[1], nn.BatchNorm2d):
                module.downsample[0] = fuse_conv_bn_eval(module.downsample[0], module.downsample[1])
                module.downsample[1] = nn.Identity()
                count += 1
    return count


def differences(output, reference):
    value = (output['depth'] - reference['depth']).abs()
    record = {
        'finite_depth': bool(torch.isfinite(output['depth']).all()),
        'valid_identical': bool(torch.equal(output['valid'], reference['valid'])),
        'depth_max_m': float(value.max()), 'depth_mean_m': float(value.mean()),
        'depth_fraction_gt_001m': float((value > .01).float().mean()),
        'depth_fraction_gt_01m': float((value > .1).float().mean()),
        'depth_fraction_gt_1m': float((value > 1).float().mean()),
    }
    for stage in ('coarse', 'fine'):
        record[stage + '_selected_change_fraction'] = float(
            (output[stage]['selected_index'] != reference[stage]['selected_index']).float().mean())
    return record


def main():
    config = json.loads((ROOT / 'research/baseline_full_comparison_20261101/configs/speed_data.json').read_text())
    output = ROOT / 'runs/active/rulermvs_v6_20261101/speed_options'
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    domain = config['datasets'][0]
    weights = config['methods']['v5_best']['weights']
    dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=False)
    indices = (0, len(dataset) // 2, len(dataset) - 1)
    inputs = [move(collate_frames([dataset[index]])['inputs'], 'cuda:0') for index in indices]
    algebra = {'factorized_fusion': True, 'hoist_reference': True, 'compact_pair': True}
    options = {
        'baseline': {}, 'algebra': algebra, 'projective': {'projective_ruler': True},
        'reuse_all': dict(algebra, projective_ruler=True),
        'bn_only': {'fuse_bn': True},
        'reuse_bn': dict(algebra, projective_ruler=True, fuse_bn=True),
        'reuse_bn_channelslast': dict(algebra, projective_ruler=True, fuse_bn=True, channels_last=True),
    }
    models = {}
    for name, setting in options.items():
        inference = {key: value for key, value in setting.items() if key not in ('fuse_bn', 'channels_last')}
        model, _ = load_model(weights, 'cuda:0', inference=dict(spatial_layout=True, **inference))
        if setting.get('fuse_bn'):
            fuse_backbone(model)
        if setting.get('channels_last'):
            model.matching.backbone.to(memory_format=torch.channels_last)
            model.matching.backbone.conv1.register_forward_pre_hook(
                lambda module, args: (args[0].contiguous(memory_format=torch.channels_last),))
        models[name] = model
    timings = {name: [] for name in models}
    checks = {name: [] for name in models}
    rng = random.Random(20261101)
    with torch.inference_mode():
        for frame, item in zip(indices, inputs):
            for model in models.values():
                for _ in range(10):
                    model(equal_pixel_inputs(item), domain, domain['reference_index'])
            torch.cuda.synchronize()
            for _ in range(30):
                order = list(models)
                rng.shuffle(order)
                for name in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    current = models[name](equal_pixel_inputs(item), domain, domain['reference_index'])
                    torch.cuda.synchronize()
                    timings[name].append((time.perf_counter() - start) * 1000)
                    del current
            reference = models['baseline'](equal_pixel_inputs(item), domain, domain['reference_index'])
            for name, model in models.items():
                current = model(equal_pixel_inputs(item), domain, domain['reference_index'])
                checks[name].append({'frame': frame, **differences(current, reference)})
                del current
            del reference
    report = {'completed': True, 'load_targets': False, 'indices': indices, 'split': 'val',
              'checkpoint': weights, 'gpu': torch.cuda.get_device_name(), 'options': options,
              'precision': 'FP32/TF32off', 'input_mode': 'equal_pixels',
              'warmup': 10, 'repetitions_per_frame': 30, 'models_resident': len(models),
              'timings': {name: {'median_ms': statistics.median(values), 'samples_ms': values}
                          for name, values in timings.items()}, 'output_differences': checks,
              'notes': '浮点重排可能改变离散选峰；三帧无标签一致性不是精度结论。BN融合只用于eval副本。'}
    write_json(output / 'results.json', report)
    print(json.dumps({name: {'median_ms': statistics.median(timings[name]),
                            'checks': checks[name]} for name in models}, ensure_ascii=False), flush=True)
    del models, inputs
    gc.collect()


if __name__ == '__main__':
    main()
