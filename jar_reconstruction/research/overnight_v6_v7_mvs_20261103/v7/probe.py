"""V7同起点32步短试与速度检查；只访问TRAIN/VAL，非正式性能结论。"""

import copy
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

import torch

from jar.data import collate_frames
from jar.pretraining.comparison import equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset
from jar.pretraining.inference import prepare_inference
from jar.models.position_cache import install_position_cache
from jar.runtime import move
from research.rulermvs_v6_20261101.model import V6Model
from research.rulermvs_v6_20261101.objective import objective
from research.overnight_v6_v7_mvs_20261103.v7.model import V7Model
from tools.smoke_outdoor_datasets import offline_only

OUTPUT = Path('/data/liuzonghao_data/dataset/public_expansion_20261102/v7_refusion_probe_20261103')


def metrics(model, dataset, device):
    rows = []
    model.eval()
    with torch.inference_mode():
        for number, domain in enumerate(dataset.domains):
            sums = {stage: [0., 0] for stage in ('coarse', 'fine', 'final')}
            indices = (0, len(dataset.datasets[number]) // 2, len(dataset.datasets[number]) - 1)
            for index in indices:
                batch = move(collate_frames([dataset[number, index, -1]]), device)
                inputs = equal_pixel_inputs(batch['inputs'])
                out = model(inputs, domain, reference=0)
                truth = batch['targets']['depth'][:, :1]
                valid = batch['targets']['depth_valid'][:, :1] & inputs['image_mask'][:, :1]
                valid &= torch.isfinite(truth) & (truth > 0)
                for stage, stride in (('coarse', 8), ('fine', 4), ('final', 1)):
                    prediction = (out['depth'] if stage == 'final' else out[stage]['depth']) * out['unit_scale']
                    mask = valid[..., ::stride, ::stride]
                    error = (prediction - truth[..., ::stride, ::stride]).abs()
                    sums[stage][0] += float(error[mask].sum())
                    sums[stage][1] += int(mask.sum())
            rows.append(dict(dataset=domain['name'], indices=indices,
                             mae={key: total / count for key, (total, count) in sums.items()}))
    return rows


def main():
    offline_only()
    torch.set_num_threads(2)
    torch.cuda.set_device(4)
    device = torch.device('cuda', 4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = json.loads((PROJECT.parent / 'data/public_expansion_20261102/train_v6_expanded.json').read_text())
    config['augmentation'] = False
    torch.manual_seed(20261103)
    saved = torch.load(config['initial_model_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    base_config = copy.deepcopy(saved['config'])
    original = V6Model(base_config, initialize=False)
    original.load_state_dict(saved['model'], strict=True)
    updated_config = dict(base_config, architecture='rulermvs_v7_refusion',
                          refusion=dict(hidden=32, heads=4, bands=8))
    updated = V7Model(updated_config, initialize=False)
    updated.load_v6_state(saved['model'])
    models = {'v6_control': original, 'v7_refusion': updated}
    for model in models.values():
        model.to(device).eval().requires_grad_(False)
        install_position_cache(model.foundation.model)
        prepare_inference(model, spatial_layout=True, factorized_fusion=True,
                          hoist_reference=True, compact_pair=True, projective_ruler=True)
    train, val = MixedDataset(config), MixedDataset(config, 'val')
    domain = train.domains[0]
    batch = move(collate_frames([train[0, 0, -1]]), device)
    inputs = equal_pixel_inputs(batch['inputs'])
    report = dict(completed=False, gpu=torch.cuda.get_device_name(), test_accessed=False,
                  weights=saved['config'].get('initial_model_checkpoint'),
                  new_parameters=sum(p.numel() for p in updated.decoder.refusion.parameters()),
                  protocol='共同V6权重起点、各32步B1、编码冻结、七域TRAIN/VAL短试；不等于从通用初始化的正式实验',
                  precision='FP32/TF32off', timings={}, initial_identity={}, training=[])
    with torch.inference_mode():
        before = original(inputs, domain)
        after = updated(inputs, domain)
        report['initial_identity'] = {key: bool(torch.equal(before[key], after[key]))
                                      for key in ('depth', 'valid', 'confidence')}
        for model in models.values():
            for _ in range(10):
                model(inputs, domain)
        torch.cuda.synchronize()
        times = {name: [] for name in models}
        for repeat in range(30):
            order = list(models) if repeat % 2 == 0 else list(reversed(models))
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                models[name](inputs, domain)
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - start) * 1000)
        report['timings'] = {name: dict(median_ms=statistics.median(values), samples_ms=values)
                             for name, values in times.items()}
    write_json(OUTPUT / 'progress.json', report)
    print('speed', {key: value['median_ms'] for key, value in report['timings'].items()}, flush=True)
    report['val_initial'] = metrics(original, val, device)
    optimizers = {}
    for name, model in models.items():
        model.decoder.requires_grad_(True)
        ordinary = [p for key, p in model.decoder.named_parameters() if not key.startswith('refusion.')]
        groups = [dict(params=ordinary, lr=1e-5)]
        if name == 'v7_refusion':
            groups.append(dict(params=model.decoder.refusion.parameters(), lr=1e-4))
        optimizers[name] = torch.optim.AdamW(groups, weight_decay=1e-4)
    for step in range(32):
        number = step % len(train.domains)
        index = (step * 137 + 11) % len(train.datasets[number])
        batch = move(collate_frames([train[number, index, -1]]), device)
        inputs = equal_pixel_inputs(batch['inputs'])
        domain = train.domains[number]
        row = dict(step=step, domain=domain['name'], index=index)
        for name, model in models.items():
            optimizers[name].zero_grad(set_to_none=True)
            model.decoder.train()
            output = model(inputs, domain)
            loss = objective(output, inputs, batch['targets'], domain, config['loss'])['total']
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'{name} loss非有限')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1., error_if_nonfinite=True)
            optimizers[name].step()
            row[name] = dict(loss=float(loss.detach()), gradient_norm=float(norm))
            del output, loss
        report['training'].append(row)
        if step % 8 == 0:
            print('step', row, flush=True)
            write_json(OUTPUT / 'progress.json', report)
    report['val_after'] = {name: metrics(model, val, device) for name, model in models.items()}
    report['completed'] = True
    write_json(OUTPUT / 'results.json', report)
    print('completed', flush=True)


if __name__ == '__main__':
    main()
