"""固定七域TRAIN/DEV的2×2研究：组内读出 × 有界公制训练项。"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.losses import depth_objective, masked_mean
from jar.runtime import move
from research.ruler_joint_20261027.model import make_decoder
from tools.probe_public_stages import error_statistics, sum_records, summarize
from tools.refine_public_candidates import FeatureDataset, join, save_record


def metric_auxiliary(prediction, row):
    """公共工作坐标的TRAIN中位深度=10；不从当前batch或测试集估计尺度。"""
    truth = row['targets']['depth'][:, :1].float() / prediction['unit_scale']
    valid = row['targets']['depth_valid'][:, :1] & row['inputs']['image_mask'][:, :1]
    valid &= row['inputs']['view_mask'][:, :1, None, None]
    valid &= torch.isfinite(truth) & (truth > 0)
    target = torch.where(valid, truth, 1.)
    error = F.smooth_l1_loss(prediction['depth'].float() / 10., target / 10.,
                             reduction='none', beta=1.)
    return masked_mean(error, valid)


@torch.no_grad()
def evaluate(model, index, device, split='development'):
    model.eval()
    results = {}
    for domain in index['domains']:
        paths = index['splits'][split].get(domain['name'], [])
        if not paths:
            continue
        scale = domain['depth_scale']
        model.set_range(domain['near'] / scale, domain['far'] / scale)
        records = []
        for path in paths:
            row = move(torch.load(path, map_location='cpu', weights_only=False), device)
            prediction = model(row['work'])
            truth = row['targets']['depth'].float()
            valid = row['targets']['depth_valid'] & row['inputs']['image_mask'][:, :1]
            valid &= row['inputs']['view_mask'][:, :1, None, None]
            valid &= torch.isfinite(truth) & (truth > 0)
            if not bool(torch.isfinite(prediction['depth']).all()):
                raise ValueError('预测非有限')
            far = valid & (truth >= torch.quantile(truth[valid], .75))
            masks = {'full': valid, 'far25': far, 'near75': valid & ~far}
            if domain['geometry_units'] == 'meters':
                masks['over80m'] = valid & (truth > 80)
            estimate = prediction['depth'] * scale
            record = {}
            for name, mask in masks.items():
                record[name] = error_statistics(estimate, truth, mask)
                record[name]['signed_relative_sum'] = float(((estimate - truth) / truth.clamp_min(1e-6))[mask].double().sum())
            records.append(record)
        results[domain['name']] = summarize(sum_records(records))
    results['macro_abs_rel'] = sum(values['full']['abs_rel'] for values in results.values()) / len(results)
    return results


def compare(result, baseline):
    """逐域、逐指标相对变化等权；避免不同单位或异常域主导宏平均。"""
    changes = {}
    failures = []
    for name in result:
        if name == 'macro_abs_rel':
            continue
        changes[name] = {}
        for region in ('full', 'far25'):
            for metric in ('mae', 'rmse', 'abs_rel'):
                ratio = result[name][region][metric] / max(baseline[name][region][metric], 1e-12) - 1
                changes[name][region + '/' + metric] = ratio
                if ratio > (.02 if region == 'full' else .03):
                    failures.append(name + '/' + region + '/' + metric)
    score = sum(row['full/' + metric] for row in changes.values()
                for metric in ('mae', 'rmse', 'abs_rel')) / (3 * len(changes))
    return {'normalized_risk_change': score, 'failures': failures,
            'per_domain_relative_change': changes,
            'passes_development_gate': score < 0 and not failures}


def run(plan):
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(plan['device'])
    torch.cuda.set_device(device)
    output = Path(plan['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('研究目录非空，禁止覆盖')
    output.mkdir(parents=True)
    index = json.loads((Path(plan['cache']) / 'index.json').read_text())
    if index['checkpoint_sha256'] != digest(plan['checkpoint']):
        raise ValueError('冻结编码器缓存不匹配')
    saved = torch.load(plan['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    decoder, settings = make_decoder(saved, plan, device)
    decoder.requires_grad_(False)
    names = list(plan.get('trainable_modules', ['fine_inherit', 'fine_score', 'fine_offset', 'upsample_mask']))
    if plan.get('relation') and 'trainable_modules' not in plan:
        names.append('relation')
    for name in names:
        getattr(decoder, name).requires_grad_(True)
    groups = []
    for name in names:
        groups.append({'params': list(getattr(decoder, name).parameters()),
                       'lr': plan['relation_lr'] if name == 'relation' else plan['lr']})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    write_json(output / 'plan.json', plan)
    write_json(output / 'audit.json', {
        'settings': settings, 'test_accessed': False, 'precision': 'FP32 TF32 off',
        'checkpoint_sha256': index['checkpoint_sha256'],
        'cache_index_sha256': digest(Path(plan['cache']) / 'index.json'),
        'sources': {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), Path(__file__).with_name('model.py'), ROOT / 'jar/pretraining/losses.py')},
        'trainable_parameters': sum(p.numel() for p in decoder.parameters() if p.requires_grad),
        'frozen_coarse_and_encoder': True, 'trainable_modules': names,
    })
    initial = evaluate(decoder, index, device)
    write_json(output / 'step_0000.json', initial)
    reference = json.loads(Path(plan['baseline_metrics']).read_text()) if plan.get('baseline_metrics') else initial
    initial_comparison = compare(initial, reference)
    write_json(output / 'comparison_0000.json', initial_comparison)
    domains = {domain['name']: domain for domain in index['domains']}
    domain_names = list(domains)
    rng = random.Random(plan['seed'])
    order = (domain_names * math.ceil(plan['steps'] / len(domain_names)))[:plan['steps']]
    rng.shuffle(order)
    loaders = {}
    for name in domain_names:
        required = order.count(name) * plan['batch_size']
        selected = []
        while len(selected) < required:
            cycle = list(index['splits']['train'][name])
            rng.shuffle(cycle)
            selected.extend(cycle)
        loaders[name] = iter(DataLoader(FeatureDataset(selected[:required]),
            batch_size=plan['batch_size'], collate_fn=join, num_workers=1, pin_memory=True))
    best = min(0., initial_comparison['normalized_risk_change']) if initial_comparison['passes_development_gate'] else 0.
    best_step = 0

    def save(name, step):
        save_record({'model': decoder.state_dict(), 'settings': settings,
                     'plan': plan, 'step': step}, output / name)

    save('best.pt', 0)
    started = time.monotonic()
    with (output / 'train.jsonl').open('w') as stream:
        for step, name in enumerate(order, 1):
            domain = domains[name]
            row = move(next(loaders[name]), device)
            scale = domain['depth_scale']
            decoder.set_range(domain['near'] / scale, domain['far'] / scale)
            decoder.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = decoder(row['work'])
            prediction['unit_scale'] = scale
            losses = depth_objective(prediction, row['inputs'], row['targets'], domain, {}, 0.)
            metric = metric_auxiliary(prediction, row)
            loss = losses['total'] + plan['metric_weight'] * metric
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            record = {'step': step, 'dataset': name, 'loss': float(loss.detach()),
                      'metric_auxiliary': float(metric.detach()), 'grad_norm': float(gradient),
                      'elapsed_s': time.monotonic() - started}
            stream.write(json.dumps(record) + '\n')
            if step % 35 == 0:
                stream.flush()
                print(json.dumps(record), flush=True)
            if step % plan['evaluate_every'] == 0 or step == plan['steps']:
                result = evaluate(decoder, index, device)
                comparison = compare(result, reference)
                write_json(output / f'step_{step:04d}.json', result)
                write_json(output / f'comparison_{step:04d}.json', comparison)
                if comparison['passes_development_gate'] and comparison['normalized_risk_change'] < best:
                    best = comparison['normalized_risk_change']
                    best_step = step
                    save('best.pt', step)
                save('last.pt', step)
                if plan.get('save_evaluation_checkpoints', False):
                    save(f'step_{step:04d}.pt', step)
                print(json.dumps({'event': 'validation', 'step': step,
                                  'macro_abs_rel': result['macro_abs_rel'],
                                  'normalized_risk_change': comparison['normalized_risk_change'],
                                  'gate_failures': comparison['failures'], 'best_step': best_step}), flush=True)
    write_json(output / 'complete.json', {'steps': plan['steps'], 'best_step': best_step,
               'best_normalized_risk_change': best, 'elapsed_s': time.monotonic() - started,
               'test_accessed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
