"""中近距离表面读出研究：复用公共DDP训练流程，不改默认训练入口。"""

import argparse
import functools
import json
import math
from pathlib import Path
import random
import sys

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.data import collate_mixed
from jar.pretraining.losses import depth_objective, masked_mean
from jar.pretraining.model import PretrainingModel
from jar.pretraining.runner import optimizer_for, run
from jar.runtime import move


def distance_limits(domain):
    """米制用10/40m；未知绝对单位用TRAIN中位深度的0.5/2倍。"""
    if domain['geometry_units'] == 'meters':
        return 10., 40.
    median = 10. * domain['depth_scale']
    return .5 * median, 2. * median


def distance_masks(truth, valid, domain):
    """与离线诊断统一为[0, near)、[near, far)、[far, +∞)。"""
    near, far = distance_limits(domain)
    return valid & (truth < near), valid & (truth >= near) & (truth < far), valid & (truth >= far)


class ReplaySampler(Sampler):
    """每域全覆盖；可重放小域，重放同时轮换参考和增强种子。"""

    def __init__(self, lengths, epoch, rank, world, batch_size, accumulation,
                 seed, start_update=0, repeat_counts=None):
        self.rank, self.world = rank, world
        self.batch_size, self.accumulation = batch_size, accumulation
        self.start_update = start_update
        repeats = repeat_counts or [1] * len(lengths)
        if len(repeats) != len(lengths) or any(int(n) != n or n < 1 for n in repeats):
            raise ValueError('每个域需要正整数repeat_count')
        rng = random.Random(seed + epoch)
        global_batch = world * batch_size * accumulation
        self.updates, self.repeats = [], {}
        for domain, (length, cycles) in enumerate(zip(lengths, repeats)):
            cycles = int(cycles)
            entries = []
            for cycle in range(cycles):
                indices = list(range(length))
                rng.shuffle(indices)
                visit_epoch = epoch * cycles + cycle
                entries.extend((index, visit_epoch) for index in indices)
            padded = math.ceil(len(entries) / global_batch) * global_batch
            self.repeats[domain] = padded - len(entries)
            entries += [entries[i % len(entries)] for i in range(padded - len(entries))]
            self.updates.extend((domain, entries[i:i + global_batch])
                                for i in range(0, padded, global_batch))
        rng.shuffle(self.updates)

    def __len__(self):
        return (len(self.updates) - self.start_update) * self.accumulation

    def __iter__(self):
        for domain, entries in self.updates[self.start_update:]:
            for micro in range(self.accumulation):
                start = (micro * self.world + self.rank) * self.batch_size
                yield [(domain, index, epoch)
                       for index, epoch in entries[start:start + self.batch_size]]


class SurfaceTrainingModel(PretrainingModel):
    """只增加训练期编码复用和显式冻结；模型state_dict中没有teacher。"""

    def set_stage(self, number):
        stage = next(s for s in self.config['stages'] if s['number'] == number)
        self.requires_grad_(False)
        prefixes = tuple(stage['trainable_prefixes'])
        for name, parameter in self.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad_(True)
        foundation_stage = stage.get('foundation_stage', 1)
        self.foundation.set_stage(foundation_stage)
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError('当前阶段没有可训练参数')

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.foundation.parameters()):
            self.foundation.eval()
        if not any(p.requires_grad for p in self.matching.parameters()):
            self.matching.eval()
        return self

    def forward(self, inputs, domain, reference=0):
        scale = float(domain['depth_scale'])
        self.decoder.set_range(domain['near'] / scale, domain['far'] / scale)
        work = self.encode(inputs, domain, reference)
        prediction = self.decoder(work)
        prediction['pyramid'] = work['pyramid'] if self.training else None
        prediction['unit_scale'] = scale
        if self.training and not any(p.requires_grad for p in self.foundation.parameters()) \
                and not any(p.requires_grad for p in self.matching.parameters()):
            prediction['_teacher_work'] = work
        return prediction


def render_losses(render, truth, valid, temperature=.05):
    """GT只生成停止梯度的归属标签；实际输出仍选择单个表面。"""
    if temperature <= 0:
        raise ValueError('render_temperature需要大于0')
    raw_depths = render['plane_depth'].float()
    allowed = render['valid'].bool() & torch.isfinite(raw_depths) & (raw_depths > 0)
    depths = torch.where(allowed, raw_depths, 1.).clamp_min(1e-6)
    logits = render['logits'].float().masked_fill(~allowed, -1e4)
    active = valid & allowed.any(1, keepdim=True)
    error = (depths.log() - truth.clamp_min(1e-6).log()).abs()
    quality = (-error.detach() / temperature).masked_fill(~allowed, -1e4).softmax(1)
    probability = logits.softmax(1)
    classification = masked_mean(-(quality * logits.log_softmax(1)).sum(1, keepdim=True), active)
    location = masked_mean((quality * error).sum(1, keepdim=True), active)
    # 这是各单表面的期望误差，不是不同表面深度的加权平均输出。
    risk = masked_mean((probability * error.clamp_max(4)).sum(1, keepdim=True), active)
    return {'render_ce': classification, 'render_location': location, 'render_risk': risk}


def surface_optimizer(model, stage, config):
    """新读出保持足够学习率，已训练的匹配/几何/基础编码器保守更新。"""
    if 'readout_lr' not in stage:
        return optimizer_for(model, stage, config)
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(('decoder.surface_readout.', 'decoder.upsample_mask.')):
            kind = 'readout'
        elif name.startswith('foundation.'):
            kind = 'foundation'
        elif name.startswith('matching.backbone.'):
            kind = 'matching'
        else:
            kind = 'new'
        decay = 0. if parameter.ndim <= 1 or name.endswith('bias') else config['train']['weight_decay']
        groups.setdefault((kind, decay), []).append(parameter)
    return torch.optim.AdamW([
        {'params': parameters, 'lr': stage[kind + '_lr'], 'initial_lr': stage[kind + '_lr'],
         'weight_decay': decay, 'group_name': kind}
        for (kind, decay), parameters in groups.items()
    ])


class SurfaceHooks:
    """五个轻量策略供生产runner调用；teacher不进入student检查点。"""

    model_factory = SurfaceTrainingModel
    optimizer_factory = staticmethod(surface_optimizer)

    def __init__(self, config):
        self.config = config
        if config.get('transfer') or config.get('zero_shot_before_training') \
                or config.get('evaluation', {}).get('test_after_training'):
            raise ValueError('本研究入口不自动迁移或读取TEST')
        if config['train']['precision'] != 'fp32' or config['train'].get('allow_tf32', True):
            raise ValueError('当前训练协议固定FP32、关闭TF32；混合精度需独立验证')
        self.sampler_factory = functools.partial(
            ReplaySampler, repeat_counts=[d.get('train_repeats', 1) for d in config['datasets']])
        self.teacher = None
        self.baseline_cache = {}

    def get_teacher(self, device):
        if self.teacher is None:
            # teacher不存进训练checkpoint；恢复时重建不得消耗student的已恢复RNG。
            devices = [device.index] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                record = torch.load(self.config['teacher_checkpoint'], map_location='cpu',
                                    weights_only=False, mmap=True)
                teacher = PretrainingModel(record['config'], initialize=False)
                teacher.load_state_dict(record['model'], strict=True)
                self.teacher = teacher.to(device).eval().requires_grad_(False)
        return self.teacher

    @torch.no_grad()
    def teacher_prediction(self, inputs, domain, work=None):
        teacher = self.get_teacher(inputs['rgb'].device)
        scale = domain['depth_scale']
        teacher.decoder.set_range(domain['near'] / scale, domain['far'] / scale)
        with torch.autocast(device_type=inputs['rgb'].device.type, enabled=False):
            if work is None:
                return teacher(inputs, domain)['depth'].float()
            return teacher.decoder(work)['depth'].float()

    def objective(self, prediction, inputs, targets, domain, settings, correspondence_weight=0.):
        metrics = depth_objective(prediction, inputs, targets, domain, settings, correspondence_weight)
        scale = prediction['unit_scale']
        truth = targets['depth'][:, :1].float() / scale
        valid = targets['depth_valid'][:, :1] & inputs['image_mask'][:, :1]
        valid &= inputs['view_mask'][:, :1, None, None] & torch.isfinite(truth) & (truth > 0)
        safe_truth = torch.where(valid, truth, 1.)
        if domain.get('eval_range'):
            valid &= (truth * scale >= domain['eval_range'][0]) & (truth * scale <= domain['eval_range'][1])
        if 'render' in prediction:
            terms = render_losses(prediction['render'], safe_truth, valid,
                                  settings.get('render_temperature', .05))
            metrics.update(terms)
            metrics['total'] = metrics['total'] + settings.get('render_ce_weight', .15) * terms['render_ce']
            metrics['total'] = metrics['total'] + settings.get('render_location_weight', .1) * terms['render_location']
            metrics['total'] = metrics['total'] + settings.get('render_risk_weight', .05) * terms['render_risk']
        work = prediction.pop('_teacher_work', None)
        teacher = self.teacher_prediction(inputs, domain, work)
        estimate = prediction['depth'].float().clamp_min(1e-6)
        teacher = teacher.clamp_min(1e-6)
        far = distance_masks(truth * scale, valid, domain)[2]
        # 只有teacher在当前TRAIN标签上至少一样好时才保持，避免强迫复制更差的预测。
        better = (teacher.log() - safe_truth.log()).abs() <= \
                 (estimate.detach().log() - safe_truth.log()).abs()
        preserve = far & better
        retention = masked_mean((estimate.log() - teacher.log()).abs(), preserve)
        metrics.update(far_retention=retention, far_retained_pixels=preserve.sum())
        metrics['total'] = metrics['total'] + settings.get('far_retention_weight', .2) * retention
        return metrics

    @staticmethod
    def selection_score(result):
        score = result['near_mid_relative_change']
        return score if not result['far_guard_failures'] and score < 0 else float('inf')

    @torch.no_grad()
    def evaluate(self, model, dataset, config, runtime, limit=None, split='val'):
        if split != 'val':
            raise ValueError('本研究仅使用VAL选模，不读取TEST')
        model.eval()
        teacher = self.get_teacher(runtime.device)
        records, failures, improvements, near_mid_regressions = [], [], [], []
        for domain_index, domain in enumerate(dataset.domains):
            length = len(dataset.datasets[domain_index])
            indices = list(range(length)) if limit is None else \
                torch.linspace(0, length - 1, min(length, limit)).long().tolist()
            keys = [(domain_index, index, -1) for index in indices[runtime.rank::runtime.world_size]]
            loader = DataLoader(dataset, batch_size=1, sampler=keys, collate_fn=collate_mixed,
                                num_workers=0, pin_memory=True,
                                generator=torch.Generator().manual_seed(config['seed']))
            cache_key = (domain['name'], tuple(indices))
            cached_baseline = self.baseline_cache.get(cache_key)
            sums = torch.zeros(2, 4, 5, dtype=torch.float64, device=runtime.device)
            for batch in loader:
                batch = move(batch, runtime.device)
                inputs, targets = batch['inputs'], batch['targets']
                with torch.autocast(device_type=runtime.device.type, enabled=False):
                    prediction = model(inputs, domain)
                    own = prediction['depth'].float() * domain['depth_scale']
                    teacher_result = None if cached_baseline is not None else teacher(inputs, domain)
                    baseline = None if teacher_result is None else \
                        teacher_result['depth'].float() * domain['depth_scale']
                truth = targets['depth'][:, :1].float()
                valid = targets['depth_valid'][:, :1] & inputs['image_mask'][:, :1]
                valid &= inputs['view_mask'][:, :1, None, None] & torch.isfinite(truth) & (truth > 0)
                if domain.get('eval_range'):
                    valid &= (truth >= domain['eval_range'][0]) & (truth <= domain['eval_range'][1])
                masks = (valid, *distance_masks(truth, valid, domain))
                for method, estimate in enumerate((own, baseline)):
                    if estimate is None:
                        continue
                    if not torch.isfinite(estimate[valid]).all():
                        raise RuntimeError(f"{domain['name']}验证预测非有限")
                    prediction_valid = prediction['valid'] if method == 0 else teacher_result['valid']
                    for region, mask in enumerate(masks):
                        error = (estimate - truth)[mask].double()
                        relative = error.abs() / truth[mask].double()
                        values = torch.stack((mask.sum().double(), error.abs().sum(),
                                              error.square().sum(), relative.sum(),
                                              prediction_valid[mask].sum().double()))
                        sums[method, region] += values
            if runtime.distributed:
                dist.all_reduce(sums)
            if cached_baseline is None:
                self.baseline_cache[cache_key] = sums[1].cpu()
            else:
                sums[1] = cached_baseline.to(runtime.device)
            methods = []
            for method in range(2):
                regions = {}
                for region, name in enumerate(('full', 'near', 'mid', 'far')):
                    count, absolute, squared, relative, covered = sums[method, region].tolist()
                    regions[name] = {'pixels': int(count), 'mae': absolute / count if count else None,
                                     'rmse': math.sqrt(squared / count) if count else None,
                                     'abs_rel': relative / count if count else None,
                                     'coverage': covered / count if count else None}
                methods.append(regions)
            own, reference = methods
            if not own['full']['pixels']:
                raise RuntimeError(f"{domain['name']}没有验证GT")
            domain_improvements = []
            for region in ('near', 'mid'):
                if own[region]['pixels']:
                    for metric in ('mae', 'rmse', 'abs_rel'):
                        change = own[region][metric] / max(reference[region][metric], 1e-12) - 1
                        domain_improvements.append(change)
                        if own[region][metric] > reference[region][metric] + 1e-6:
                            near_mid_regressions.append(domain['name'] + '/' + region + '/' + metric)
            if domain_improvements:
                improvements.append(sum(domain_improvements) / len(domain_improvements))
            if own['far']['pixels']:
                for metric in ('mae', 'rmse', 'abs_rel'):
                    if own['far'][metric] > reference['far'][metric] + 1e-6:
                        failures.append(domain['name'] + '/far/' + metric)
            records.append({'dataset': domain['name'], 'units': domain['geometry_units'],
                            'frames': len(indices), 'distance_limits': distance_limits(domain),
                            'own': own, 'reference': reference})
        model.train()
        return {'split': split, 'full': limit is None, 'precision': 'fp32',
                'domains': records, 'macro_abs_rel': sum(r['own']['full']['abs_rel'] for r in records) / len(records),
                'near_mid_relative_change': sum(improvements) / max(len(improvements), 1),
                'near_mid_regressions': near_mid_regressions,
                'far_guard_failures': failures, 'far_guard_atol': 1e-6,
                'teacher_checkpoint': config['teacher_checkpoint']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default='')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run(config, args.resume or None, hooks=SurfaceHooks(config))


if __name__ == '__main__':
    main()
