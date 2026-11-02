"""RulerMVS-V2再训练：全覆盖弱域重放、候选实际风险、分阶段共同适配。"""

import argparse
import functools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.losses import depth_objective
from jar.pretraining.model import PretrainingModel
from jar.pretraining.runner import run
from research.near_mid_20261027.training import SurfaceHooks, distance_masks
from research.v2_training_20261028.objectives import excess_error, fine_actual_regret
from research.v2_training_20261028.sampling import RebalancedFullPassSampler


class V2TrainingModel(PretrainingModel):
    """只控制训练参数；保持V2原共享头、原凸上采样及state_dict。"""

    def __init__(self, config, initialize=True):
        if config['model'].get('decoder_kind', 'calibrated') != 'calibrated' \
                or config['model'].get('independent_fine_heads', False):
            raise ValueError('本轮必须是原V2共享头和凸上采样结构')
        super().__init__(config, initialize)

    def set_stage(self, number):
        stage = next(s for s in self.config['stages'] if s['number'] == number)
        self.active_stage = stage
        self.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if name.startswith(tuple(stage['trainable_prefixes'])):
                parameter.requires_grad_(True)
        count = stage.get('foundation_blocks', 0)
        if count:
            for block in self.foundation.model.blocks[-count:]:
                block.requires_grad_(True)
        # 共享norm作用于四层输出，不能误称只影响已解冻的末尾block。
        self.foundation.model.norm.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.foundation.parameters()):
            self.foundation.eval()
        if not any(p.requires_grad for p in self.matching.parameters()):
            self.matching.eval()
        return self


def v2_optimizer(model, stage, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('foundation.'):
            kind = 'foundation'
        elif name.startswith('matching.backbone.'):
            kind = 'matching'
        elif name.startswith('matching.'):
            kind = 'fpn'
        else:
            kind = 'decoder'
        decay = 0. if parameter.ndim <= 1 or name.endswith('bias') else config['train']['weight_decay']
        groups.setdefault((kind, decay), []).append(parameter)
    return torch.optim.AdamW([
        {'params': values, 'lr': stage[kind + '_lr'], 'initial_lr': stage[kind + '_lr'],
         'weight_decay': decay, 'group_name': kind}
        for (kind, decay), values in groups.items()
    ])


class V2Hooks(SurfaceHooks):
    """复用已验逐域VAL统计；替换训练目标和成熟模型的全局守门。"""

    optimizer_factory = staticmethod(v2_optimizer)

    def __init__(self, config):
        super().__init__(config)
        self.model_factory = self.build_model
        self.full_sampler_factory = functools.partial(
            RebalancedFullPassSampler,
            domain_multipliers=[d.get('train_multiplier', 1.) for d in config['datasets']],
            domain_schedule=config['train'].get('domain_schedule', 'shuffle'))
        self.sampler_factory = self.make_sampler

    def make_sampler(self, *args, **kwargs):
        sampler = self.full_sampler_factory(*args, **kwargs)
        limit = self.config['train'].get('development_updates_per_epoch')
        if limit:
            # 仅短试使用：取固定调度前缀，不声称本轮覆盖全部TRAIN。
            sampler.updates = sampler.updates[:limit]
        return sampler

    def build_model(self, config, initialize=True):
        self.student = V2TrainingModel(config, initialize)
        return self.student

    def prepare_batch(self, batch, domain):
        # 只扰动TRAIN候选搜索范围，不改K/T、标签、单位或推理范围。
        amplitude = self.student.active_stage.get('range_log_jitter', 0.)
        probability = self.student.active_stage.get('range_jitter_probability', 0.)
        if amplitude and torch.rand(()) < probability:
            factor = math.exp(float(torch.empty(()).uniform_(-amplitude, amplitude)))
            domain = dict(domain, near=domain['near'] * factor, far=domain['far'] * factor,
                          nominal_near=domain['near'], nominal_far=domain['far'])
        return batch, domain

    def objective(self, prediction, inputs, targets, domain, settings, correspondence_weight=0.):
        terms = depth_objective(prediction, inputs, targets, domain, settings, correspondence_weight)
        risk = fine_actual_regret(prediction, inputs, targets, domain)
        if settings.get('actual_fine_risk', False):
            # 原函数已经包含0.1*old_risk，替换这一个项，不叠加两份风险。
            terms['total'] = terms['total'] + .1 * (risk['new_risk'] - risk['old_risk'])
        terms.update(fine_actual_risk=risk['new_risk'], fine_actual_regret=risk['regret'])
        weight = settings.get('excess_error_weight', 0.)
        if weight:
            nominal = dict(domain, near=domain.get('nominal_near', domain['near']),
                           far=domain.get('nominal_far', domain['far']))
            # 始终由冻结V2独立编码，不能复用已更新学生的特征冒充teacher。
            with torch.no_grad():
                teacher = self.get_teacher(inputs['rgb'].device)(inputs, nominal)['depth']
            scale = domain['depth_scale']
            truth = targets['depth'][:, :1].float() / scale
            valid = targets['depth_valid'][:, :1] & inputs['image_mask'][:, :1]
            valid &= inputs['view_mask'][:, :1, None, None] & torch.isfinite(truth) & (truth > 0)
            if domain.get('eval_range'):
                low, high = domain['eval_range']
                valid &= (truth * scale >= low) & (truth * scale <= high)
            masks = distance_masks(truth * scale, valid, domain)
            values = torch.stack([excess_error(prediction['depth'], teacher, truth, mask) for mask in masks])
            active = torch.stack([mask.any() for mask in masks]).to(values.dtype)
            penalty = values.sum() / active.sum().clamp_min(1.)
            terms['excess_error'] = penalty
            terms['total'] = terms['total'] + weight * penalty
        return terms

    @torch.no_grad()
    def evaluate(self, model, dataset, config, runtime, limit=None, split='val'):
        result = super().evaluate(model, dataset, config, runtime, limit, split)
        settings = config['evaluation']
        failures, risks, near_mid = [], [], []
        for row in result['domains']:
            changes = {}
            for region in ('full', 'near', 'mid', 'far'):
                current, baseline = row['own'][region], row['reference'][region]
                if not current['pixels']:
                    continue
                changes[region] = {}
                for metric in ('mae', 'rmse', 'abs_rel'):
                    relative = current[metric] / max(baseline[metric], 1e-12) - 1
                    changes[region][metric] = relative
                    tolerance = settings['guard_relative_tolerance'][region]
                    if relative > tolerance and current[metric] > baseline[metric] + 1e-6:
                        failures.append(f"{row['dataset']}/{region}/{metric}")
                    if region == 'full':
                        risks.append(relative)
                    elif region in ('near', 'mid'):
                        near_mid.append(relative)
            row['relative_changes'] = changes
        result.update(macro_relative_change=sum(risks) / len(risks), guard_failures=failures,
                      selection_rule='域等权full MAE/RMSE/AbsRel相对原V2改善，且各距离段守门',
                      guard_relative_tolerance=settings['guard_relative_tolerance'])
        return result

    @staticmethod
    def selection_score(result):
        value = result['macro_relative_change']
        return value if value < 0 and not result['guard_failures'] else math.inf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default='')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run(config, args.resume or None, hooks=V2Hooks(config))


if __name__ == '__main__':
    main()
