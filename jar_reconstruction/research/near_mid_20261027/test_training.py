"""重放、路由监督与守门逻辑的CPU测试；不加载权重或真实数据。"""

import math
import copy
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.near_mid_20261027.training import ReplaySampler, SurfaceHooks, distance_limits, distance_masks, render_losses, surface_optimizer
from jar.pretraining import runner
from jar.runtime import capture_rng


def test_replay():
    options = dict(lengths=[5, 4], epoch=3, world=2, batch_size=1, accumulation=2,
                   seed=7, repeat_counts=[1, 2])
    samplers = [ReplaySampler(rank=rank, **options) for rank in range(2)]
    assert samplers[0].updates == samplers[1].updates
    seen = [key for sampler in samplers for batch in sampler for key in batch]
    assert {(d, i) for d, i, _ in seen} == {(d, i) for d, n in enumerate([5, 4]) for i in range(n)}
    assert {epoch for d, _, epoch in seen if d == 0} == {3}
    assert {epoch for d, _, epoch in seen if d == 1} == {6, 7}
    restored = ReplaySampler(rank=0, start_update=1, **options)
    assert list(restored) == list(samplers[0])[2:]


def test_routing():
    depth = torch.tensor([1., 2., 5.]).view(1, 3, 1, 1).requires_grad_()
    logits = torch.zeros_like(depth, requires_grad=True)
    truth = torch.tensor([[[[2.2]]]])
    valid = torch.ones_like(truth, dtype=torch.bool)
    terms = render_losses({'plane_depth': depth, 'logits': logits,
                           'valid': torch.ones_like(depth, dtype=torch.bool)}, truth, valid)
    sum(terms.values()).backward()
    assert torch.isfinite(depth.grad).all() and depth.grad.abs().sum() > 0
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert logits.grad[0, 1, 0, 0] < 0
    rejected = render_losses({'plane_depth': depth, 'logits': logits,
                              'valid': torch.zeros_like(depth, dtype=torch.bool)}, truth, valid)
    assert all(torch.isfinite(value) and value == 0 for value in rejected.values())


def test_guard_and_units():
    assert distance_limits({'geometry_units': 'meters'}) == (10., 40.)
    assert distance_limits({'geometry_units': 'scene_unit', 'depth_scale': 2.}) == (10., 40.)
    assert SurfaceHooks.selection_score({'near_mid_relative_change': -.1, 'far_guard_failures': []}) == -.1
    assert math.isinf(SurfaceHooks.selection_score(
        {'near_mid_relative_change': -.1, 'far_guard_failures': ['domain/far/mae']}))
    assert math.isinf(SurfaceHooks.selection_score(
        {'near_mid_relative_change': .1, 'far_guard_failures': []}))
    truth = torch.tensor([1., 9.9, 10., 39.9, 40., 100.])
    masks = distance_masks(truth, torch.ones_like(truth, dtype=torch.bool), {'geometry_units': 'meters'})
    assert [mask.nonzero().flatten().tolist() for mask in masks] == [[0, 1], [2, 3], [4, 5]]
    assert torch.stack(masks).sum(0).eq(1).all()


def test_stage_boundary_resume_rng():
    """实际走runner的optimizer=None恢复路径，模型重建不得吞掉已保存RNG。"""
    runtime = SimpleNamespace(device=torch.device('cpu'), rank=0, world_size=1,
                              distributed=False, is_main=True)

    class ToyModel(torch.nn.Linear):
        architecture = 'test'

        def __init__(self, config, initialize=False):
            super().__init__(2, 1)

    with tempfile.TemporaryDirectory() as directory:
        config = {'output': directory, 'seed': 321, 'datasets': [], 'stages': [],
                  'train': {'minimum_free_gib': 0, 'allow_tf32': False}, 'evaluation': {}}
        model = ToyModel(config)
        rng = capture_rng(runtime)
        resume = Path(directory) / 'resume.pt'
        torch.save({'config': config, 'world_size': 1, 'model': model.state_dict(),
                    'optimizer': None, 'state': {}, 'rng_by_rank': [rng]}, resume)
        hooks = SimpleNamespace(model_factory=ToyModel, objective=None,
                                sampler_factory=None, evaluate=None, selection_score=None)
        with patch.object(runner, 'MixedDataset', return_value=[]):
            runner._run(config, resume, runtime, hooks)
        assert torch.equal(torch.get_rng_state(), rng['torch'])
        assert random.getstate() == rng['python']


def test_optimizer_groups_and_resume():
    model = torch.nn.Module()
    model.foundation = torch.nn.Linear(2, 2)
    model.matching = torch.nn.ModuleDict({'backbone': torch.nn.Linear(2, 2),
                                         'lateral': torch.nn.Linear(2, 2)})
    model.decoder = torch.nn.ModuleDict({'surface_readout': torch.nn.Linear(2, 2),
                                        'fine_score': torch.nn.Linear(2, 1)})
    model.decoder.fine_score.bias.requires_grad_(False)
    stage = {'readout_lr': 1e-4, 'new_lr': 1e-5, 'matching_lr': 2e-6, 'foundation_lr': 2e-7}
    config = {'train': {'weight_decay': 1e-4}}
    optimizer = surface_optimizer(model, stage, config)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters() if p.requires_grad}
    for group in optimizer.param_groups:
        assert group['lr'] == stage[group['group_name'] + '_lr']
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    resumed_model = copy.deepcopy(model)
    resumed_optimizer = surface_optimizer(resumed_model, stage, config)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    assert resumed_optimizer.state_dict()['param_groups'] == optimizer.state_dict()['param_groups']
    assert all(value['step'] == 1 for value in resumed_optimizer.state.values())
    # control的旧输出头必须同样训练，使用同一readout学习率。
    model.decoder['upsample_mask'] = model.decoder.pop('surface_readout')
    control = surface_optimizer(model, stage, config)
    assert sum(p.numel() for g in control.param_groups if g['group_name'] == 'readout' for p in g['params']) == 6


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_replay()
    test_routing()
    test_guard_and_units()
    test_stage_boundary_resume_rng()
    test_optimizer_groups_and_resume()
    print('5 CPU training protocol tests passed')
