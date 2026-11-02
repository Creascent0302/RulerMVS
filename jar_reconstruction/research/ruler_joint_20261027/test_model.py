"""候选组三点关系的函数保持、几何坐标与梯度契约。"""

import copy

import torch

from jar.pretraining.model import CalibratedDecoder
from research.ruler_joint_20261027.model import FineGroupRelation, JointDecoder


def test_prediction_identity_and_gradients():
    torch.manual_seed(27)
    torch.set_num_threads(2)
    settings = {'channels': 8, 'raw_channels': 8, 'depth_bins': 8,
                'independent_fine_heads': True, 'use_scene': False}
    old = CalibratedDecoder(settings)
    new = JointDecoder(settings)
    new.load_state_dict({**new.state_dict(), **old.state_dict()}, strict=True)
    features = torch.randn(2, 8, 12, 3, 5)
    center = torch.rand(2, 4, 1, 3, 5) * 8 + 10
    z = (center * torch.tensor([.8, 1., 1.2])[None, None, :, None, None]).flatten(1, 2)
    similarity = torch.randn_like(z)
    valid = torch.ones_like(z, dtype=torch.bool)
    valid[1] = False
    source_valid = valid[:, None].expand(-1, 2, -1, -1, -1)
    observation = {'stride': 4, 'source_valid': source_valid, 'view_count': 3,
                   'reference_index': 1}
    args = (features, z, valid, similarity, source_valid.sum(1), z * .9, z * 1.1, observation)
    baseline = old._prediction(*args)
    result = new._prediction(*args)
    for key in baseline:
        assert torch.equal(baseline[key], result[key]), key
    assert torch.isfinite(result['depth']).all()
    loss = (result['candidate_depth'] - 12).square().mean() + result['scores'][0].square().mean()
    loss.backward()
    assert new.relation.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        new.relation.output.weight -= .0001 * new.relation.output.weight.grad
    new.zero_grad(set_to_none=True)
    result = new._prediction(*args)
    (result['candidate_depth'] - 12).square().mean().backward()
    assert new.relation.hidden.weight.grad.abs().sum() > 0


def test_group_isolation_and_units():
    torch.manual_seed(27)
    relation = FineGroupRelation(8)
    torch.nn.init.normal_(relation.output.weight, std=.01)
    features = torch.randn(1, 8, 12, 2, 2)
    z = torch.rand(1, 12, 2, 2) + 1
    similarity = torch.rand_like(z)
    support = torch.ones_like(z)
    baseline = relation(features, z, similarity, support, 3)
    changed = features.clone()
    changed[:, :, 3:6] += 1
    result = relation(changed, z, similarity, support, 3)
    keep = [0, 1, 2, 6, 7, 8, 9, 10, 11]
    for first, second in zip(baseline, result):
        assert torch.equal(first[:, keep], second[:, keep])
    rescaled = relation(features, z * 17, similarity, support, 3)
    for first, second in zip(baseline, rescaled):
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
