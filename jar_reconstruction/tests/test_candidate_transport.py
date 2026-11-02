"""候选运输的斜面关系、实际梯度、退化与物理候选格测试。"""

import copy

import torch

from jar.models.candidate_transport import CandidateSurfaceTransport, _sample_curves


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(23)
    cost = torch.randn(1, 8, 7, 5, 5, requires_grad=True)
    scores = torch.randn(1, 7, 5, 5, requires_grad=True)
    valid = torch.ones_like(scores, dtype=torch.bool)
    inverse = torch.tensor([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7])
    intrinsic = torch.tensor([[4.0, 0, 2], [0, 4.0, -1], [0, 0, 1]])
    transform = torch.eye(4)
    transform[:3, :3] = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
    features = (torch.ones(1, 4, 5, 5) + 0.05 * torch.randn(1, 4, 5, 5)).requires_grad_()
    model = CandidateSurfaceTransport(8, 4, {"hidden_dim": 8})
    return model, (cost, scores, valid, inverse, intrinsic, transform, 1, features)


def test_transport_finite_gradients_and_bounds():
    model, arguments = setup()
    result = model(*arguments)
    assert result.shape == arguments[1].shape
    assert torch.isfinite(result).all()
    assert (result >= 0).all() and (result <= 2).all() and result.max() > 0
    result.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    for tensor in (arguments[0], arguments[1], arguments[-1]):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_unknown_flat_and_single_neighbor_are_neutral():
    model, arguments = setup()
    arguments = list(arguments)
    arguments[2] = torch.zeros_like(arguments[2])
    result = model(*arguments)
    assert torch.isfinite(result).all() and not result.any()
    arguments[2] = torch.ones_like(arguments[2])
    arguments[1] = torch.ones_like(arguments[1]) * 3
    torch.testing.assert_close(model(*arguments), torch.zeros_like(result), atol=1e-7, rtol=0)
    arguments[1] = torch.randn_like(arguments[1])
    arguments[2].zero_()
    arguments[2][:, :, 2, 2:4] = True
    assert not model(*arguments).any()


def test_nonuniform_unordered_interpolation_does_not_cross_unknown():
    lattice = torch.tensor([0.4, 0.1, 0.2, 0.7])
    values = (2 * lattice + 1).reshape(1, 4, 1, 1)
    valid = torch.ones_like(values, dtype=torch.bool)
    query = torch.tensor([0.1, 0.15, 0.3, 0.7, 0.8]).reshape(1, 1, 5, 1, 1)
    sampled, known = _sample_curves(values, valid, lattice, query)
    torch.testing.assert_close(sampled[known], (2 * query + 1)[known])
    assert known.flatten().tolist() == [True, True, True, True, False]
    valid[:, 2] = False
    _, known = _sample_curves(values, valid, lattice, query)
    assert known.flatten().tolist() == [True, False, False, True, False]


def test_calibrated_horizontal_plane_aligns_different_neighbor_depths():
    model, arguments = setup()
    arguments = list(arguments)
    lattice = arguments[3]
    # 已知水平面在此相机下满足 rho(v)=0.1*(v+1)，是一个真实投影关系。
    scores = torch.full_like(arguments[1], -4)
    for row in range(5):
        index = (lattice - 0.1 * (row + 1)).abs().argmin()
        scores[0, index, row] = 4
    arguments[1] = scores
    arguments[-1] = torch.ones_like(arguments[-1])
    with torch.no_grad():
        model.plane_logits.weight.zero_()
        model.plane_logits.bias.fill_(-10)
        model.plane_logits.bias[1] = 10
    same_depth = copy.deepcopy(model)
    same_depth.transport = "same_depth"
    aligned = model(*arguments)
    unaligned = same_depth(*arguments)
    center_depth = int((lattice - 0.3).abs().argmin())
    assert aligned[0, center_depth, 2, 2] > 10 * unaligned[0, center_depth, 2, 2]
    assert aligned[0, center_depth, 2, 2] > 0.1
    # 打乱候选编号并同步打乱数据，物理结果必须只发生同样的排列。
    order = torch.tensor([3, 6, 0, 2, 5, 1, 4])
    shuffled = list(arguments)
    shuffled[0] = arguments[0][:, :, order]
    shuffled[1] = arguments[1][:, order]
    shuffled[2] = arguments[2][:, order]
    shuffled[3] = lattice[order]
    torch.testing.assert_close(model(*shuffled), aligned[:, order], atol=1e-6, rtol=1e-6)
