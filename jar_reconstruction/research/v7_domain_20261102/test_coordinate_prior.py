"""连续候选码使用真实深度，且候选置换只置换相应输出。"""

import torch

from research.v7_domain_20261102.coordinate_prior import CoordinateEvidencePrior


def example():
    torch.manual_seed(20261102)
    model = CoordinateEvidencePrior(raw_channels=8, candidates=6, hidden=16)
    raw = [torch.randn(2, 8, 3, 5) for _ in range(4)]
    depths = torch.linspace(.5, 20., 6)[None, :, None, None].expand(2, -1, 3, 5).clone()
    information = torch.full_like(depths, .1, requires_grad=True)
    null = torch.full_like(depths, .2, requires_grad=True)
    return model, raw, depths, information, null


def test_zero_initialization_and_two_updates():
    model, raw, depths, information, null = example()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    output = model(raw, information, null, depths)
    assert torch.equal(output, torch.zeros_like(output))
    target = torch.randn_like(output)
    (output - target).square().mean().backward()
    assert model.output.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()
    (model(raw, information, null, depths) - target).square().mean().backward()
    assert model.project[0].weight.grad.abs().sum() > 0
    assert model.depth_code[0].weight.grad.abs().sum() > 0
    assert information.grad is None and null.grad is None
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_candidate_permutation_and_values():
    model, raw, depths, information, null = example()
    with torch.no_grad():
        model.output.weight.normal_(std=.1)
    # 相同门控只研究内容坐标；零信息使gate恒为1。
    information = torch.zeros_like(information)
    order = torch.tensor([4, 0, 3, 1, 5, 2])
    output = model(raw, information, null, depths)
    permuted = model(raw, information[:, order], null[:, order], depths[:, order])
    torch.testing.assert_close(permuted, output[:, order], rtol=1e-6, atol=1e-7)
    changed = model(raw, information, null, depths * 2)
    assert not torch.allclose(changed, output)
    # 改候选数不需要更换预测头，且同值候选拥有相同内容码。
    duplicate = depths[:, :1].expand(-1, 9, -1, -1)
    value = model(raw, information[:, :1].expand_as(duplicate),
                  null[:, :1].expand_as(duplicate), duplicate)
    torch.testing.assert_close(value, value[:, :1].expand_as(value), rtol=0, atol=0)


def test_depth_gradient_and_strict_reload():
    model, raw, depths, information, null = example()
    with torch.no_grad():
        model.output.weight.normal_(std=.1)
    depths.requires_grad_()
    output = model(raw, information, null, depths)
    output.square().mean().backward()
    assert depths.grad.abs().sum() > 0 and torch.isfinite(depths.grad).all()
    restored = CoordinateEvidencePrior(raw_channels=8, candidates=6, hidden=16)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(output, restored(raw, information, null, depths))
