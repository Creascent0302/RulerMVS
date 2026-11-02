"""上下文补证据模块的数学契约；不代替完整模型精度/速度实验。"""

import torch

from research.v7_domain_20261102.prior import EvidencePrior


def example():
    torch.manual_seed(20261102)
    raw = [torch.randn(2, 8, 3, 5) for _ in range(4)]
    candidates = torch.linspace(1., 20., 6)[None, :, None, None].expand(2, -1, 3, 5)
    information = torch.full((2, 1, 6, 3, 5), 0.1, requires_grad=True)
    null = torch.full_like(information, 0.2, requires_grad=True)
    model = EvidencePrior(raw_channels=8, candidates=6, hidden=16)
    return model, raw, information, null, candidates


def test_zero_initialization_and_first_update():
    model, raw, information, null, candidates = example()
    residual = model(raw, information, null, candidates)
    assert torch.equal(residual, torch.zeros_like(residual))
    target = torch.randn_like(residual)
    (residual - target).square().mean().backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert information.grad is None
    assert null.grad is None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    updated = model(raw, information, null, candidates)
    (updated - target).square().mean().backward()
    assert model.project[0].weight.grad.abs().sum() > 0
    assert torch.isfinite(updated).all()


def test_gate_units_and_quality_order():
    model, raw, information, null, candidates = example()
    gate = model.quality_gate(information, null, candidates)
    # 同一几何更换长度单位：Z乘s，投影导数除s；门保持相同。
    converted = model.quality_gate(information / 4., null, candidates * 2.)
    torch.testing.assert_close(gate, converted, rtol=0., atol=0.)
    stronger = model.quality_gate(information * 10., null, candidates)
    assert torch.all(stronger <= gate)
    rejected = model.quality_gate(information, torch.ones_like(null), candidates)
    absent = model.quality_gate(torch.zeros_like(information), null, candidates)
    assert torch.equal(rejected, torch.ones_like(rejected))
    assert torch.equal(absent, torch.ones_like(absent))
    assert gate.min() >= 0 and gate.max() <= 1


def test_checkpoint_reload_and_unavailable_evidence():
    model, raw, information, null, candidates = example()
    with torch.no_grad():
        model.output.weight.normal_(std=0.01)
    information = torch.full_like(information, float("nan"))
    null = torch.full_like(null, float("nan"))
    prediction = model(raw, information, null, candidates)
    clone = EvidencePrior(raw_channels=8, candidates=6, hidden=16)
    clone.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(prediction, clone(raw, information, null, candidates))
    assert torch.isfinite(prediction).all()
