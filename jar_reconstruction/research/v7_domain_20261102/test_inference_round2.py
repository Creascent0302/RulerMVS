"""推理复用只改变公共格点的重复计算，非共享候选完整回退。"""

import torch

from research.v7_domain_20261102.coordinate_prior import CoordinateEvidencePrior
from research.v7_domain_20261102.inference_round2 import prepare_coordinate_reuse


def test_depth_code_reuse_and_exact_disable():
    torch.manual_seed(42)
    model = CoordinateEvidencePrior(raw_channels=8).eval().requires_grad_(False)
    model.output.weight.normal_(std=.01)
    model.output.bias.normal_(std=.01)
    raw = [torch.randn(1, 8, 4, 8) for _ in range(4)]
    candidates = torch.logspace(-1, 2, 96)[None, :, None, None].expand(1, -1, 4, 8)
    information = torch.rand(1, 1, 96, 4, 8)
    null = torch.rand_like(information)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    shapes = []
    handle = model.depth_code.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape)))
    original = model(raw, information, null, candidates)
    prepare_coordinate_reuse(model)
    reduced = model(raw, information, null, candidates)
    assert shapes[:2] == [(1, 96, 4, 8, 13), (1, 96, 1, 1, 13)]
    torch.testing.assert_close(original, reduced, rtol=2e-5, atol=1e-7)
    # 即使数值相同，非零空间stride仍走原forward，不进行昂贵逐元素判等。
    dense = candidates.contiguous()
    fallback = model(raw, information, null, dense)
    assert shapes[-1] == (1, 96, 4, 8, 13)
    assert torch.equal(fallback, original)
    prepare_coordinate_reuse(model, False)
    restored = model(raw, information, null, candidates)
    assert torch.equal(restored, original)
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    assert set(model.state_dict()) == set(state)
    handle.remove()
