"""实际运行官方占据 CUDA 算子；直接调用原扩展，不用替代实现推理。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
GF = ROOT / "upstream/gaussianformer"


def package(name, path):
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_grad(tensors):
    for tensor in tensors:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def gaussian_prob():
    sys.path.insert(0, str(GF / "model/head/localagg_prob"))
    from local_aggregate_prob import LocalAggregator

    agg = LocalAggregator(4, 8, 8, 8, [0, 0, 0], 1.0).cuda()
    means = torch.tensor([[[3.1, 3.2, 3.3], [4.4, 4.2, 4.1]]], device="cuda", requires_grad=True)
    opa = torch.tensor([[0.3, 0.7]], device="cuda", requires_grad=True)
    sem = torch.randn(1, 2, 18, device="cuda").softmax(-1).requires_grad_()
    scales = torch.full((1, 2, 3), 1.2, device="cuda")
    cov = (torch.eye(3, device="cuda")[None, None].repeat(1, 2, 1, 1) / 1.2**2).requires_grad_()
    # One sample per voxel: the upstream backward kernel uses voxel2pts.
    pts = torch.tensor([[[2.5, 3.5, 3.5], [3.5, 3.5, 3.5], [4.5, 4.5, 4.5]]], device="cuda")
    out = agg(pts, means, opa, sem, scales, cov)
    coeff = torch.randn_like(out[0])
    loss = (out[0] * coeff).sum() + out[1].sum() + 0.1 * out[2].sum()
    loss.backward()
    check_grad([means, opa, sem, cov])
    # Independent dense formula is used ONLY as a numerical test oracle.
    m, o, s, c = [x.detach().clone().requires_grad_() for x in [means, opa, sem, cov]]
    delta = m[:, None] - pts[:, :, None]
    power = torch.exp(-0.5 * torch.einsum("bngi,bgij,bngj->bng", delta, c, delta))
    prob = power * torch.linalg.det(c).sqrt()[:, None] * o[:, None] / (2 * math.pi) ** 1.5
    ref_sem = (prob[..., None] * s[:, None]).sum(2) / prob.sum(2, keepdim=True)
    ref_bin = 1 - (1 - power).prod(2)
    ref_density = power.sum(2)
    refs = [ref_sem[0], ref_bin[0], ref_density[0]]
    errors = [float((a - b).detach().abs().max()) for a, b in zip(out, refs)]
    for a, b in zip(out, refs):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    ((ref_sem[0] * coeff).sum() + ref_bin.sum() + 0.1 * ref_density.sum()).backward()
    for a, b in [(means, m), (opa, o), (sem, s)]:
        torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=1e-4)
    # Upstream uses packed symmetric covariance: diagonal gradients compare directly.
    torch.testing.assert_close(
        cov.grad.diagonal(dim1=-2, dim2=-1), c.grad.diagonal(dim1=-2, dim2=-1), atol=2e-5, rtol=1e-4
    )
    return {"forward_backward": "passed", "formula_max_abs_errors": errors}


def deformable():
    mod = package("_gf2_probe_ops", GF / "model/encoder/gaussian_encoder/ops")
    fn = mod.DeformableAggregationFunction
    features = [torch.randn(1, 4, 8, h, h, device="cuda", requires_grad=True) for h in [4, 2]]
    packed = fn.feature_maps_format(features)
    uv = (0.1 + 0.8 * torch.rand(1, 6, 4, 2, device="cuda")).requires_grad_()
    weights = torch.rand(1, 6, 4, 2, 2, device="cuda", requires_grad=True)
    result = fn.apply(*packed, uv, weights)
    assert result.shape == (1, 6, 8) and torch.isfinite(result).all()
    result.square().mean().backward()
    check_grad(features + [uv, weights])
    return {"forward_backward": "passed", "output_shape": list(result.shape)}


def octree_mask():
    sys.path.insert(0, str(ROOT / "upstream/octreeocc"))
    import octree_ops

    source = torch.rand(2, 3, 4, 2, device="cuda") > 0.5
    out = torch.zeros(2, 6, 8, 4, dtype=torch.bool, device="cuda")
    octree_ops.octree_mask_l1_to_l2_forward(source, out)
    expected = source.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3)
    assert torch.equal(out, expected)
    final = torch.zeros(2, 12, 16, 8, dtype=torch.bool, device="cuda")
    octree_ops.octree_mask_l2_to_l3_forward(out, final)
    assert torch.equal(
        final, expected.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3)
    )
    return {"forward": "passed", "backward": "not_applicable_boolean_operator"}


def sparse_conv():
    import spconv.pytorch as spconv

    layer = spconv.SubMConv3d(8, 8, 3, padding=1, bias=False).cuda()
    feature = torch.randn(8, 8, device="cuda", requires_grad=True)
    indices = torch.tensor(
        [[0, a, b, c] for a in (1, 2) for b in (1, 2) for c in (1, 2)],
        device="cuda",
        dtype=torch.int32,
    )
    out = layer(spconv.SparseConvTensor(feature, indices, [4, 4, 4], 1)).features
    out.square().sum().backward()
    check_grad([feature, layer.weight])
    return {"forward_backward": "passed"}


def dcn():
    sys.path.insert(0, str(ROOT / "build_src/mmcv-2.0.1"))
    from mmcv.ops import ModulatedDeformConv2dPack

    layer = ModulatedDeformConv2dPack(8, 8, 3, padding=1).cuda()
    feature = torch.randn(2, 8, 8, 8, device="cuda", requires_grad=True)
    layer(feature).square().mean().backward()
    check_grad([feature, layer.weight, layer.conv_offset.weight])
    return {"forward_backward": "passed"}


def fps():
    import pointops

    points = torch.randn(128, 3, device="cuda")
    offsets = torch.tensor([128], dtype=torch.int32, device="cuda")
    new_offsets = torch.tensor([16], dtype=torch.int32, device="cuda")
    chosen = pointops.farthest_point_sampling(points, offsets, new_offsets).long()
    assert chosen.numel() == 16 and chosen.unique().numel() == 16
    # 索引选择离散无反向；下游点坐标 gather 仍可传梯度。
    differentiable = points.clone().requires_grad_()
    differentiable[chosen].square().sum().backward()
    check_grad([differentiable])
    return {
        "forward": "passed",
        "backward": "not_applicable_discrete_selection",
        "gather_gradient": "passed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["gaussian_prob", "deformable", "octree_mask", "sparse_conv", "dcn", "fps"],
    )
    args = parser.parse_args()
    torch.manual_seed(17)
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    failures = 0
    for name in args.ops:
        try:
            report[name] = globals()[name]()
            torch.cuda.synchronize()
        except Exception as exc:
            report[name] = {"status": "failed", "error": repr(exc)}
            failures += 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
