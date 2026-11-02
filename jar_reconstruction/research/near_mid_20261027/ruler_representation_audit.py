"""CPU标尺表示反例与真实TRAIN标定审计；不训练、不读取GT、不修改模型。

将结果打印为JSON，输出路径由调用者决定。这里的指定深度是几何探针，
不是模型实际候选，因此不能把数值变化当作精度提升。
"""

import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.projective_ruler import compile_projective_ruler


def quantiles(values, mask):
    values = values[mask].double()
    if not len(values):
        return {'count': 0}
    return {'count': len(values), 'median': values.median().item(),
            'p90': torch.quantile(values, .9).item(), 'max': values.max().item()}


def exact_chord(derivative, delta, camera_slope, camera_depth):
    """uv(Z+delta)-uv(Z)，仅用于源光轴深度仍为正的点。"""
    return derivative * delta[..., None] / (1 + camera_slope * delta / camera_depth)[..., None]


def geometry_counterexample():
    # 同方向、同g、同h，因此旧源pair的几何输入及g²相同，但有限区间透视不同。
    records = []
    for depth, fb in ((10., 100.), (30., 900.)):
        half = 2.
        g = fb / depth**2
        left = fb / (depth - half) - fb / depth
        right = fb / depth - fb / (depth + half)
        records.append({'z': depth, 'feature_fb': fb, 'g': g, 'half': half,
                        'linear_span': g * half, 'kappa': half / depth,
                        'exact_left': left, 'exact_right': right})
    assert records[0]['g'] == records[1]['g']
    assert records[0]['linear_span'] == records[1]['linear_span']
    assert abs(records[0]['exact_left'] - records[1]['exact_left']) > .3
    return records


def shared_feature_direction():
    """非等间隔采样的有符号特征响应，不对分类score拟合抛物线。"""
    dtype = torch.float64
    t = torch.tensor([-.7, 0., 1.], dtype=dtype)
    gradients = torch.tensor([[2., .4], [-1., .3]], dtype=dtype)
    truth_delta = .2
    residual = -gradients * truth_delta
    features = residual[:, None] + gradients[:, None] * t[None, :, None]
    differences = features[:, [0, 2]] - features[:, 1:2]
    derivative = (differences * t[[0, 2]][None, :, None]).sum(1) / t[[0, 2]].square().sum()
    torch.testing.assert_close(derivative, gradients, rtol=1e-12, atol=1e-12)
    h = derivative.square().sum(-1)
    b = (derivative * residual).sum(-1)
    update = -b.sum() / h.sum()
    conflict = (b.square() / h).sum() - b.sum().square() / h.sum()
    assert abs(update.item() - truth_delta) < 1e-12
    assert abs(conflict.item()) < 1e-12
    # 源图中的移动方向可以相反，但反推的同一深度方向一致。
    image_directions = gradients[:, 0] * truth_delta
    assert image_directions[0] > 0 and image_directions[1] < 0
    conflict_residual = residual.clone()
    conflict_residual[1] *= -1
    bad_b = (derivative * conflict_residual).sum(-1)
    bad_conflict = (bad_b.square() / h).sum() - bad_b.sum().square() / h.sum()
    assert bad_conflict > .1
    # 高几何敏感度但恒定图像纹理：特征H为0，不允许几何J替代匹配信息。
    flat = torch.ones_like(features)
    flat_derivative = ((flat[:, [0, 2]] - flat[:, 1:2]) * t[[0, 2]][None, :, None]).sum(1)
    assert flat_derivative.square().sum() == 0
    # 各层特征本身为常数，单靠层路由随t变化也会制造非零“导数”。
    level0 = torch.tensor([0., 0.], dtype=dtype)
    level1 = torch.tensor([1., 2.], dtype=dtype)
    level_weight = .5 + .2 * t
    routed = level0 + level_weight[:, None] * (level1 - level0)
    routing_derivative = ((routed[[0, 2]] - routed[1]) * t[[0, 2], None]).sum(0) / t[[0, 2]].square().sum()
    assert routing_derivative.square().sum() > .1
    return {'recovered_delta': update.item(), 'truth_delta': truth_delta,
            'coherent_conflict': conflict.item(), 'incompatible_conflict': bad_conflict.item(),
            'image_horizontal_moves': image_directions.tolist(), 'flat_feature_information': 0.,
            'constant_image_route_only_feature_information': routing_derivative.square().sum().item(),
            'caveat': '共同错误同样可以Q=0；这些合成线性特征只验证数学，不是网络准确率。'}


def probe_domain(domain, cache_path):
    row = torch.load(ROOT / cache_path, map_location='cpu', weights_only=False, mmap=True)
    work = row['work']
    intrinsic, transform = work['K'][0].double(), work['T_cam_from_rig'][0].double()
    height, width = work['image_size']
    reference = work['reference_index']
    source = [i for i in range(len(intrinsic)) if i != reference]
    y, x = torch.meshgrid(torch.linspace(.2 * (height - 1), .8 * (height - 1), 5),
                          torch.linspace(.2 * (width - 1), .8 * (width - 1), 9), indexing='ij')
    pixels = torch.stack((x, y), -1).reshape(-1, 2).double()
    ruler = compile_projective_ruler(pixels, intrinsic[reference], transform[reference],
                                     intrinsic[source], transform[source])
    scale = work['matching_scale'][0].double()
    feature_scale = scale / 4
    unit = domain['depth_scale']
    near, far = domain['near'] / unit, domain['far'] / unit
    locations = (5., 15., 30.) if domain['geometry_units'] == 'meters' else tuple(
        n * 10 * unit for n in (.25, .75, 1.5))
    result = []
    for location in locations:
        center = torch.full((1, len(pixels)), location / unit, dtype=torch.float64)
        if not near < center[0, 0] < far:
            continue
        geometry = ruler.project(center, (height, width))
        derivative = geometry['depth_derivative'][:, 0] * feature_scale
        g = derivative.norm(dim=-1)
        valid = geometry['valid'][:, 0] & (g > 1e-12)
        strongest = torch.where(valid, g, 0).amax(0)
        lattice = torch.logspace(math.log10(near), math.log10(far), 96, dtype=torch.float64)
        edges = torch.cat((lattice[:1], (lattice[1:] + lattice[:-1]) / 2, lattice[-1:]))
        index = (lattice.log() - center[0, 0].log()).abs().argmin()
        half_cell = (edges[index + 1] - edges[index]) / 2
        step = torch.maximum(half_cell, .5 / strongest.clamp_min(1e-6)).clamp(.005, 100.)
        candidates = (center + torch.tensor([-1., 0., 1.], dtype=torch.float64)[:, None] * step).clamp(near, far)
        projected = ruler.project(candidates, (height, width))
        camera_depth = geometry['optical_depth'][:, 0]
        camera_slope = ruler.camera_direction[..., 2]
        kappa = camera_slope * step[None] / (2 * camera_depth)
        exact_error = []
        shifts = []
        for side in (0, 2):
            delta = (candidates[side] - center[0])[None].expand_as(g)
            actual = (projected['uv'][:, side] - geometry['uv'][:, 0]) * feature_scale
            predicted = exact_chord(derivative, delta, camera_slope, camera_depth)
            positive = valid & (projected['optical_depth'][:, side] > 1e-4)
            torch.testing.assert_close(actual[positive], predicted[positive], rtol=1e-9, atol=1e-9)
            exact_error.append((actual - predicted).abs()[positive].max().item() if positive.any() else 0.)
            shifts.append(actual.norm(dim=-1))
        pair_valid = valid & projected['valid'][:, 0] & projected['valid'][:, 2]
        asymmetry = shifts[0] / shifts[1].clamp_min(1e-12)
        footprint = projected['pixel_derivative']
        spans = torch.stack(((footprint[..., 0] * scale).norm(dim=-1) * 4,
                             (footprint[..., 1] * scale).norm(dim=-1) * 4,
                             (projected['depth_derivative'] * scale).norm(dim=-1) * step[None, None]), -1)
        levels = (spans.amax(-1).clamp_min(1) / 16).log2().clamp(0, 2)
        route_change = (levels[:, [0, 2]] - levels[:, 1:2]).abs().amax(1)
        # 统一物理单位改变后：光轴深度、平移和步长同步缩放，弦长/κ不变。
        factor = 17.3
        changed = transform.clone()
        changed[:, :3, 3] *= factor
        scaled_ruler = compile_projective_ruler(pixels, intrinsic[reference], changed[reference],
                                               intrinsic[source], changed[source])
        scaled = scaled_ruler.project(candidates * factor, (height, width))
        # helper对相机后方使用固定安全分母；该占位值不代表物理投影，
        # 不能要求它具有单位协变性。只检查两种单位下均在相机前的投影。
        positive_depth = (projected['optical_depth'] > 1e-4) & (scaled['optical_depth'] > 1e-4)
        torch.testing.assert_close(scaled['uv'][positive_depth], projected['uv'][positive_depth],
                                   rtol=1e-11, atol=1e-10)
        scaled_kappa = scaled_ruler.camera_direction[..., 2] * (step[None] * factor / 2)
        scaled_kappa /= scaled['optical_depth'][:, 1]
        torch.testing.assert_close(scaled_kappa[valid], kappa[valid], rtol=1e-11, atol=1e-11)
        result.append({'probe_depth_original_units': location,
                       'probe_unit': domain['geometry_units'], 'central_valid': int(valid.sum()),
                       'step_original_units': quantiles(step * unit, strongest > 0),
                       'half_interval_kappa': quantiles(kappa, valid),
                       'full_three_point_left_over_right_chord': quantiles(asymmetry, pair_valid),
                       'route_changed_fraction': float((route_change[pair_valid] > 1e-6).float().mean())
                       if pair_valid.any() else None,
                       'max_exact_formula_error': max(exact_error),
                       'unit_covariance_verified': True})
    return {'cache': cache_path, 'num_views': len(intrinsic), 'records': result}


def main():
    torch.set_num_threads(1)
    index = json.loads((ROOT / 'runs/active/architecture_screen_20261026/features/index.json').read_text())
    result = {'protocol': 'CPU FP64；每域首个既有TRAIN缓存，仅读K/T和尺寸；人工探针深度非模型候选；不读GT、不访问TEST、不运行网络、不修改正在训练代码。',
              'finite_interval_alias': geometry_counterexample(),
              'feature_direction_checks': shared_feature_direction(),
              'datasets': {}}
    for domain in index['domains']:
        path = index['splits']['train'][domain['name']][0]
        result['datasets'][domain['name']] = probe_domain(domain, path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
