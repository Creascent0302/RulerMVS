"""只在 CPU 审计固定 DEV 标定字段；不运行网络、不选择或修改模型。

96 粗候选为实际格点。GT 深度探针只作几何诊断，不是模型细候选。
"""

import json
import math
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.geometry import pixel_grid
from jar.models.metric_ruler_network import depth_cell_edges
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.projective_ruler import compile_projective_ruler


def regions(depth, domain):
    if domain['geometry_units'] == 'meters':
        low, high = 10., 40.
    else:
        median = 10 * domain['depth_scale']
        low, high = .5 * median, 2 * median
    return {'all': torch.ones_like(depth, dtype=torch.bool), 'near': depth < low,
            'middle': (depth >= low) & (depth < high), 'far': depth >= high}


def collect(destination, group, name, value, mask):
    selected = value.expand_as(mask)[mask].flatten().float()
    if selected.numel():
        if not bool(torch.isfinite(selected).all()):
            raise ValueError((group, name, '非有限数值'))
        # 保留可精确汇总的有限轻量几何值，运行结束后只存统计，不存向量。
        destination.setdefault(group, {}).setdefault(name, []).append(selected)


def summarize(destination):
    result = {}
    for group, fields in destination.items():
        result[group] = {}
        for name, chunks in fields.items():
            values = torch.cat(chunks)
            quantiles = torch.quantile(values, torch.tensor([0., .1, .5, .9, .99, 1.]))
            result[group][name] = {'count': values.numel(), 'mean': float(values.double().mean()),
                                   **dict(zip(['min', 'p10', 'p50', 'p90', 'p99', 'max'], quantiles.tolist()))}
    return result


def audit_projection(work, depths, half, mask, domain, bag, prefix, ruler, grid):
    """所有长度先按生产 depth_scale 归一化；细层的实际预测不在本审计中。"""
    scale = float(domain['depth_scale'])
    ref = work['reference_index']
    sources = [i for i in range(work['K'].shape[1]) if i != ref]
    image_hw = work['image_size']
    projected = ruler.project(depths, image_hw, work['view_mask'][0, sources],
                              work['image_mask'][0, sources])
    raw_derivative = projected['depth_derivative']
    matching_scale = work['matching_scale'][0]
    derivative = raw_derivative * matching_scale / 4
    g = torch.linalg.vector_norm(derivative, dim=-1)
    geometric_valid = projected['valid'] & mask[None]
    derivative_pass = torch.linalg.vector_norm(raw_derivative, dim=-1) > 1e-5
    valid = geometric_valid & derivative_pass
    amplitude = g * half[None]
    encoded = amplitude.log1p().clamp_max(8)
    footprint = projected['pixel_derivative']
    spans = torch.stack((
        torch.linalg.vector_norm(footprint[..., 0] * matching_scale, dim=-1) * 8,
        torch.linalg.vector_norm(footprint[..., 1] * matching_scale, dim=-1) * 8,
        amplitude * 8), -1)
    route = (spans.amax(-1).clamp_min(1) / 16).log2().clamp(0, 2)
    # h/Z 描述局部线性化半径，精确两侧投影检测一阶幅度近似的非对称性。
    low = (depths - half).clamp_min(domain['near'] / scale)
    high = (depths + half).clamp_max(domain['far'] / scale)
    low_p = ruler.project(low, image_hw)
    high_p = ruler.project(high, image_hw)
    low_delta = torch.linalg.vector_norm((low_p['uv'] - projected['uv']) * matching_scale / 4, dim=-1)
    high_delta = torch.linalg.vector_norm((high_p['uv'] - projected['uv']) * matching_scale / 4, dim=-1)
    reliable_endpoints = valid & low_p['valid'] & high_p['valid'] & (amplitude > 1e-6)
    direction = derivative / g[..., None].clamp_min(1e-6)
    for region, region_mask in regions(depths * scale, domain).items():
        selected = region_mask[None] & valid
        group = prefix + '/' + region
        metrics = {
            'depth_original_units': depths[None] * scale,
            'half_width_original_units': half[None] * scale,
            'half_width_over_depth': (half / depths)[None],
            'g_match4_pixels_per_normalized_depth_unit': g,
            'g_match4_pixels_per_original_depth_unit': g / scale,
            'g_times_depth_dimensionless': g * depths[None],
            'half_cell_match4_pixel_motion': amplitude,
            'encoded_pair_log1p_amplitude': encoded,
            'pair_clip8': (amplitude.log1p() >= 8).float(),
            'pair_encoding_derivative_wrt_amplitude': (1 + amplitude).reciprocal(),
            'route': route,
            'route_at0': (route == 0).float(),
            'route_at2': (route == 2).float(),
            'route_depth_span_dominates': (spans[..., 2] > spans[..., :2].amax(-1)).float(),
        }
        for name, value in metrics.items():
            collect(bag, group, name, value, selected)
        collect(bag, group, 'endpoint_low_over_linear', low_delta / amplitude.clamp_min(1e-6),
                reliable_endpoints & region_mask[None])
        collect(bag, group, 'endpoint_high_over_linear', high_delta / amplitude.clamp_min(1e-6),
                reliable_endpoints & region_mask[None])
        collect(bag, group, 'derivative_threshold_rejected', (~derivative_pass).float(),
                geometric_valid & region_mask[None])
        pixel_mask = mask & region_mask
        max_g = torch.where(valid, g, 0).amax(0)
        # learned information = sum(w*g^2), 0<=sum(w)<=1。这里仅精确上界，非注意力实测。
        collect(bag, group, 'learned_information_log1p_upper_bound', max_g.square().log1p(), pixel_mask)
        collect(bag, group, 'no_source', (~valid.any(0)).float(), pixel_mask)
        collect(bag, group, 'source_count', valid.sum(0).float(), pixel_mask)
        if depths.shape[0] > 1:
            pair_valid = valid[:, :-1] & valid[:, 1:] & region_mask[None, :-1] & region_mask[None, 1:]
            amplitude_gap = (encoded[:, 1:] - encoded[:, :-1]).abs()
            collect(bag, group, 'adjacent_log1p_amplitude_gap', amplitude_gap, pair_valid)
            collect(bag, group, 'adjacent_amplitude_bfloat16_collision',
                    (encoded[:, 1:].bfloat16() == encoded[:, :-1].bfloat16()).float(), pair_valid)
            collect(bag, group, 'adjacent_direction_delta',
                    torch.linalg.vector_norm(direction[:, 1:] - direction[:, :-1], dim=-1), pair_valid)
    return valid, g


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = ROOT / 'runs/active/near_mid_20261027/geometry_encoding_audit'
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent.resolve()).free < 204 * 2**30:
        raise RuntimeError('数据盘可用空间小于204GiB')
    output.mkdir(parents=True)
    index_path = ROOT / 'runs/active/architecture_screen_20261026/features/index.json'
    index = json.loads(index_path.read_text())
    summary = {}
    start = time.monotonic()
    for domain in index['domains']:
        bag = {}
        near, far = domain['near'] / domain['depth_scale'], domain['far'] / domain['depth_scale']
        lattice = torch.logspace(math.log10(near), math.log10(far), 96)
        edges = depth_cell_edges(lattice)
        half = (edges[1:] - edges[:-1]) / 2
        for path in index['splits']['development'][domain['name']]:
            row = torch.load(ROOT / path, map_location='cpu', weights_only=False, mmap=True)
            work = row['work']
            ref = work['reference_index']
            sources = [i for i in range(work['K'].shape[1]) if i != ref]
            height, width = work['image_size']
            grid = pixel_grid(height // 8, width // 8, stride=8).reshape(-1, 2)
            ruler = compile_projective_ruler(grid, work['K'][0, ref], work['T_cam_from_rig'][0, ref],
                                             work['K'][0, sources], work['T_cam_from_rig'][0, sources])
            valid = work['image_mask'][0, ref, ::8, ::8].flatten() & work['view_mask'][0, ref]
            depths = lattice[:, None].expand(-1, len(grid))
            audit_projection(work, depths, half[:, None].expand_as(depths), valid[None].expand_as(depths),
                             domain, bag, 'actual_coarse_lattice', ruler, grid)
            truth = row['targets']['depth'][0, 0, ::8, ::8].flatten()
            truth_valid = row['targets']['depth_valid'][0, 0, ::8, ::8].flatten() & valid
            truth_valid &= torch.isfinite(truth) & (truth > 0)
            gt = torch.where(truth_valid, truth / domain['depth_scale'], lattice[0])[None]
            # nearest log格宽；不是预测fine候选或fine宽度。
            nearest = (gt.log() - lattice[:, None].log()).abs().argmin(0)
            gt_half = half[nearest][None]
            audit_projection(work, gt, gt_half, truth_valid[None], domain, bag,
                             'gt_ray_with_nearest_coarse_half_width', ruler, grid)
        summary[domain['name']] = {'geometry_units': domain['geometry_units'], 'depth_scale': domain['depth_scale'],
                                  'normalized_range': [near, far], 'statistics': summarize(bag)}
        write_json(output / (domain['name'] + '.json'), summary[domain['name']])
        print(domain['name'], round(time.monotonic() - start, 2), flush=True)
    write_json(output / 'summary.json', summary)
    write_json(output / 'protocol.json', {
        'device': 'cpu', 'threads': 2, 'model_forward': False, 'training': False, 'test_accessed': False,
        'frames': 56, 'index_sha256': digest(index_path), 'script_sha256': digest(__file__),
        'checkpoint_origin_sha256': index['checkpoint_sha256'],
        'unit': 'Znormalized=Zoriginal/depth_scale; ray derivative is matching-stride4 pixels / normalized depth unit',
        'metric_information': 'actual learned weights not evaluated; log1p(max valid source g^2) is upper bound only',
        'gt_probes': 'GT only selects offline probe depth and distance bin; not actual fine predictions',
        'routing': 'coarse stride8 formula; GT probe also uses stride8 footprint for controlled audit',
        'binning': 'meters:<10/10..40/>=40;scene_unit:<.5/.5..2/>=2 times TRAINmedian=10*depth_scale',
        'clip': 'pair amplitude log1p clipped at8; geometric threshold original-image derivative norm>1e-5',
        'endpoint': 'only both projected cell endpoints in view, clamped to preregistered TRAIN depth range',
    })
    shutil.copy2(__file__, output / Path(__file__).name)
    write_json(output / 'complete.json', {'exit_code': 0, 'elapsed_seconds': time.monotonic() - start,
                                        'free_gib': shutil.disk_usage(output.resolve()).free / 2**30})


if __name__ == '__main__':
    main()
