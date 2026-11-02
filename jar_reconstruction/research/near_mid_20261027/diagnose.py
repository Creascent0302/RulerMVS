"""固定56组DEV：中近距阶段、表面边界和上采样误差诊断。

GT仅用于离线分组和oracle，不进入原模型前向；不访问TEST或修改权重。
"""

import json
from pathlib import Path
import shutil
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.geometry import safe_masked_softmax
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.model import CalibratedDecoder
from jar.runtime import move
from tools.probe_public_stages import (
    convex_diagnostics, edge_groups, error_statistics, fine_error_decomposition,
    stage_statistics, sum_records, summarize,
)


def distance_masks(truth, valid, domain):
    if domain['geometry_units'] == 'meters':
        near, middle = 10., 40.
    else:
        median = 10 * domain['depth_scale']
        near, middle = .5 * median, 2 * median
    return {
        'all': valid,
        'near': valid & (truth < near),
        'middle': valid & (truth >= near) & (truth < middle),
        'far': valid & (truth >= middle),
    }


def pack(value):
    batch, _, _, _, height, width = value.shape
    return value.permute(0, 1, 4, 2, 5, 3).reshape(batch, 1, height * 4, width * 4)


def local_plane_groups(truth, valid, gradient_band):
    """理想透视平面逆Z对像素坐标仿射；非平面仍不等于真实遮挡。"""
    inverse = torch.where(valid, truth.clamp_min(1e-8).reciprocal(), 0.)
    batch, _, height, width = truth.shape
    patches = F.unfold(inverse, 3, padding=1).reshape(batch, 9, height, width)
    legal = F.unfold(valid.float(), 3, padding=1).reshape_as(patches) > .5
    dx = truth.new_tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1])[None, :, None, None]
    dy = truth.new_tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1])[None, :, None, None]
    intercept = patches.mean(1, keepdim=True)
    slope_x = (patches * dx).sum(1, keepdim=True) / 6
    slope_y = (patches * dy).sum(1, keepdim=True) / 6
    residual = (patches - intercept - dx * slope_x - dy * slope_y).abs().amax(1, keepdim=True)
    residual = residual / inverse.clamp_min(1e-8)
    valid_fit = legal.all(1, keepdim=True)
    nonplanar = F.max_pool2d((valid_fit & (residual > .03)).float(), 5, stride=1, padding=2) > .5
    whole_band_valid = F.max_pool2d((~valid_fit).float(), 5, stride=1, padding=2) < .5
    return {
        'gradient_nonplanar': gradient_band & whole_band_valid & nonplanar,
        'gradient_planar': gradient_band & whole_band_valid & ~nonplanar,
        'gradient_unclassified': gradient_band & ~whole_band_valid,
    }


def upsample_diagnostics(output, logits, truth, valid, scale):
    fine = output['fine']
    depth = fine['depth'] * scale
    batch, _, height, width = depth.shape
    legal = F.unfold(fine['ref_valid'].float(), 3, padding=1)
    legal = legal.reshape(batch, 1, 9, 1, 1, height, width) > .5
    weights = safe_masked_softmax(logits.reshape(batch, 1, 9, 4, 4, height, width).float(), legal, dim=2)
    neighbors = F.unfold(depth, 3, padding=1).reshape(batch, 1, 9, 1, 1, height, width)
    full_truth = truth.reshape(batch, 1, height, 4, width, 4).permute(0, 1, 3, 5, 2, 4)
    closest = (neighbors - full_truth[:, :, None]).abs().masked_fill(~legal, torch.inf)
    neighbor_error = pack(closest.amin(2))
    inverse, low, high = convex_diagnostics(fine, logits, scale)
    convex_error = torch.maximum(low - truth, truth - high).clamp_min(0)
    ground = truth[..., ::4, ::4]
    ground_valid = valid[..., ::4, ::4]
    ground_neighbors = F.unfold(ground, 3, padding=1).reshape(batch, 1, 9, 1, 1, height, width)
    ground_legal = F.unfold(ground_valid.float(), 3, padding=1).reshape_as(legal) > .5
    ground_weights = weights * ground_legal
    denom = ground_weights.sum(2, keepdim=True)
    ground_weights = ground_weights / denom.clamp_min(1e-8)
    true_grid = pack((ground_weights * ground_neighbors).sum(2))
    has_ground = pack(ground_legal.any(2).expand(-1, -1, 4, 4, -1, -1))
    true_grid = torch.where(has_ground, true_grid, output['depth'] * scale)
    true_low = ground_neighbors.masked_fill(~ground_legal, torch.inf).amin(2)
    true_high = ground_neighbors.masked_fill(~ground_legal, -torch.inf).amax(2)
    true_low = pack(true_low.expand(-1, -1, 4, 4, -1, -1))
    true_high = pack(true_high.expand(-1, -1, 4, 4, -1, -1))
    true_error = torch.maximum(true_low - truth, truth - true_high).clamp_min(0)
    true_error = torch.where(has_ground, true_error, (output['depth'] * scale - truth).abs())
    return {
        'final': output['depth'] * scale,
        'nearest_block': depth.repeat_interleave(4, -2).repeat_interleave(4, -1),
        'inverse_same_weights': inverse,
        'oracle_neighbor_choice': truth + neighbor_error,
        'oracle_convex': truth + convex_error,
        'oracle_true_fine_same_weights': true_grid,
        'oracle_true_fine_convex': truth + true_error,
    }, int((valid & ~has_ground).sum())


def statistics(output, logits, row, domain):
    truth = row['targets']['depth'].float()
    valid = row['targets']['depth_valid'] & row['inputs']['image_mask'][:, :1]
    valid &= row['inputs']['view_mask'][:, :1, None, None]
    valid &= torch.isfinite(truth) & (truth > 0)
    scale = domain['depth_scale']
    masks = distance_masks(truth, valid, domain)
    boundaries = edge_groups(truth, valid)
    boundaries.update(local_plane_groups(truth, valid, boundaries['edge_5px']))
    variants, fallback_count = upsample_diagnostics(output, logits, truth, valid, scale)
    result = {'oracle_no_gt_neighbor_fallback_count': fallback_count, 'full': {}}
    for region, mask in masks.items():
        result['full'][region] = {
            part: {name: error_statistics(estimate, truth, mask & zone)
                   for name, estimate in variants.items()}
            for part, zone in {'all': valid, **boundaries}.items()
        }
    common_truth = truth[..., ::8, ::8]
    result['common_stride8'] = {}
    for region, mask in masks.items():
        mask8 = mask[..., ::8, ::8]
        estimates = {
            'coarse': output['coarse']['depth'] * scale,
            'fine': output['fine']['depth'][..., ::2, ::2] * scale,
            'final': output['depth'][..., ::8, ::8] * scale,
        }
        result['common_stride8'][region] = {
            name: error_statistics(estimate, common_truth, mask8)
            for name, estimate in estimates.items()
        }
    fine = output['fine']
    truth4 = truth[..., ::4, ::4]
    valid4 = valid[..., ::4, ::4]
    nearest = (fine['sampling_depth'].clamp_min(1e-8).log() - (truth4 / scale).clamp_min(1e-8).log()).abs()
    nearest = nearest.masked_fill(~fine['valid'], torch.inf).argmin(1, keepdim=True)
    support = fine['support'].gather(1, nearest)
    source_masks = {'all': valid4, 'gt_nearest_no_source': valid4 & (support == 0),
                    'gt_nearest_has_source': valid4 & (support > 0),
                    **{k: v[..., ::4, ::4] for k, v in boundaries.items()}}
    result['fine'] = {}
    for region, mask in masks.items():
        result['fine'][region] = {}
        for part, zone in source_masks.items():
            combined = mask[..., ::4, ::4] & zone
            result['fine'][region][part] = {
                'stages': stage_statistics(fine, truth4, combined, scale),
                'decomposition': fine_error_decomposition(fine, truth4, combined, scale),
            }
    return result


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    torch.manual_seed(20261027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    out = ROOT / 'runs/active/near_mid_20261027/diagnostics_plane'
    if out.exists():
        raise FileExistsError(out)
    if shutil.disk_usage(out.parent.resolve().parent).free < 204 * 2**30:
        raise RuntimeError('数据盘空间不足204GiB')
    out.mkdir(parents=True)
    checkpoint = ROOT / 'runs/active/public_refinement_20261025/recommended/pretrained.pt'
    index_path = ROOT / 'runs/active/architecture_screen_20261026/features/index.json'
    index = json.loads(index_path.read_text())
    assert digest(checkpoint) == index['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    decoder = CalibratedDecoder(saved['config']['model'])
    state = {k.removeprefix('decoder.'): v for k, v in saved['model'].items() if k.startswith('decoder.')}
    decoder.load_state_dict(state, strict=True)
    decoder.to(device).eval().requires_grad_(False)
    capture = {}
    hook = decoder.upsample_mask.register_forward_hook(lambda module, inputs, result: capture.update(logits=result))
    plan = {
        'checkpoint': str(checkpoint), 'sha256': digest(checkpoint), 'index_sha256': digest(index_path),
        'script_sha256': digest(__file__), 'precision': 'FP32 TF32off', 'device': 'cuda:1',
        'frames': 56, 'test_accessed': False, 'training': False,
        'regions': 'meters: [0,10),[10,40),[40,infinity); scene_unit: [0,.5),[.5,2),[2,infinity) times TRAIN median=10*depth_scale',
        'edges': 'both sides of neighboring valid GT log-depth jumps>.1, dilated5x5, diagnostic only',
        'plane_split': '3x3 least-squares affine inverseZ residual(max abs)/center inverseZ>.03 then dilation5x5; gradient_planar/nonplanar only if whole5x5 band has9 valid GT neighbors, else unclassified. Not an occlusion certificate.',
        'oracle': 'GT hindsight not deployable. Convex interval can interpolate across unrelated surfaces; optimistic representation bound only.',
        'source': 'GT-nearest fine sample geometric sampling support, not occlusion/correspondence certification',
        'nearest_block': 'repeats fine site into its 4x4 output block, aligned with convex-packing block ownership',
    }
    write_json(out / 'protocol.json', plan)
    summary = {}
    start = time.monotonic()
    for domain in index['domains']:
        decoder.set_range(domain['near'] / domain['depth_scale'], domain['far'] / domain['depth_scale'])
        records = []
        for path in index['splits']['development'][domain['name']]:
            row = move(torch.load(ROOT / path, map_location='cpu', weights_only=False), device)
            prediction = decoder(row['work'])
            data = statistics(prediction, capture.pop('logits'), row, domain)
            records.append({'cache': path, 'statistics': data})
        values = summarize(sum_records([r['statistics'] for r in records]))
        summary[domain['name']] = {'units': domain['geometry_units'], 'depth_scale': domain['depth_scale'], 'statistics': values}
        write_json(out / (domain['name'] + '.json'), {'records': records, 'summary': summary[domain['name']]})
        print(domain['name'], 'done', flush=True)
    hook.remove()
    write_json(out / 'summary.json', summary)
    write_json(out / 'complete.json', {'frames': 56, 'elapsed_s': time.monotonic() - start, 'test_accessed': False})
    shutil.copy2(__file__, out / 'diagnose.py')


if __name__ == '__main__':
    run()
