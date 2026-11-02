"""冻结V6的表面/纹理/可观测性分层：ETH历史外测诊断、CARLA固定VAL。"""

import copy
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import add_statistics, digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from research.rulermvs_v6_20261101.diagnose import candidate_statistics, cpu_prediction
from research.rulermvs_v6_20261101.evaluation import Models
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import offline_only

BASE = ROOT / 'runs/active/rulermvs_v6_20261101/full_training'
OUTPUT = ROOT / 'runs/active/v7_domain_20261102/diagnosis'


def strata(batch, truth, mask, domain):
    """GT用于事后分层，不进入推理；边缘和灰度梯度不冒充语义标签。"""
    inputs = batch['inputs']
    reference = domain['reference_index']
    rgb = inputs['rgb'][:, reference].double()
    gray = (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    local_mean = F.avg_pool2d(gray, 5, stride=1, padding=2)
    variance = (F.avg_pool2d(gray.square(), 5, stride=1, padding=2) - local_mean.square()).clamp_min(0)
    texture = variance.sqrt()
    result = regions(truth, mask, domain)
    result.update(texture_low=mask & (texture < .02), texture_mid=mask & (texture >= .02) & (texture < .08),
                  texture_high=mask & (texture >= .08))
    height, width = truth.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    pixels = torch.stack([x, y, torch.ones_like(x)], 0).reshape(3, -1).double()
    intrinsic = inputs['K'][0].double()
    transforms = inputs['T_cam_from_rig'][0].double()
    rays = torch.linalg.solve(intrinsic[reference], pixels)
    z = truth.reshape(-1)
    sensitivity = torch.zeros_like(z)
    in_view = torch.zeros_like(z, dtype=torch.bool)
    observed = torch.zeros_like(z, dtype=torch.bool)
    occluded = torch.zeros_like(z, dtype=torch.bool)
    baseline = []
    inverse_reference = torch.linalg.inv(transforms[reference])
    for source in range(len(transforms)):
        if source == reference or not inputs['view_mask'][0, source]:
            continue
        relative = transforms[source] @ inverse_reference
        baseline.append(float(relative[:3, 3].norm()))
        direction = relative[:3, :3] @ rays
        xyz = direction * z + relative[:3, 3:4]
        homogeneous = intrinsic[source] @ xyz
        derivative = intrinsic[source] @ direction
        denominator = homogeneous[2].clamp_min(1e-12)
        uv = homogeneous[:2] / denominator
        inside = (xyz[2] > 0) & (uv[0] >= 0) & (uv[0] <= width - 1) & (uv[1] >= 0) & (uv[1] <= height - 1)
        # d(pixel)/d(log Z): 对平行双目恰等于视差幅值 fB/Z。
        jacobian = (derivative[:2] * denominator - homogeneous[:2] * derivative[2]) / denominator.square()
        current = (jacobian * z).norm(dim=0)
        sensitivity = torch.maximum(sensitivity, torch.where(inside, current, 0))
        in_view |= inside
        grid = torch.stack([2 * (uv[0] + .5) / width - 1, 2 * (uv[1] + .5) / height - 1], -1).reshape(1, height, width, 2)
        source_valid = F.grid_sample(batch['targets']['depth_valid'][:, source:source + 1].double(), grid,
                                     mode='nearest', align_corners=False).reshape(-1) > .5
        source_z = F.grid_sample(batch['targets']['depth'][:, source:source + 1].double(), grid,
                                 mode='nearest', align_corners=False).reshape(-1)
        known = inside & source_valid
        observed |= known & ((source_z - xyz[2]).abs() <= .05 * xyz[2])
        occluded |= known & (source_z < .95 * xyz[2])
    sensitivity = sensitivity.reshape_as(truth)
    in_view = in_view.reshape_as(mask)
    result.update(no_source_projection=mask & ~in_view,
                  source_projection=mask & in_view,
                  gt_source_consistent=mask & observed.reshape_as(mask),
                  gt_source_occluded=mask & occluded.reshape_as(mask))
    for label, low, high in [('lt05', 0, .5), ('05to2', .5, 2), ('2to8', 2, 8), ('ge8', 8, float('inf'))]:
        result['parallax_' + label] = mask & in_view & (sensitivity >= low) & (sensitivity < high)
    return result, {'baseline_m': baseline,
                    'sensitivity_quantiles_px_per_logZ': torch.quantile(sensitivity[mask], torch.tensor([.1, .5, .9], dtype=torch.double)).tolist(),
                    'texture_std_quantiles': torch.quantile(texture[mask], torch.tensor([.1, .5, .9], dtype=torch.double)).tolist()}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not OUTPUT.resolve().is_relative_to('/data'):
        raise ValueError('诊断输出必须位于数据盘')
    OUTPUT.mkdir(parents=True)
    offline_only()
    eth = json.loads((BASE / 'paper_eth3d/evaluation.json').read_text())
    carla = json.loads((BASE / 'evaluation_recovery/configs/carla_new_main_val.json').read_text())
    eth['datasets'] = eth['datasets'][:1]
    selected = json.loads((ROOT / 'runs/active/rulermvs_v6_20261101/diagnostics/full/summary.json').read_text())['plans']
    report = {'completed': False, 'source_sha256': digest(__file__), 'datasets': {},
              'notes': ['固定已发布V6及MVS权重，无训练、无参数搜索。',
                        'ETH27对为既有TEST事后诊断；CARLA各参考使用原先16个VAL索引。',
                        '梯度带=相邻log深度差>0.1并5x5膨胀；不是非平面或植被的语义真值。',
                        '纹理=RGB亮度5x5局部标准差，边界采用零填充；阈值0.02/0.08事先固定。',
                        'parallax是给定GT下最佳源的|d(pixel)/dlogZ|，不等于匹配一定可靠。',
                        '源GT一致和遮挡可重叠（不同源），缺源GT不等于遮挡；ETH只有参考GT。',
                        '候选oracle均是事后诊断，不是部署精度。']}
    write_json(OUTPUT / 'plan.json', report)
    for config, split in [(eth, 'test'), (carla, 'val')]:
        config['methods'] = {k: config['methods'][k] for k in ['v6_best', 'mvsanywhere_refined']}
        first = copy.deepcopy(config['methods']['mvsanywhere_refined'])
        first['refinements'] = 0
        config['methods']['mvsanywhere_firstpass'] = first
        device = configure(config, 0)
        models = Models(config, device)
        for domain in config['datasets']:
            dataset = build_dataset(dataset_config(domain, config), split, load_targets=True)
            indices = list(range(len(dataset))) if split == 'test' else selected[domain['name']]['indices']
            record = {'split': split, 'indices': indices, 'frames': [], 'totals': {}, 'candidate_sums': {},
                      'manifest_sha256': digest(domain['manifests'][split]),
                      'weights_sha256': {m: digest(s['weights']) for m, s in config['methods'].items()}}
            manifest = [json.loads(line) for line in Path(domain['manifests'][split]).read_text().splitlines()]
            for index in indices:
                batch = collate_frames([dataset[index]])
                truth, mask = ground_truth(batch, domain)
                masks, metadata = strata(batch, truth, mask, domain)
                inputs = move(equal_pixel_inputs(batch['inputs']), device)
                row = {'index': index, 'sample_id': manifest[index]['frame_id'], 'sequence': manifest[index]['sequence'],
                       'geometry': metadata, 'methods': {}}
                with torch.inference_mode():
                    for method in config['methods']:
                        if method == 'v6_best':
                            raw = models.get(method, domain)(inputs, domain, reference=domain['reference_index'])
                            native = cpu_prediction(raw)
                            scale = float(raw['unit_scale'])
                            prediction = {'depth': raw['depth'] * raw['unit_scale'], 'valid': raw['valid']}
                            diagnostics = candidate_statistics(native, truth, scale)
                            stages = {'coarse': native['coarse'] * scale,
                                      'fine': native['fine']['depth'][..., ::2, ::2] * scale}
                            del raw
                        else:
                            prediction = models.forward(method, domain, inputs)
                            stages = {}
                        depth, valid = models.convert(prediction, inputs)
                        depth, valid = depth.cpu().double(), valid.cpu()
                        values = {name: depth_sums(depth, valid, truth, selected_mask) for name, selected_mask in masks.items()}
                        row['methods'][method] = values
                        for name, sums in values.items():
                            add_statistics(record['totals'].setdefault(method, {}).setdefault(name, {}), sums)
                        if method == 'v6_best':
                            stages['final'] = depth[..., ::8, ::8]
                            for name, selected_mask in masks.items():
                                small = selected_mask[..., ::8, ::8]
                                stats = {'pixels': int(small.sum())}
                                for key, values_ in diagnostics.items():
                                    stats[key] = float(values_[small].double().sum())
                                for stage, values_ in stages.items():
                                    stats[stage + '_stride8_MAE'] = float((values_ - truth[..., ::8, ::8]).abs()[small].sum())
                                add_statistics(record['candidate_sums'].setdefault(name, {}), stats)
                        del prediction, depth, valid
                record['frames'].append(row)
                print(json.dumps({'dataset': domain['name'], 'index': index, 'done': len(record['frames'])}), flush=True)
            record['metrics'] = {method: {name: depth_metrics(sums) for name, sums in values.items()}
                                 for method, values in record['totals'].items()}
            record['candidates'] = {name: {key: value / sums['pixels'] if sums['pixels'] else None
                                          for key, value in sums.items() if key != 'pixels'}
                                     for name, sums in record['candidate_sums'].items()}
            report['datasets'][domain['name']] = record
            write_json(OUTPUT / 'results.json', report)
        models.loaded.clear()
        del models
        gc.collect()
        torch.cuda.empty_cache()
    report['completed'] = True
    write_json(OUTPUT / 'results.json', report)


if __name__ == '__main__':
    main()
