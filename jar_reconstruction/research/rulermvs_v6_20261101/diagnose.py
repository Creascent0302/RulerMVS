"""固定V5权重的VAL候选反事实诊断；不训练，不读取TEST，不测速度。"""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import add_statistics, digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.pretraining.inference import prepare_inference
from jar.runtime import move, seed_all
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from research.rulermvs_v6_20261101.handoff import masked_probability
from research.rulermvs_v6_20261101.model import build_from_checkpoint
from tools.smoke_outdoor_datasets import offline_only

VARIANTS = ('v5_winner', 'v5_mode_mean', 'v6_marginal', 'v6_parent')
SOURCES = {
    'public': ROOT / 'runs/active/rulermvs_v5_20261031/evaluation/public_equal_pixels/full_test/audit.json',
    'carla': ROOT / 'runs/active/baseline_full_comparison_20261101/carla_four_references/full_test/audit.json',
}
DESTINATION = ROOT / 'runs/active/rulermvs_v6_20261101/diagnostics'


def cpu_prediction(prediction):
    """只保存统计所需小张量；图像特征不落盘。"""
    result = {key: prediction[key].detach().cpu() for key in ('depth', 'valid')}
    result['coarse'] = prediction['coarse']['depth'].detach().cpu()
    result['fine'] = {key: value.detach().cpu() for key, value in prediction['fine'].items()
                      if key in ('scores', 'valid', 'depth', 'candidate_depth', 'sampling_depth',
                                 'selected_index', 'handoff_gate')}
    return result


def candidate_statistics(prediction, truth, scale):
    """全部在前向完成后计算，oracle不会反馈候选构造或读出。"""
    fine = prediction['fine']
    target = truth[:, :, ::8, ::8]
    coarse = prediction['coarse'] * scale
    actual = fine['depth'][..., ::2, ::2] * scale
    scores = fine['scores'][..., ::2, ::2].float()
    valid = fine['valid'][..., ::2, ::2]
    candidates = fine['candidate_depth'][..., ::2, ::2] * scale
    batch, count, height, width = scores.shape
    grouped_valid = valid.reshape(batch, count // 3, 3, height, width)
    conditional = masked_probability(scores.reshape_as(grouped_valid), grouped_valid, 2)
    group_depth = (conditional * candidates.reshape_as(grouped_valid)).sum(2)
    group_valid = grouped_valid.any(2)
    errors = (candidates - target).abs().masked_fill(~valid, torch.inf)
    group_errors = (group_depth - target).abs().masked_fill(~group_valid, torch.inf)
    continuous_oracle = errors.amin(1, keepdim=True)
    group_oracle = group_errors.amin(1, keepdim=True)
    sampling = fine['sampling_depth'][..., ::2, ::2] * scale
    sampling_oracle = (sampling - target).abs().masked_fill(~valid, torch.inf).amin(1, keepdim=True)
    selected = fine['selected_index'][..., ::2, ::2] // 3
    selected_group_error = group_errors.gather(1, selected)
    coarse_error = (coarse - target).abs()
    fine_error = (actual - target).abs()
    tolerance = .05 * target
    coarse_good = coarse_error <= tolerance
    fine_good = fine_error <= tolerance
    values = {
        'coarse_MAE': coarse_error,
        'fine_MAE': fine_error,
        'continuous_candidate_oracle_MAE': continuous_oracle,
        'sampling_candidate_oracle_MAE': sampling_oracle,
        'group_mean_oracle_MAE': group_oracle,
        'selected_group_mean_MAE': selected_group_error,
        'group_selection_regret': selected_group_error - group_oracle,
        'continuous_coverage_5pct': (continuous_oracle <= tolerance).float(),
        'group_mean_coverage_5pct': (group_oracle <= tolerance).float(),
        'coarse_good': coarse_good.float(),
        'fine_good': fine_good.float(),
        'coarse_good_fine_bad': (coarse_good & ~fine_good).float(),
        'coarse_bad_fine_good': (~coarse_good & fine_good).float(),
    }
    if 'handoff_gate' in fine:
        values['parent_gate'] = fine['handoff_gate'][..., ::2, ::2]
    return values


def update_record(record, prediction, truth, masks, scale, down_camera):
    record['frames'] = record.get('frames', 0) + 1
    depth = prediction['depth'] * scale
    for region, mask in masks.items():
        sums = depth_sums(depth, prediction['valid'], truth, mask)
        add_statistics(record.setdefault('depth', {}).setdefault(region, {}), sums)
    stages = {'coarse': prediction['coarse'] * scale,
              'fine': prediction['fine']['depth'][..., ::2, ::2] * scale,
              'final': depth[..., ::8, ::8]}
    for stage, value in stages.items():
        for region, mask in masks.items():
            sums = depth_sums(value, torch.isfinite(value) & (value > 0),
                              truth[..., ::8, ::8], mask[..., ::8, ::8])
            target = record.setdefault('stages_stride8', {}).setdefault(stage, {}).setdefault(region, {})
            add_statistics(target, sums)
    diagnostics = candidate_statistics(prediction, truth, scale)
    for region, mask in masks.items():
        mask = mask[..., ::8, ::8]
        if any(not bool(value[mask].isfinite().all()) for value in diagnostics.values()):
            raise RuntimeError('GT有效格点上候选诊断非有限，拒绝静默删掉失败样本')
        target = record.setdefault('candidates_stride8', {}).setdefault(region, {})
        add_statistics(target, {'pixels': int(mask.sum()),
                                **{key: float(value[mask].double().sum()) for key, value in diagnostics.items()}})
    if down_camera:
        for py in range(8):
            for px in range(8):
                offset = (..., slice(py, None, 8), slice(px, None, 8))
                sums = depth_sums(depth[offset], prediction['valid'][offset], truth[offset], masks['full'][offset])
                add_statistics(record.setdefault('final_pixel_phases', {}).setdefault(f'{py},{px}', {}), sums)


def finish(record):
    result = {'frames': record['frames'],
              'depth': {k: depth_metrics(v) for k, v in record['depth'].items()},
              'stages_stride8': {stage: {k: depth_metrics(v) for k, v in values.items()}
                                 for stage, values in record['stages_stride8'].items()}}
    result['candidates_stride8'] = {}
    for region, values in record['candidates_stride8'].items():
        count = values['pixels']
        item = {key: value / count if count else None for key, value in values.items() if key != 'pixels'}
        item['pixels'] = count
        item['coarse_good_destroyed_fraction'] = (values['coarse_good_fine_bad'] / values['coarse_good']
                                                 if values['coarse_good'] else None)
        result['candidates_stride8'][region] = item
    if 'final_pixel_phases' in record:
        result['final_pixel_phases'] = {k: depth_metrics(v) for k, v in record['final_pixel_phases'].items()}
    return result


def load_models(weights, device):
    saved = torch.load(weights, map_location='cpu', weights_only=False, mmap=True)
    models = {}
    for variant in ('v5', 'marginal', 'parent'):
        model, _ = build_from_checkpoint(saved, variant)
        model = model.to(device).eval().requires_grad_(False)
        install_position_cache(model.foundation.model)
        prepare_inference(model, spatial_layout=True)
        models[variant] = model
    if models['v5'].decoder.fine_readout != 'winner':
        raise ValueError('本诊断预登记的V5对照必须是原winner读出')
    return models


@torch.inference_mode()
def evaluate_domain(models, config, domain, plan, device, output):
    dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=True)
    loader = DataLoader(dataset, batch_size=1, sampler=plan['indices'], num_workers=0,
                        collate_fn=collate_frames)
    totals = {partition: {variant: {} for variant in VARIANTS}
              for partition in ('all', 'development', 'confirmation')}
    frame_records = []
    scale = float(domain['depth_scale'])
    max_coarse_difference = 0.
    for model in models.values():
        model.decoder.set_range(domain['near'] / scale, domain['far'] / scale)
    for index, batch in zip(plan['indices'], loader):
        # 只把inputs交给编码器和解码器；targets保留CPU，直到模型返回才统计。
        inputs = equal_pixel_inputs(move(batch['inputs'], device))
        work = models['v5'].encode(inputs, domain, domain['reference_index'])
        predictions = {}
        for variant in VARIANTS:
            model = models['v5'] if variant.startswith('v5_') else models[variant.removeprefix('v6_')]
            if variant.startswith('v5_'):
                model.decoder.fine_readout = 'mode_mean' if variant == 'v5_mode_mean' else 'winner'
            predictions[variant] = cpu_prediction(model.decoder(work))
        models['v5'].decoder.fine_readout = 'winner'
        del work, inputs
        before = predictions['v5_winner']
        for variant, prediction in predictions.items():
            max_coarse_difference = max(max_coarse_difference,
                                        float((prediction['coarse'] - before['coarse']).abs().max()))
            torch.testing.assert_close(prediction['coarse'], before['coarse'], rtol=0, atol=0)
            torch.testing.assert_close(prediction['fine']['candidate_depth'], before['fine']['candidate_depth'],
                                       rtol=0, atol=0)
        truth, mask = ground_truth(batch, domain)
        masks = regions(truth, mask, domain)
        partition = 'development' if index in plan['development'] else 'confirmation'
        per_frame = {'index': index, 'partition': partition, 'methods': {}}
        for variant, prediction in predictions.items():
            record = {}
            update_record(record, prediction, truth, masks, scale, domain['name'].endswith('stereo_down'))
            per_frame['methods'][variant] = finish(record)
            for name in ('all', partition):
                update_record(totals[name][variant], prediction, truth, masks, scale,
                              domain['name'].endswith('stereo_down'))
        frame_records.append(per_frame)
    result = {'completed': True, 'dataset': domain['name'], 'units': domain['geometry_units'],
              'frames': len(plan['indices']), 'coarse_max_difference': max_coarse_difference,
              'partitions': {p: {v: finish(r) for v, r in values.items() if r}
                             for p, values in totals.items()}, 'per_frame': frame_records}
    write_json(output / (domain['name'] + '.json'), result)
    return {key: value for key, value in result.items() if key != 'per_frame'}


def run(smoke=False, device='cuda:1'):
    offline_only()
    seed_all(20261101)
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = DESTINATION / ('smoke' if smoke else 'full')
    output.mkdir(parents=True, exist_ok=True)
    report = {'completed': False, 'test_accessed': False, 'new_training': False,
              'device': str(device), 'input_mode': 'equal_pixels', 'variants': list(VARIANTS),
              'notes': ['V5默认winner；mode_mean仅改变winner所属组三点读出。',
                        'V6只改粗细质量交接与组读出，所有变体共享同V5编码和候选。',
                        '当前固定权重反事实不等于训练后结构上限；此处不测时间。',
                        '组oracle是每组三点加权均值的oracle，非可部署真值选择。'],
              'plans': {}, 'datasets': {}}
    for phase, source in SOURCES.items():
        config = copy.deepcopy(json.loads(source.read_text())['config'])
        weights = config['methods']['v5_best']['weights']
        config['datasets'] = config['datasets'][:1] if smoke else config['datasets']
        for domain in config['datasets']:
            domain['manifests'] = {'val': domain['manifests']['val']}
            manifest = Path(domain['manifests']['val'])
            length = sum(bool(line.strip()) for line in manifest.read_text().splitlines())
            indices = torch.linspace(0, length - 1, min(1 if smoke else 16, length)).long().tolist()
            report['plans'][domain['name']] = {'indices': indices, 'development': indices[::2],
                                              'confirmation': indices[1::2],
                                              'manifest_sha256': digest(manifest), 'weights': weights,
                                              'weights_sha256': digest(weights), 'domain': domain}
        write_json(output / 'summary.json', report)
        models = load_models(weights, device)
        for domain in config['datasets']:
            name = domain['name']
            result = evaluate_domain(models, config, domain, report['plans'][name], device, output)
            report['datasets'][name] = result
            write_json(output / 'summary.json', report)
            print(json.dumps({'completed_domain': name, 'frames': result['frames'],
                              'MAE': {v: r['depth']['full']['MAE']
                                      for v, r in result['partitions']['all'].items()}}, ensure_ascii=False), flush=True)
        del models
        gc.collect()
        torch.cuda.empty_cache()
    report['completed'] = True
    report['source_sha256'] = {str(path.relative_to(ROOT)): digest(path)
                               for path in Path(__file__).parent.glob('*.py')}
    write_json(output / 'summary.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda:1')
    options = parser.parse_args()
    try:
        run(options.smoke, options.device)
    except Exception:
        output = DESTINATION / ('smoke' if options.smoke else 'full')
        write_json(output / 'failure.json', {'completed': False, 'traceback': traceback.format_exc()})
        raise
