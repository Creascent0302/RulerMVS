"""固定11域各16帧VAL，检查部署优化对相同权重深度与指标的影响，不测速度。"""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import add_statistics, digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from research.rulermvs_v6_20261101.diagnose import SOURCES
from research.rulermvs_v6_20261101.speed_final import METHODS, initialize, load


def compare(fast, eager, scale):
    a, b = fast['depth'] * scale, eager['depth'] * scale
    difference = (a - b).abs()
    relative = difference / b.abs().clamp_min(1e-8)
    return {
        'finite_depth': bool(torch.isfinite(a).all()),
        'valid_identical': bool(torch.equal(fast['valid'], eager['valid'])),
        'depth_max_difference': float(difference.max()),
        'depth_mean_difference': float(difference.mean()),
        'depth_relative_difference_gt_1e_minus4_fraction': float((relative > 1e-4).float().mean()),
        'coarse_changed_fraction': float((fast['coarse']['selected_index'] != eager['coarse']['selected_index']).float().mean()),
        'fine_changed_fraction': float((fast['fine']['selected_index'] != eager['fine']['selected_index']).float().mean()),
        'fine_group_changed_fraction': float(((fast['fine']['selected_index'] // 3)
                                              != (eager['fine']['selected_index'] // 3)).float().mean()),
        'coarse_depth_max_difference': float((fast['coarse']['depth'] - eager['coarse']['depth']).abs().max()) * scale,
        'fine_depth_max_difference': float((fast['fine']['depth'] - eager['fine']['depth']).abs().max()) * scale,
    }


def main(settings):
    initialize(settings)
    output = Path(settings['output'])
    if output.exists() and any(output.iterdir()) and not settings.get('resume', False):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    plans = json.loads(Path(settings['diagnostics']).read_text())['plans']
    report = {'completed': False, 'test_accessed': False, 'new_training': False,
              'settings': settings, 'precision': 'FP32/TF32off', 'input_mode': 'equal_pixels',
              'datasets': {}, 'weights_sha256': {}, 'notes': [
                  '只对相同权重的eager与fast比较；不是V6结构收益实验。',
                  '复用诊断预登记各16个VAL索引，单位与有效GT掩码不变。',
                  '误差差值含融合代数重排及局部编译；不把局部编译近似称严格无损。']}
    if settings.get('resume') and (output / 'results.json').exists():
        previous = json.loads((output / 'results.json').read_text())
        if dict(previous['settings'], resume=False) != dict(settings, resume=False):
            raise ValueError('续跑必须保留原权重、数据、索引和部署选项')
        report = previous
        report['settings'] = settings
        report['notes'].append('首次在Hypersim无远距GT区域汇总None差值失败；修复为空后续跑，保留已完成域。')
    device = 'cuda:' + str(settings.get('gpu', 0))
    with torch.inference_mode():
        for phase, source in SOURCES.items():
            config = copy.deepcopy(json.loads(source.read_text())['config'])
            phase_settings = dict(settings, **settings['phases'][phase])
            models = {name: load(phase_settings, name) for name in METHODS}
            hashes = {v: digest(phase_settings[v + '_checkpoint']) for v in ('v5', 'v6')}
            if phase in report['weights_sha256'] and hashes != report['weights_sha256'][phase]:
                raise ValueError('续跑时权重发生变化')
            report['weights_sha256'][phase] = hashes
            for domain in config['datasets']:
                name = domain['name']
                plan = plans[name]
                if name in report['datasets']:
                    assert report['datasets'][name]['indices'] == plan['indices']
                    assert report['datasets'][name]['manifest_sha256'] == digest(domain['manifests']['val'])
                    continue
                dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=True)
                assert digest(domain['manifests']['val']) == plan['manifest_sha256']
                totals = {method: {} for method in METHODS}
                checks = []
                started = time.perf_counter()
                for index in plan['indices']:
                    batch = collate_frames([dataset[index]])
                    inputs = equal_pixel_inputs(move(batch['inputs'], device))
                    predictions = {method: model(inputs, domain, domain['reference_index'])
                                   for method, model in models.items()}
                    for version in ('v5', 'v6'):
                        checks.append({'index': index, 'version': version,
                                       **compare(predictions[version + '_fast'], predictions[version + '_eager'],
                                                 domain['depth_scale'])})
                    truth, mask = ground_truth(batch, domain)
                    masks = regions(truth, mask, domain)
                    for method, prediction in predictions.items():
                        depth = prediction['depth'].cpu() * domain['depth_scale']
                        valid = prediction['valid'].cpu()
                        for region, selected in masks.items():
                            add_statistics(totals[method].setdefault(region, {}),
                                           depth_sums(depth, valid, truth, selected))
                    del inputs, predictions, batch, prediction
                metrics = {method: {region: depth_metrics(values) for region, values in rows.items()}
                           for method, rows in totals.items()}
                differences = {}
                for version in ('v5', 'v6'):
                    differences[version] = {region: {
                        key: (metrics[version + '_fast'][region][key] - metrics[version + '_eager'][region][key]
                              if metrics[version + '_eager'][region][key] is not None else None)
                        for key in ('MAE', 'RMSE', 'AbsRel')} for region in metrics[version + '_eager']}
                report['datasets'][name] = {
                    'frames': len(plan['indices']), 'indices': plan['indices'],
                    'units': domain['geometry_units'], 'num_views': domain['num_views'],
                    'manifest_sha256': plan['manifest_sha256'], 'depth': metrics,
                    'fast_minus_eager_metrics': differences, 'numerical_checks': checks,
                    'seconds_including_compilation_and_data': time.perf_counter() - started,
                }
                write_json(output / 'results.json', report)
                print(json.dumps({'completed_domain': name,
                                  'full_metric_changes': {v: differences[v]['full'] for v in ('v5', 'v6')}},
                                 ensure_ascii=False), flush=True)
            model = None
            models.clear()
            del models
            gc.collect()
            torch.cuda.empty_cache()
    report['completed'] = True
    report['source_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in (
        Path(__file__), ROOT / 'research/rulermvs_v6_20261101/inference.py')}
    write_json(output / 'results.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    main(json.loads(Path(arguments.config).read_text()))
