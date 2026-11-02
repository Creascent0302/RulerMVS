"""冻结短训末步的公共VAL压力测试：灰度与最近真实源，不访问ETH。"""

import copy
import csv
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import add_statistics, digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.pretraining.inference import prepare_inference
from jar.runtime import move
from research.near_evidence_20261030.pilot import parameter_hash
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from research.v7_domain_20261102.model import build_model
from tools.smoke_outdoor_datasets import fingerprint, offline_only

BASE = ROOT / 'runs/active/v7_domain_20261102'
OUTPUT = BASE / 'stress_validation'
METHODS = ('v6_frozen', 'control_original', 'control_augment', 'prior_original', 'prior_augment')
CONDITIONS = ('clean', 'grayscale', 'closest_real_pair', 'gray_closest_pair')


def changed_input(inputs, reference, condition):
    """只读输入选真实相机，K/T不变，不复制图像或使用GT挑源。"""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    batch, views = inputs['view_mask'].shape
    retained = inputs['view_mask'].clone()
    selected = []
    for sample in range(batch):
        transforms = inputs['T_cam_from_rig'][sample].double()
        rotation = transforms[:, :3, :3]
        centers = -(rotation.transpose(-1, -2) @ transforms[:, :3, 3:4])[..., 0]
        distance = torch.linalg.vector_norm(centers - centers[reference], dim=-1)
        sources = [i for i in range(views) if i != reference and bool(retained[sample, i])]
        nonzero = [i for i in sources if float(distance[i]) > 1e-6]
        chosen = min(nonzero, key=lambda i: (float(distance[i]), i)) if nonzero else None
        if condition in ('closest_real_pair', 'gray_closest_pair') and chosen is not None:
            retained[sample] = False
            retained[sample, reference] = inputs['view_mask'][sample, reference]
            retained[sample, chosen] = True
        selected.append({'closest_source': chosen, 'original_source_count': len(sources),
                         'source_baselines_original_units': {str(i): float(distance[i]) for i in sources},
                         'pure_rotation_only_kept_existing': chosen is None and bool(sources)})
    result = dict(inputs, view_mask=retained)
    grayscale = condition in ('grayscale', 'gray_closest_pair')
    for key in ('rgb', 'matching_rgb'):
        if key not in inputs:
            continue
        value = inputs[key]
        if grayscale:
            weights = value.new_tensor([.299, .587, .114])[None, None, :, None, None]
            value = (value * weights).sum(2, keepdim=True).expand_as(value)
        result[key] = value * retained[:, :, None, None, None]
    for key in ('image_mask', 'matching_image_mask', 'camera_valid'):
        if key in inputs:
            value = inputs[key]
            shape = (*retained.shape, *([1] * (value.ndim - 2)))
            result[key] = value & retained.reshape(shape)
    if 'image_sizes' in inputs:
        result['image_sizes'] = inputs['image_sizes'] * retained[:, :, None]
    # 保持全部方法同一主图信息；不从原生高分辨率matching图取额外纹理。
    result = equal_pixel_inputs(result)
    return result, selected


def tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    offline_only()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'
    config = json.loads((ROOT / 'runs/active/rulermvs_v6_20261101/full_training/evaluation_recovery/configs/public_main_val.json').read_text())
    plans = json.loads((BASE / 'pilots_round1/control_original/public/plan.json').read_text())['validation']
    paths = {name: BASE / 'pilots_round1' / name / 'public/final.pt' for name in METHODS[1:]}
    paths['v6_frozen'] = Path(config['methods']['v6_best']['weights'])
    models, encoder_hashes = {}, {}
    for name in METHODS:
        saved = torch.load(paths[name], map_location='cpu', weights_only=False, mmap=True)
        model, _ = build_model(saved)
        encoder_hashes[name] = parameter_hash({k: v for k, v in saved['model'].items() if not k.startswith('decoder.')})
        model = model.to(device).eval().requires_grad_(False)
        install_position_cache(model.foundation.model)
        prepare_inference(model, spatial_layout=True)
        models[name] = model
    del saved, model
    if len(set(encoder_hashes.values())) != 1:
        raise RuntimeError('编码器参数/buffers不同，不允许共享编码')
    sources = [Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).with_name('prior.py')]
    report = {'completed': False, 'split': 'val', 'test_accessed': False, 'ETH_accessed': False,
              'conditions': CONDITIONS, 'datasets': {}, 'encoder_sha256': encoder_hashes,
              'weights_sha256': {name: digest(path) for name, path in paths.items()},
              'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in sources},
              'shared_encoding_full_forward_checks': 0,
              'notes': ['五模型同条件RGB/标定、GT、范围、有效掩码一致；参考槽不变。',
                        'closest_pair只挑现存有效相机中非零最小平移；不构造新图、不改变K/T。',
                        '相机中心距离是低视差压力的proxy，不等于得到新窄基线传感器；视角/场景结构也影响视差。',
                        '原双目条件中pair操作可能恒等，明确保留结果而不假称新证据。',
                        '每域16固定VAL；指标单位逐域保留，不把scene_unit称米；不按ETH选参数。',
                        '参数和buffers完整编码hash相同；每域首帧每条件5模型各一次完整forward逐位核对。']}
    write_json(OUTPUT / 'plan.json', report)
    with torch.inference_mode():
        for domain in config['datasets']:
            name = domain['name']
            plan = plans[name]
            assert digest(domain['manifests']['val']) == plan['manifest_sha256']
            dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=True)
            record = {'units': domain['geometry_units'], 'indices': plan['indices'],
                      'manifest_sha256': plan['manifest_sha256'], 'frames': [], 'totals': {}}
            scale = float(domain['depth_scale'])
            for model in models.values():
                model.decoder.set_range(domain['near'] / scale, domain['far'] / scale)
            for frame_number, index in enumerate(plan['indices']):
                batch = collate_frames([dataset[index]])
                truth, mask = ground_truth(batch, domain)
                masks = regions(truth, mask, domain)
                original_hash = fingerprint(batch['inputs'])
                row = {'index': index, 'conditions': {}}
                for condition in CONDITIONS:
                    cpu, selections = changed_input(batch['inputs'], domain['reference_index'], condition)
                    assert torch.equal(cpu['K'], batch['inputs']['K'])
                    assert torch.equal(cpu['T_cam_from_rig'], batch['inputs']['T_cam_from_rig'])
                    inputs = move(cpu, device)
                    encoded = models['v6_frozen'].encode(inputs, domain, domain['reference_index'])
                    originals = [(tensor, tensor.clone()) for tensor in tensors(encoded)]
                    item = {'valid_views': int(cpu['view_mask'].sum()), 'selections': selections,
                            'input_sha256': fingerprint(cpu), 'methods': {}}
                    for method in METHODS:
                        raw = models[method].decoder(encoded)
                        if frame_number == 0:
                            independent = models[method](inputs, domain, domain['reference_index'])
                            for key in ('depth', 'valid'):
                                assert torch.equal(raw[key], independent[key]), (name, condition, method, key)
                            for stage in ('coarse', 'fine'):
                                assert torch.equal(raw[stage]['depth'], independent[stage]['depth']), (name, condition, method, stage)
                            report['shared_encoding_full_forward_checks'] += 1
                            del independent
                        depth, valid = raw['depth'].cpu().double() * scale, raw['valid'].cpu()
                        values = {region: depth_sums(depth, valid, truth, selected) for region, selected in masks.items()}
                        stages = {stage: depth_sums(raw[stage]['depth'].cpu().double()[..., ::step, ::step] * scale,
                                                   raw[stage]['ref_valid'].cpu()[..., ::step, ::step],
                                                   truth[..., ::8, ::8], mask[..., ::8, ::8])
                                  for stage, step in [('coarse', 1), ('fine', 2)]}
                        item['methods'][method] = {'depth': values, 'stages': stages}
                        destination = record['totals'].setdefault(condition, {}).setdefault(method, {'depth': {}, 'stages': {}})
                        for region, value in values.items():
                            add_statistics(destination['depth'].setdefault(region, {}), value)
                        for stage, value in stages.items():
                            add_statistics(destination['stages'].setdefault(stage, {}), value)
                        del raw, depth, valid
                    assert all(torch.equal(tensor, original) for tensor, original in originals)
                    row['conditions'][condition] = item
                    del encoded, originals, inputs
                assert fingerprint(batch['inputs']) == original_hash
                record['frames'].append(row)
                print(json.dumps({'dataset': name, 'index': index, 'frames_done': len(record['frames'])}), flush=True)
            record['metrics'] = {condition: {method: {kind: {region: depth_metrics(sums) for region, sums in regions_.items()}
                                                      for kind, regions_ in values.items()}
                                               for method, values in methods.items()}
                                 for condition, methods in record['totals'].items()}
            report['datasets'][name] = record
            write_json(OUTPUT / 'results.json', report)
    assert report['shared_encoding_full_forward_checks'] == 140
    report['completed'] = True
    report['model_predictions'] = 2240
    report['additional_independent_full_forwards'] = 140
    for source in sources:
        assert digest(source) == report['source_sha256'][str(source.relative_to(ROOT))]
    write_json(OUTPUT / 'results.json', report)
    rows = []
    for name, domain in report['datasets'].items():
        for condition, methods in domain['metrics'].items():
            for method, values in methods.items():
                for region, metric in values['depth'].items():
                    clean = domain['metrics']['clean'][method]['depth'][region]
                    rows.append({'dataset': name, 'units': domain['units'], 'condition': condition, 'method': method,
                                 'region': region, **metric,
                                 'MAE_change_vs_own_clean': metric['MAE'] - clean['MAE'] if metric['MAE'] is not None else None,
                                 'MAE_ratio_vs_own_clean': metric['MAE'] / clean['MAE'] if metric['MAE'] is not None else None})
    with (OUTPUT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'completed': True, 'model_predictions': 2240, 'full_forward_checks': 140}), flush=True)


if __name__ == '__main__':
    main()
