"""固定V6 parent权重：只替换上采样mask头的输入，不重算粗/细候选。"""

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
from research.rulermvs_v6_20261101.group_feature import pool_selected_group_features
from research.rulermvs_v6_20261101.handoff import masked_probability
from research.rulermvs_v6_20261101.model import build_from_checkpoint
from tools.smoke_outdoor_datasets import offline_only

RUN = ROOT / 'runs/active/rulermvs_v6_20261101'


@torch.inference_mode()
def paired_render(model, inputs, domain):
    """同一次前向捕获fine和特征；两个输出只差mask头收到的特征。"""
    decoder = model.decoder
    original_render = decoder._render
    captured = {}

    def capture(fine, features, image_mask, prepared):
        captured.update(fine=fine, features=features, image_mask=image_mask, prepared=prepared)
        return original_render(fine, features, image_mask, prepared)

    decoder._render = capture
    try:
        original = model(inputs, domain, reference=domain['reference_index'])
    finally:
        decoder._render = original_render
    fine = captured['fine']
    before_coarse = original['coarse']['depth'].clone()
    before_fine = fine['depth'].clone()
    conditional = masked_probability(fine['scores'].unflatten(1, (-1, 3)),
                                     fine['valid'].unflatten(1, (-1, 3)), dim=2)
    pooled = pool_selected_group_features(captured['features'], fine['selected_group'],
                                          conditional, fine['valid'])
    handle = decoder.upsample_mask.register_forward_pre_hook(lambda module, args: (pooled,))
    try:
        alternative = original_render(**captured)
    finally:
        handle.remove()
    if not torch.equal(before_coarse, original['coarse']['depth']) or not torch.equal(before_fine, fine['depth']):
        raise RuntimeError('重渲染改变了粗/细深度，拒绝把该结果称为固定候选诊断')
    result = {name: {key: value[key].detach().cpu() for key in ('depth', 'valid')}
              for name, value in [('winner_feature', original), ('pooled_feature', alternative)]}
    result['fine'] = before_fine.detach().cpu()
    return result


def collect(prediction, fine, truth, masks, scale):
    record = {'frames': 1, 'depth': {}, 'fine_stride4': {}, 'final_stride4': {}, 'phases': {}}
    depth = prediction['depth'] * scale
    for region, mask in masks.items():
        record['depth'][region] = depth_sums(depth, prediction['valid'], truth, mask)
        record['fine_stride4'][region] = depth_sums(
            fine * scale, torch.isfinite(fine) & (fine > 0), truth[..., ::4, ::4], mask[..., ::4, ::4])
        record['final_stride4'][region] = depth_sums(
            depth[..., ::4, ::4], prediction['valid'][..., ::4, ::4], truth[..., ::4, ::4], mask[..., ::4, ::4])
    for py in range(4):
        for px in range(4):
            index = (..., slice(py, None, 4), slice(px, None, 4))
            record['phases'][f'{py},{px}'] = depth_sums(
                depth[index], prediction['valid'][index], truth[index], masks['full'][index])
    return record


def accumulate(total, record):
    total['frames'] = total.get('frames', 0) + record['frames']
    for kind in ('depth', 'fine_stride4', 'final_stride4', 'phases'):
        for name, sums in record[kind].items():
            add_statistics(total.setdefault(kind, {}).setdefault(name, {}), sums)


def finish(record):
    return {'frames': record['frames'],
            **{kind: {key: depth_metrics(value) for key, value in record[kind].items()}
               for kind in ('depth', 'fine_stride4', 'final_stride4', 'phases')}}


def run(smoke=False, device='cuda:1'):
    offline_only()
    seed_all(20261101)
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = RUN / 'feature_probe' / ('smoke' if smoke else 'full')
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).with_name('group_feature.py'),
                    Path(__file__).with_name('model.py'), Path(__file__).with_name('handoff.py')]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    report = {'completed': False, 'test_accessed': False, 'training_run': False,
              'timing_run': False, 'input_mode': 'equal_pixels', 'device': device,
              'source_sha256': hashes, 'weights': {}, 'datasets': {},
              'notes': ['使用首轮旧目标的V6 parent末步权重，不是round2修正目标权重。',
                        '粗/细各一次前向，仅重算_render，深度和组选择不变。',
                        '原mask头用组内winner特征，probe改为同组条件概率池化特征。',
                        '临时hook调用后移除，不改变生产默认；固定16VAL，不据GT切换方法。']}
    for phase in ('public', 'carla'):
        config = json.loads((RUN / 'full_validation/configs' / (phase + '.json')).read_text())
        weights = config['methods']['v6_parent']['weights']
        saved = torch.load(weights, map_location='cpu', weights_only=False, mmap=True)
        model, _ = build_from_checkpoint(saved, 'parent')
        model.requires_grad_(False).to(device).eval()
        install_position_cache(model.foundation.model)
        prepare_inference(model, spatial_layout=True)
        del saved
        report['weights'][phase] = {'path': weights, 'sha256': digest(weights)}
        domains = config['datasets'][:1] if smoke else config['datasets']
        for original_domain in domains:
            domain = copy.deepcopy(original_domain)
            domain['manifests'] = {'val': domain['manifests']['val']}
            dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=True)
            indices = torch.linspace(0, len(dataset) - 1, min(1 if smoke else 16, len(dataset))).long().tolist()
            plan = {'indices': indices, 'development': indices[::2], 'confirmation': indices[1::2],
                    'manifest_sha256': digest(domain['manifests']['val'])}
            loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate_frames, num_workers=0)
            totals = {name: {variant: {} for variant in ('winner_feature', 'pooled_feature')}
                      for name in ('all', 'development', 'confirmation')}
            per_frame = []
            for index, batch in zip(indices, loader):
                inputs = equal_pixel_inputs(move(batch['inputs'], device))
                predictions = paired_render(model, inputs, domain)
                truth, valid = ground_truth(batch, domain)
                masks = regions(truth, valid, domain)
                partition = 'development' if index in plan['development'] else 'confirmation'
                frame = {'index': index, 'partition': partition, 'methods': {}}
                for variant in ('winner_feature', 'pooled_feature'):
                    record = collect(predictions[variant], predictions['fine'], truth, masks, domain['depth_scale'])
                    frame['methods'][variant] = finish(record)
                    for name in ('all', partition):
                        accumulate(totals[name][variant], record)
                per_frame.append(frame)
            result = {'completed': True, 'frames': len(indices), 'units': domain['geometry_units'],
                      'plan': plan, 'coarse_fine_unchanged': True,
                      'partitions': {name: {variant: finish(value) for variant, value in methods.items() if value}
                                     for name, methods in totals.items()}}
            write_json(output / (domain['name'] + '.json'), dict(result, per_frame=per_frame))
            report['datasets'][domain['name']] = result
            write_json(output / 'summary.json', report)
            print(json.dumps({'dataset': domain['name'], 'frames': len(indices),
                              'full_MAE': {k: v['depth']['full']['MAE']
                                           for k, v in result['partitions']['all'].items()}}, ensure_ascii=False), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if any(digest(ROOT / name) != sha for name, sha in hashes.items()):
        raise RuntimeError('probe期间源码发生改变')
    report['completed'] = True
    write_json(output / 'summary.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    try:
        run(args.smoke, args.device)
    except Exception:
        folder = RUN / 'feature_probe' / ('smoke' if args.smoke else 'full')
        write_json(folder / 'failure.json', {'completed': False, 'traceback': traceback.format_exc()})
        raise
