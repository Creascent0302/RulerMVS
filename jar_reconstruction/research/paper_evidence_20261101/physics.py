"""固定VAL上的尺度、rig坐标、标定压力与原生置信度诊断。"""

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import add_statistics, digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from research.rulermvs_v6_20261101.evaluation import Models
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import offline_only


def rotation_y(degrees, like):
    angle = math.radians(degrees)
    sine, cosine = math.sin(angle), math.cos(angle)
    return like.new_tensor([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])


def alter(inputs, domain, condition):
    """仅改输入标定与预登记范围规则；不接收targets。"""
    result = dict(inputs)
    changed_domain = copy.deepcopy(domain)
    transform = inputs['T_cam_from_rig'].clone()
    kind = condition['kind']
    scale = 1.0
    if kind in ('units', 'physical_scale'):
        scale = condition['scale']
        transform[..., :3, 3] *= scale
        for key in ('near', 'far'):
            changed_domain[key] *= scale
        if kind == 'units':
            changed_domain['depth_scale'] *= scale
    elif kind == 'rig':
        change = torch.eye(4, dtype=transform.dtype, device=transform.device)
        change[:3, :3] = rotation_y(23, transform)
        change[:3, 3] = transform.new_tensor([1.3, -0.7, 0.4])
        transform = transform @ torch.linalg.inv(change)
    elif kind == 'focal':
        for key in ('K', 'matching_K'):
            if key in result:
                intrinsic = result[key].clone()
                intrinsic[..., 0, 0] *= condition['factor']
                intrinsic[..., 1, 1] *= condition['factor']
                result[key] = intrinsic
    elif kind == 'source_yaw':
        reference = domain['reference_index']
        sources = [i for i in range(inputs['rgb'].shape[1])
                   if i != reference and bool(inputs['view_mask'][0, i])]
        if not sources:
            raise ValueError('标定压力样本必须有一个真实源相机')
        source = sources[0]
        # 绕该相机光心改变相机朝向，保持其物理中心不变。
        turn = rotation_y(condition['degrees'], transform)
        transform[:, source, :3] = turn @ transform[:, source, :3]
    elif kind != 'clean':
        raise ValueError(kind)
    result['T_cam_from_rig'] = transform
    if 'T_cam_to_rig' in result:
        result['T_cam_to_rig'] = torch.linalg.inv(transform)
    return result, changed_domain, scale


def confidence_statistics(confidence, relative_error):
    confidence = np.concatenate(confidence).astype(np.float64)
    error = np.concatenate(relative_error).astype(np.float64)
    if not len(error):
        return {'pixels': 0}
    ordered = error[np.argsort(-confidence, kind='stable')]
    cumulative = np.cumsum(ordered) / np.arange(1, len(error) + 1)
    oracle = np.cumsum(np.sort(error)) / np.arange(1, len(error) + 1)
    coverage = np.linspace(.01, 1, 100)
    positions = np.maximum(1, np.ceil(coverage * len(error)).astype(int)) - 1
    bins = []
    for index in range(10):
        mask = (confidence >= index / 10) & (confidence < (index + 1) / 10)
        if index == 9:
            mask |= confidence == 1
        bins.append({'count': int(mask.sum()),
                     'mean_confidence': float(confidence[mask].mean()) if mask.any() else None,
                     'accuracy_rel5': float((error[mask] < .05).mean()) if mask.any() else None})
    return {'pixels': len(error), 'coverage': coverage.tolist(),
            'risk_AbsRel': cumulative[positions].tolist(),
            'oracle_risk_AbsRel': oracle[positions].tolist(),
            'random_expected_risk': float(error.mean()),
            'AURC': float(cumulative.mean()), 'oracle_AURC': float(oracle.mean()),
            'excess_AURC': float((cumulative - oracle).mean()),
            'uncalibrated_bins': bins}


def run(settings):
    output = Path(settings['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if not output.resolve().is_relative_to(Path('/data')):
        raise ValueError('结果必须存放到数据盘')
    output.mkdir(parents=True)
    plan = json.loads(Path(settings['evaluation_plan']).read_text())
    indices_plan = json.loads(Path(settings['diagnostics']).read_text())['plans']
    report = {'completed': False, 'split': 'VAL', 'test_accessed': False,
              'settings': settings, 'datasets': {}, 'source_sha256': digest(__file__),
              'notes': [
                  '所有条件共享参考RGB和GT像素；前向函数不接收targets。',
                  'units改变单位并同步depth_scale，是实现检查，不是泛化结果。',
                  'units只适用于接受depth_scale单位契约的自研模型；MVS原生公制接口不作任意改单位的性能比较。',
                  'physical_scale同步平移及TRAIN范围、保持归一尺度不变；所有方法获得同样范围条件。',
                  'rig只是重新表达坐标；focal/source_yaw是真实错误标定压力，不要求结果不变。',
                  '尺度条件的预测除以s后与原GT评分，单位回到原数据域；不混合米与scene_unit。',
                  '置信度只评估V5/V6原生confidence，无原生可比MVS置信度时不伪造。',
                  'AURC按所有可评分像素排序；置信度相同时固定光栅顺序，不使用GT打破平局。',
              ]}
    write_json(output / 'plan.json', report)
    offline_only()
    for phase in ('public', 'carla_new'):
        config = json.loads(Path(plan['groups'][phase + '_main']['val_config']).read_text())
        config['methods'] = {name: config['methods'][name] for name in ('v5_best', 'v6_best', 'mvsanywhere_refined')}
        first = copy.deepcopy(config['methods']['mvsanywhere_refined'])
        first['refinements'] = 0
        config['methods']['mvsanywhere_firstpass'] = first
        device = configure(config, settings['gpu'])
        models = Models(config, device)
        for domain in config['datasets']:
            name = domain['name']
            selected = indices_plan[name]
            if digest(domain['manifests']['val']) != selected['manifest_sha256']:
                raise ValueError('预登记VAL清单改变')
            dataset = build_dataset(dataset_config(domain, config), 'val', load_targets=True)
            totals = {c['name']: {m: {} for m in config['methods']} for c in settings['conditions']}
            confidence = {m: [[], []] for m in ('v5_best', 'v6_best')}
            frames = []
            for index in selected['indices']:
                batch = collate_frames([dataset[index]])
                inputs = move(equal_pixel_inputs(batch['inputs']), device)
                truth, mask = ground_truth(batch, domain)
                masks = regions(truth, mask, domain)
                clean = {}
                row = {'index': index, 'conditions': {}}
                for condition in settings['conditions']:
                    work, local_domain, scale = alter(inputs, domain, condition)
                    item = {}
                    with torch.inference_mode():
                        for method in config['methods']:
                            if condition['kind'] == 'units' and config['methods'][method]['kind'] != 'own':
                                item[method] = {'not_applicable': '原生接口要求公制输入；单位换算仅检查自研归一化实现'}
                                continue
                            native_confidence = None
                            if config['methods'][method]['kind'] == 'own':
                                model = models.get(method, local_domain)
                                raw = model(work, local_domain, local_domain['reference_index'])
                                prediction = {'depth': raw['depth'] * raw['unit_scale'], 'valid': raw['valid']}
                                if condition['name'] == 'clean':
                                    native_confidence = raw['confidence'].detach().cpu()
                                del raw
                            else:
                                prediction = models.forward(method, local_domain, work)
                            depth, valid = models.convert(prediction, work)
                            depth, valid = depth.cpu().double() / scale, valid.cpu()
                            del prediction
                            sums = {r: depth_sums(depth, valid, truth, masks[r]) for r in ('full', 'near', 'mid', 'far')}
                            method_total = totals[condition['name']][method]
                            for region, values in sums.items():
                                add_statistics(method_total.setdefault(region, {}), values)
                            if condition['name'] == 'clean':
                                clean[method] = (depth.clone(), valid.clone())
                            same = valid & clean[method][1] & mask
                            difference = (depth[same] - clean[method][0][same]).abs()
                            delta = {'sum_abs': float(difference.sum()), 'pixels': int(same.sum()),
                                     'max_abs': float(difference.max()) if len(difference) else None,
                                     'valid_mask_equal': bool(torch.equal(valid, clean[method][1]))}
                            item[method] = {'depth_sums': sums, 'relative_to_clean_prediction': delta}
                            if native_confidence is not None:
                                selected_pixels = mask & valid
                                assert native_confidence.shape == depth.shape
                                confidence[method][0].append(native_confidence[selected_pixels].numpy())
                                confidence[method][1].append(((depth - truth).abs() / truth.clamp_min(1e-12))[selected_pixels].numpy())
                    row['conditions'][condition['name']] = item
                frames.append(row)
                print(json.dumps({'domain': name, 'index': index, 'done': len(frames)}, ensure_ascii=False), flush=True)
            domain_result = {
                'units': domain['geometry_units'], 'indices': selected['indices'],
                'manifest_sha256': selected['manifest_sha256'], 'frames': frames,
                'methods': {c: {m: {r: depth_metrics(v) for r, v in regions_.items()}
                                for m, regions_ in values.items()} for c, values in totals.items()},
                'confidence': {m: confidence_statistics(*values) for m, values in confidence.items()},
                'weights_sha256': {m: digest(s['weights']) for m, s in config['methods'].items()},
            }
            report['datasets'][name] = domain_result
            write_json(output / 'results.json', report)
        models.loaded.clear()
        del models, model
        gc.collect()
        torch.cuda.empty_cache()
    report['completed'] = True
    write_json(output / 'results.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
