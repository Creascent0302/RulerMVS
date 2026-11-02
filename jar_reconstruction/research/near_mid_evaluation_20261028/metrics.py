"""固定GT分母的深度指标与单参考可见表面指标，不作全场景占据评价。"""

import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F

from official_methods.metrics import depth_statistics


def distance_limits(domain):
    if domain['geometry_units'] == 'meters':
        return 10., 40.
    median = 10 * domain['depth_scale']
    return .5 * median, 2 * median


def ground_truth(batch, domain):
    reference = domain['reference_index']
    inputs, targets = batch['inputs'], batch['targets']
    truth = targets['depth'][:, reference:reference + 1].double()
    mask = targets['depth_valid'][:, reference:reference + 1].clone()
    mask &= inputs['image_mask'][:, reference:reference + 1]
    mask &= inputs['view_mask'][:, reference:reference + 1, None, None]
    mask &= torch.isfinite(truth) & (truth > 0)
    if domain.get('eval_range'):
        low, high = domain['eval_range']
        mask &= (truth >= low) & (truth <= high)
    return truth, mask


def regions(truth, mask, domain):
    near, far = distance_limits(domain)
    logz = truth.clamp_min(1e-12).log()
    horizontal = (logz[..., 1:] - logz[..., :-1]).abs() > .1
    horizontal &= mask[..., 1:] & mask[..., :-1]
    vertical = (logz[..., 1:, :] - logz[..., :-1, :]).abs() > .1
    vertical &= mask[..., 1:, :] & mask[..., :-1, :]
    edge = F.pad(horizontal, (0, 1)) | F.pad(horizontal, (1, 0))
    edge |= F.pad(vertical, (0, 0, 0, 1)) | F.pad(vertical, (0, 0, 1, 0))
    edge = F.max_pool2d(edge.float(), 5, stride=1, padding=2) > 0
    return {'full': mask, 'near': mask & (truth < near),
            'mid': mask & (truth >= near) & (truth < far), 'far': mask & (truth >= far),
            'gradient_band': mask & edge, 'smooth': mask & ~edge}


def depth_sums(prediction, valid, truth, mask):
    prediction = prediction.double()
    result = depth_statistics(prediction, truth, mask, valid, 0., math.inf)
    usable = mask & valid & torch.isfinite(prediction) & (prediction > 0)
    error = prediction[usable] - truth[usable]
    log_error = prediction[usable].log() - truth[usable].log()
    result.update(signed_sum=float(error.sum()), sq_rel_sum=float((error.square() / truth[usable]).sum()),
                  log_sq_sum=float(log_error.square().sum()), log_abs_sum=float(log_error.abs().sum()))
    return result


def depth_metrics(sums):
    total, count = sums['gt_pixels'], sums['predicted_pixels']
    mean = lambda key: sums[key] / count if count else None
    return {'MAE': mean('abs_sum'), 'RMSE': math.sqrt(mean('sq_sum')) if count else None,
            'AbsRel': mean('abs_rel_sum'), 'SqRel': mean('sq_rel_sum'),
            'RMSE_log': math.sqrt(mean('log_sq_sum')) if count else None,
            'log_MAE': mean('log_abs_sum'), 'signed_bias': mean('signed_sum'),
            **{f'delta_{i}': sums[f'delta_{i}_hits'] / total if total else None for i in (1, 2, 3)},
            'coverage': count / total if total else None,
            'gt_pixels': total, 'predicted_pixels': count,
            'full_gt_MAE': mean('abs_sum') if total and count == total else None}


def query_indices(length, budget):
    return np.linspace(0, length - 1, min(length, budget)).round().astype(np.int64)


class VisibleSurface:
    """完整对方点集作最近邻参考，仅均匀抽取距离查询点，避免双侧稀疏化偏差。"""

    def __init__(self, truth, mask, intrinsic, domain, budget=1024):
        self.mask = mask[0, 0].cpu().numpy()
        height, width = self.mask.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        pixels = np.stack((x, y, np.ones_like(x)), -1).astype(np.float64)
        self.rays = pixels @ np.linalg.inv(intrinsic.double().cpu().numpy()).T
        self.points = self.rays[self.mask] * truth[0, 0].cpu().numpy()[self.mask, None]
        self.tree = cKDTree(self.points) if len(self.points) else None
        self.queries = self.points[query_indices(len(self.points), budget)] if len(self.points) else self.points
        self.budget = budget
        self.thresholds = (.05, .1, .2, .5) if domain['geometry_units'] == 'meters' else tuple(
            10 * domain['depth_scale'] * value for value in (.005, .01, .02, .05))

    def measure(self, depth, valid):
        z = depth[0, 0].double().cpu().numpy()
        usable = self.mask & valid[0, 0].cpu().numpy() & np.isfinite(z) & (z > 0)
        points = self.rays[usable] * z[usable, None]
        queries = points[query_indices(len(points), self.budget)] if len(points) else points
        a = self.tree.query(queries, workers=1)[0] if self.tree is not None and len(queries) else np.empty(0)
        c = cKDTree(points).query(self.queries, workers=1)[0] if len(points) else np.full(len(self.queries), np.inf)
        result = {'pred_queries': len(queries), 'gt_queries': len(self.queries),
                  'pred_reference_points': len(points), 'gt_reference_points': len(self.points),
                  'accuracy_sum': float(a.sum()), 'completeness_sum': float(c[np.isfinite(c)].sum()),
                  'missing_prediction_frames': int(len(self.points) > 0 and not len(points))}
        for i, threshold in enumerate(self.thresholds):
            result[f'precision_hits_{i}'] = int((a < threshold).sum())
            result[f'recall_hits_{i}'] = int((c < threshold).sum())
        return result


def surface_metrics(sums, thresholds):
    p, g = sums['pred_queries'], sums['gt_queries']
    accuracy = sums['accuracy_sum'] / p if p and g else None
    complete = sums['completeness_sum'] / g if g and not sums['missing_prediction_frames'] else None
    result = {'Accuracy': accuracy, 'Completeness': complete,
              'Chamfer_L1': (accuracy + complete) / 2 if accuracy is not None and complete is not None else None,
              'pred_queries': p, 'gt_queries': g, 'thresholds': list(thresholds)}
    for i, threshold in enumerate(thresholds):
        precision = sums[f'precision_hits_{i}'] / p if p else 0.
        recall = sums[f'recall_hits_{i}'] / g if g else 0.
        result[f'Fscore_{threshold:g}'] = 2 * precision * recall / max(precision + recall, 1e-12) if g else None
    return result
