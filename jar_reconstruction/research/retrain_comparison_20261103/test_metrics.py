"""核对横轴归一化与帕累托方向，不用真实TEST调整绘图规则。"""
import torch
from research.near_mid_evaluation_20261028.metrics import depth_sums
from research.retrain_comparison_20261103.evaluate import finish
from research.retrain_comparison_20261103.report import nondominated


def test_scale_invariance_and_missing():
    gt = torch.tensor([1., 10., 100.])
    pred = torch.tensor([2., 12., 90.])
    valid = torch.ones(3, dtype=torch.bool)
    values = []
    for scale in (1., 1000.):
        sums = depth_sums(pred * scale, valid, gt * scale, valid)
        sums['gt_depth_sum'] = float((gt * scale).sum())
        values.append(finish(sums)['NMAE'])
    assert abs(values[0] - 13 / 111) < 1e-12
    assert values[0] == values[1]
    sums = depth_sums(pred, torch.tensor([True, False, True]), gt, valid)
    sums['gt_depth_sum'] = float(gt.sum())
    assert finish(sums)['NMAE'] is None


def test_pareto_minimizes_both_and_preserves_ties():
    assert nondominated([(1, 3), (2, 2), (3, 3), (2, 2)]) == [True, True, False, True]
