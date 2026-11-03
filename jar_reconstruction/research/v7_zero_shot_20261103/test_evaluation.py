"""验证原尺寸评价的误差分区和有效像素分母。"""

import torch
from research.v7_zero_shot_20261103.evaluate import score_native


def test_regions_keep_all_valid_gt_and_do_not_align_scale():
    gt = torch.tensor([[[[1., 10., 30., 150., float('inf')]]]])
    predicted = torch.tensor([[[[2., 20., 60., 300., 1.]]]])
    result = score_native(predicted, torch.ones_like(predicted, dtype=torch.bool), gt)
    assert result['full']['gt_pixels'] == 4
    assert result['full']['abs_sum'] == 191
    assert result['full']['abs_rel_sum'] == 4
    assert sum(result[k]['gt_pixels'] for k in ['near', 'mid', 'far']) == 4
    assert result['far']['gt_pixels'] == 2  # 100米以外GT没有被模型输入范围裁掉。


def test_invalid_prediction_does_not_remove_gt_from_denominator():
    gt = torch.ones(1, 1, 2, 2)
    valid = torch.ones_like(gt, dtype=torch.bool)
    valid[..., 0, 0] = False
    sums = score_native(gt, valid, gt)['full']
    assert sums['gt_pixels'] == 4 and sums['predicted_pixels'] == 3
