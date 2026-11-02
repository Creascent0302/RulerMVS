"""新增共同协议的单位标记与监督分离测试。"""

import pytest
import torch

from official_methods.metrics import Evaluator, depth_statistics, finish_depth, surface_metrics


def test_scene_unit_metrics_are_never_labeled_as_meters():
    gt = torch.tensor([2.0, 4.0])
    pred = torch.tensor([3.0, 6.0])
    stats = depth_statistics(pred, gt, gt > 0, pred > 0, 1, 10)
    result = finish_depth(stats, "scene_unit")
    assert result["MAE_scene_unit"] == 1.5
    assert "MAE_m" not in result and "RMSE_m" not in result
    cloud = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    surface = surface_metrics(cloud, cloud, [0.1], "scene_unit")
    assert surface["Fscore_0.1scene_unit"] == 1
    assert "Accuracy_m" not in surface
    with pytest.raises(ValueError, match="surface_thresholds_scene_unit"):
        Evaluator({"data": {"geometry_units": "scene_unit"}})


def test_meter_defaults_and_missing_prediction_denominator_remain_explicit():
    gt = torch.tensor([2.0, 4.0])
    pred = torch.tensor([2.0, float("nan")])
    stats = depth_statistics(pred, gt, gt > 0, torch.isfinite(pred), 1, 10)
    result = finish_depth(stats)
    assert result["MAE_m"] == 0  # 只对已预测像素统计，必须伴随coverage解读。
    assert result["prediction_coverage"] == 0.5
    assert result["delta_1"] == 0.5  # 未预测不能从全部GT分母删除。
    assert "分母" in result["error_denominator"]
