"""相同但不正确的预测，不应被只惩罚退化的保持项再次施加GT梯度。"""

import torch

from research.v7_domain_20261102.retention_round2 import retention_penalty


def test_equal_quality_zero_gradient():
    prediction = torch.tensor([[[[1.5, 2.0]]]], requires_grad=True)
    inputs = {"view_mask": torch.ones(1, 1, dtype=torch.bool),
              "image_mask": torch.ones(1, 1, 1, 2, dtype=torch.bool)}
    targets = {"depth": torch.ones(1, 1, 1, 2),
               "depth_valid": torch.ones(1, 1, 1, 2, dtype=torch.bool)}
    loss = retention_penalty({"depth": prediction, "unit_scale": 1},
                             {"depth": prediction.detach()}, inputs, targets, {})
    loss.backward()
    assert loss == 0 and not prediction.grad.any()


def test_improvement_free_and_worsening_penalized():
    prediction = torch.tensor([[[[1.4, 1.6]]]], requires_grad=True)
    teacher = torch.full_like(prediction, 1.5, requires_grad=True)
    inputs = {"view_mask": torch.ones(1, 1, dtype=torch.bool),
              "image_mask": torch.ones(1, 1, 1, 2, dtype=torch.bool)}
    targets = {"depth": torch.ones(1, 1, 1, 2),
               "depth_valid": torch.ones(1, 1, 1, 2, dtype=torch.bool)}
    loss = retention_penalty({"depth": prediction, "unit_scale": 1},
                             {"depth": teacher}, inputs, targets, {})
    loss.backward()
    assert loss > 0 and prediction.grad[0, 0, 0, 0] == 0
    assert prediction.grad[0, 0, 0, 1] > 0 and teacher.grad is None
