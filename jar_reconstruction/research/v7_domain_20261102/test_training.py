"""TRAIN 保持约束允许真实改进，忽略未知标签，且旧模型不接受梯度。"""

import torch

from research.v7_domain_20261102.train import retention_penalty


def test_retention_direction_and_teacher_detached():
    current = torch.tensor([[[[1.2, 1.1, 1.0, 10.0]]]], requires_grad=True)
    teacher = torch.tensor([[[[1.1, 1.2, 1.0, 1.0]]]], requires_grad=True)
    inputs = {"image_mask": torch.ones(1, 1, 1, 4, dtype=torch.bool),
              "view_mask": torch.ones(1, 1, dtype=torch.bool)}
    targets = {"depth": torch.ones(1, 1, 1, 4),
               "depth_valid": torch.tensor([[[[True, True, True, False]]]])}
    loss = retention_penalty({"depth": current, "unit_scale": 1.}, {"depth": teacher},
                             inputs, targets, {}, margin=0.)
    loss.backward()
    assert current.grad[0, 0, 0, 0] > 0
    assert not current.grad[0, 0, 0, 1:].any()
    assert teacher.grad is None


def test_retention_empty_supervision_is_finite_zero():
    value = torch.ones(1, 1, 2, 2, requires_grad=True)
    inputs = {"image_mask": torch.ones(1, 1, 2, 2, dtype=torch.bool),
              "view_mask": torch.ones(1, 1, dtype=torch.bool)}
    targets = {"depth": torch.full((1, 1, 2, 2), float("nan")),
               "depth_valid": torch.zeros(1, 1, 2, 2, dtype=torch.bool)}
    loss = retention_penalty({"depth": value, "unit_scale": 1.}, {"depth": value.detach()},
                             inputs, targets, {})
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    assert torch.isfinite(value.grad).all()
