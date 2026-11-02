"""近处逆深度、远处米制均匀采样；候选顺序始终从远到近。"""

import torch


def inverse_depth_grid(near, far, count, mode="inverse", transition=16.0, near_fraction=2 / 3):
    if not 0 < near < far or count < 3:
        raise ValueError("深度范围须递增且候选数至少3")
    if mode == "inverse":
        return torch.linspace(1 / far, 1 / near, count)
    if mode != "hybrid" or not near < transition < far or not 0 < near_fraction < 1:
        raise ValueError("hybrid须满足near < transition < far且0 < near_fraction < 1")
    close_count = max(2, min(count - 1, round(count * near_fraction)))
    close = torch.linspace(1 / near, 1 / transition, close_count).reciprocal()
    distant = torch.linspace(transition, far, count - close_count + 1)[1:]
    return torch.cat((close, distant)).flip(0).reciprocal()


def configured_inverse_depths(model, count):
    return inverse_depth_grid(
        model.get("near", 2.0),
        model.get("far", 80.0),
        count,
        model.get("depth_sampling", "inverse"),
        model.get("depth_transition", 16.0),
        model.get("depth_near_fraction", 2 / 3),
    )
