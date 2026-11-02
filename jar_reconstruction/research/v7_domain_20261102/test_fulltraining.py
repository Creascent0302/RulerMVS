"""新增模块学习率与已有成熟层严格分组，不以名字误用旧LR。"""

import torch
from torch import nn

from research.v7_domain_20261102.full_training import optimizer_for


def test_optimizer_groups():
    model = nn.Module()
    model.decoder = nn.ModuleDict({name: nn.Linear(2, 1) for name in
                                   ("center", "context_prior", "handoff", "score")})
    model.bridge = nn.Linear(2, 1)
    model.foundation = nn.Linear(2, 1)
    model.matching = nn.ModuleDict({"backbone": nn.Linear(2, 1), "lateral": nn.Linear(2, 1)})
    rates = {"center": 3e-4, "prior": 3e-4, "handoff": 3e-6, "decoder": 3e-6,
             "bridge": 1e-6, "foundation": 5e-8, "matching": 2e-7, "fpn": 1e-6}
    optimizer = optimizer_for(model, {name + "_lr": lr for name, lr in rates.items()},
                              {"train": {"weight_decay": 1e-4}})
    assert {group["group_name"] for group in optimizer.param_groups} == set(rates)
    parameters = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(parameters) == len(set(parameters)) == len(list(model.parameters()))
    for group in optimizer.param_groups:
        assert group["lr"] == group["initial_lr"] == rates[group["group_name"]]
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
