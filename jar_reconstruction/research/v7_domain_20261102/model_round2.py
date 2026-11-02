"""第二轮工厂：固定索引先验与连续坐标先验，第一轮模型源码保持冻结。"""

import copy

import torch

from research.rulermvs_v6_20261101.model import V6Model
from .coordinate_prior import CoordinateEvidencePrior
from .model import V7Model


class V7CoordinateModel(V7Model):
    architecture = "rulermvs_v7_coordinate"

    def __init__(self, config, initialize=True):
        super().__init__(config, initialize)
        with torch.random.fork_rng(devices=[]):
            self.decoder.context_prior = CoordinateEvidencePrior(
                raw_channels=config["model"]["raw_channels"],
                candidates=config["model"]["depth_bins"],
            )
        self.model_name = "RulerMVS-V7-coordinate-research"


def build_model(saved, variant="coordinate"):
    if variant not in ("control", "fixed", "coordinate"):
        raise ValueError(f"未知第二轮分支：{variant}")
    config = copy.deepcopy(saved["config"])
    if variant == "control":
        if saved["architecture"] != "rulermvs_v6":
            raise ValueError("旧V6保持教师/控制只接受完整V6权重")
        model = V6Model(config, initialize=False)
        model.load_state_dict(saved["model"], strict=True)
        return model, config
    architecture = "rulermvs_v7" if variant == "fixed" else "rulermvs_v7_coordinate"
    factory = V7Model if variant == "fixed" else V7CoordinateModel
    options = {"projective_center": False, "context_prior": True}
    if saved["architecture"] == "rulermvs_v6":
        config.update(architecture=architecture, v7=options)
        model = factory(config, initialize=False)
        extra = {name: value for name, value in model.state_dict().items()
                 if name.startswith("decoder.context_prior.")}
        model.load_state_dict({**saved["model"], **extra}, strict=True)
    elif saved["architecture"] == architecture and config.get("v7") == options:
        model = factory(config, initialize=False)
        model.load_state_dict(saved["model"], strict=True)
    else:
        raise ValueError("第二轮恢复必须与实际先验结构严格相同，禁止静默改名或漏加载")
    return model, config
