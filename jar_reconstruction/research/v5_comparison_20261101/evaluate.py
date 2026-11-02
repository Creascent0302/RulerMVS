"""共同评测器的薄扩展：V5、V2及独立原生深度基线，不改变预测头。"""

import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import Models as ExistingModels
from official_methods.registry import build_adapter
from research.rulermvs_v5_20261031 import evaluation as v5_factory
from research.near_mid_evaluation_20261028 import evaluate


class Models(ExistingModels):
    def get(self, method, domain):
        settings = self.config["methods"][method]
        kind = settings["kind"]
        if kind not in ("patchmatchnet", "diffmvs"):
            return super().get(method, domain)
        key = (method, settings["weights"])
        if key not in self.loaded:
            config = {
                "method": kind,
                "model": {
                    **copy.deepcopy(settings.get("model", {})),
                    "depth_range": [domain["near"], domain["far"]],
                    "weights": "" if settings.get("project_checkpoint") else settings["weights"],
                    "reference_indices": [domain["reference_index"]],
                },
                "evaluation": {"reference_indices": [domain["reference_index"]]},
                "_loading_shared_checkpoint": bool(settings.get("project_checkpoint")),
            }
            adapter = build_adapter(config)
            if settings.get("project_checkpoint"):
                saved = torch.load(settings["weights"], map_location="cpu", weights_only=False)
                if saved["method"] != kind:
                    raise ValueError("权重方法与配置不符")
                adapter.load_state_dict(saved["model"], strict=True)
            self.loaded[key] = adapter.to(self.device).eval().requires_grad_(False)
        return self.loaded[key]

    def forward(self, method, domain, inputs):
        if self.config["methods"][method]["kind"] not in ("patchmatchnet", "diffmvs"):
            return super().forward(method, domain, inputs)
        adapter = self.get(method, domain)
        adapter.depth_min, adapter.depth_max = domain["near"], domain["far"]
        adapter.config["evaluation"]["reference_indices"] = [domain["reference_index"]]
        return adapter(inputs)


evaluate.Models = Models

if __name__ == "__main__":
    evaluate.main()
