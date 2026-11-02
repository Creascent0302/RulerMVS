"""固定输入和随机流的完整对比，不修改任何网络算法。"""

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from research.v5_comparison_20261101.evaluate import Models as NativeModels
from research.near_mid_evaluation_20261028 import evaluate
from tools.smoke_outdoor_datasets import fingerprint


class Models(NativeModels):
    def forward(self, method, domain, inputs):
        settings = self.config["methods"][method]
        if settings["kind"] not in ("patchmatchnet", "diffmvs"):
            return super().forward(method, domain, inputs)
        # 加载权重的随机初始化消耗与实际推理随机数分离。
        self.get(method, domain)
        rgb_geometry = {key: inputs[key] for key in ("rgb", "K", "T_cam_from_rig", "view_mask")}
        token = f"{fingerprint(rgb_geometry)}:{domain['reference_index']}:{settings.get('sample_seed', 1701)}"
        seed = int(hashlib.sha256(token.encode()).hexdigest()[:15], 16)
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            return super().forward(method, domain, inputs)


evaluate.Models = Models

if __name__ == "__main__":
    evaluate.main()
