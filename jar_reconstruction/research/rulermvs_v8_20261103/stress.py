"""固定VAL小样本的分阶段压力测试，所有方法共用RGB、标定与GT掩码。"""

import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from jar.pretraining.comparison import equal_pixel_inputs, write_json
from jar.pretraining.data import collate_mixed
from jar.runtime import move
from research.overnight_v6_v7_mvs_20261103.common_data import SharedDataset
from research.rulermvs_v8_20261103.train import build_model
from tools.smoke_outdoor_datasets import offline_only


def scenario_inputs(inputs, mode):
    if mode != "appearance":
        return inputs
    result = dict(inputs)
    rgb = inputs["rgb"].clone()
    # 固定轻微但未用于选强度的源相机曝光/gamma偏移；K/GT完全不变。
    for view in range(1, rgb.shape[1]):
        gamma = 1.2 if view % 2 else .8
        gain = .8 if view % 2 else 1.15
        rgb[:, view] = (rgb[:, view].clamp(0, 1).pow(gamma) * gain).clamp(0, 1)
    result["rgb"] = rgb
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-proposals", action="store_true",
                        help="诊断：固定权重，只恢复V6的候选阈值单位，不改变标尺信息")
    args = parser.parse_args()
    offline_only()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = copy.deepcopy(saved["config"])
    if args.legacy_proposals:
        if config["method"] != "v8":
            raise ValueError("该诊断只针对V8")
        config["v8"]["physical_thresholds"] = False
    model = build_model(config, initialize=False)
    model.load_state_dict(saved["model"], strict=True)
    model.cuda().eval().requires_grad_(False)
    dataset = SharedDataset(config, "val")
    rows = []
    with torch.inference_mode():
        for di, domain in enumerate(dataset.domains):
            indices = torch.linspace(0, len(dataset.datasets[di]) - 1, 16).long().tolist()
            sums = {mode: {stage: torch.zeros(5, dtype=torch.float64, device="cuda")
                           for stage in ("coarse", "fine", "final")}
                    for mode in ("clean", "wide_range", "appearance")}
            for index in indices:
                batch = move(collate_mixed([dataset[(di, index, -1)]]), "cuda")
                truth = batch["targets"]["depth"][:, :1]
                valid = batch["targets"]["depth_valid"][:, :1] & batch["inputs"]["image_mask"][:, :1]
                valid &= torch.isfinite(truth) & (truth > 0)
                for mode in sums:
                    work = equal_pixel_inputs(scenario_inputs(batch["inputs"], mode))
                    settings = dict(domain)
                    if mode == "wide_range":
                        settings.update(near=domain["near"] * .25, far=domain["far"] * 4)
                    prediction = model(work, settings)
                    for stage, stride in (("coarse", 8), ("fine", 4), ("final", 1)):
                        estimate = (prediction if stage == "final" else prediction[stage])["depth"] * prediction["unit_scale"]
                        target, mask = truth[..., ::stride, ::stride], valid[..., ::stride, ::stride]
                        if not bool((torch.isfinite(estimate[mask]) & (estimate[mask] > 0)).all()):
                            raise RuntimeError("无效预测不得静默移出GT分母")
                        error = estimate[mask].double() - target[mask].double()
                        sums[mode][stage] += torch.stack((error.new_tensor(error.numel()),
                            error.abs().sum(), error.square().sum(),
                            (error.abs() / target[mask]).sum(), error.sum()))
            for mode, stages in sums.items():
                row = dict(dataset=domain["name"], mode=mode, units=domain["geometry_units"], frames=len(indices))
                for stage, values in stages.items():
                    count, absolute, squared, relative, signed = values.cpu().tolist()
                    row[stage] = dict(pixels=int(count), mae=absolute / count,
                                      rmse=math.sqrt(squared / count), abs_rel=relative / count,
                                      signed_error=signed / count)
                rows.append(row)
            print(domain["name"], "done", flush=True)
    write_json(args.output, dict(checkpoint=args.checkpoint, rows=rows, split="fixed_VAL16_per_domain",
                                test_accessed=False, precision="fp32", tf32=False,
                                legacy_proposals=args.legacy_proposals))


if __name__ == "__main__":
    main()
