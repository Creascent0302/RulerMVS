"""只干预编码条件的VAL诊断；不修改候选投影K/T，不训练或选择权重。

错误/缺失条件的反事实只能检查模型对条件的敏感性，不能单独证明跨相机泛化。
angular_canonical保留真实ray与基线，仅把角足迹及导数夹角换为固定90度水平视场。
"""

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset
from jar.runtime import seed_all
from research.early_geometry_20261030 import encoding
from research.early_geometry_20261030.pilot import load_model, path
from research.near_evidence_20261030.pilot import evaluate, parameter_hash
from tools.smoke_outdoor_datasets import offline_only


@contextmanager
def intervention(model, name):
    """单进程诊断上下文；所有临时函数、hook和gain在退出时恢复。"""
    descriptor = encoding.camera_descriptor
    handle = None
    original_gain = None

    def changed_descriptor(inputs, domain, grid, footprint):
        actual = descriptor(inputs, domain, grid, footprint)
        if name == "baseline_zero":
            actual[:, :, 6] = 0.
        else:
            height, width = inputs["rgb"].shape[-2:]
            canonical = torch.zeros_like(inputs["K"])
            canonical[..., 0, 0] = width / 2
            canonical[..., 1, 1] = width / 2
            canonical[..., 0, 2] = (width - 1) / 2
            canonical[..., 1, 2] = (height - 1) / 2
            canonical[..., 2, 2] = 1.
            # 浅拷贝仅传给descriptor；原inputs与后续精确投影使用原K/T。
            hypothetical = descriptor(dict(inputs, K=canonical), domain, grid, footprint)
            actual[:, :, 3:6] = hypothetical[:, :, 3:6]
        return actual

    try:
        if name in {"baseline_zero", "angular_canonical"}:
            if not hasattr(model, "_encoding_options"):
                raise ValueError(f"{name}需要已经安装早期条件模块")
            if model._encoding_options["camera_mode"] != "geometry":
                raise ValueError(f"{name}只适用于真实几何条件组")
            encoding.camera_descriptor = changed_descriptor
        elif name in {"matching_off", "vit_off"}:
            branch = model.matching if name == "matching_off" else model.foundation
            if not hasattr(branch, "camera_adapter"):
                raise ValueError(f"{name}对应adapter不存在")
            handle = branch.camera_adapter.register_forward_hook(lambda module, arguments, output: arguments[0])
        elif name == "late_off":
            if not hasattr(model.decoder, "late_gain"):
                raise ValueError("late_off需要后融合模块")
            original_gain = model.decoder.late_gain.detach().clone()
            with torch.no_grad():
                model.decoder.late_gain.zero_()
        elif name != "original":
            raise ValueError(f"未知干预：{name}")
        yield
    finally:
        encoding.camera_descriptor = descriptor
        if handle is not None:
            handle.remove()
        if original_gain is not None:
            with torch.no_grad():
                model.decoder.late_gain.copy_(original_gain)


def run(settings):
    offline_only()
    output = path(settings["output"])
    if output.exists():
        raise FileExistsError(f"拒绝覆盖诊断：{output}")
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(settings.get("device", "cuda:0"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = path(settings["checkpoint"])
    model, saved = load_model(checkpoint, device)
    if saved.get("experiment") != "early_geometry_20261030":
        raise ValueError("此脚本只诊断本轮固定末步权重")
    plan_file = checkpoint.parent / "plan.json"
    plan = json.loads(plan_file.read_text())["validation"]
    config = copy.deepcopy(saved["config"])
    selected_names = set(settings["datasets"])
    domains = config.get("validation_datasets", config["datasets"])
    config["datasets"] = [domain for domain in domains if domain["name"] in selected_names]
    if {domain["name"] for domain in config["datasets"]} != selected_names:
        raise ValueError("诊断数据域必须属于该实验原VAL计划")
    config.pop("validation_datasets", None)
    for domain in config["datasets"]:
        domain["manifests"] = {"val": domain["manifests"]["val"]}
        if digest(path(domain["manifests"]["val"])) != plan[domain["name"]]["manifest_sha256"]:
            raise RuntimeError("VAL manifest已改变，拒绝混用原计划")
    plan = {name: plan[name] for name in selected_names}
    dataset = MixedDataset(config, "val")
    before = parameter_hash(model.state_dict())
    results = {}
    names = ["original"] + settings["interventions"]
    for name in names:
        if name in results:
            raise ValueError(f"重复干预：{name}")
        with intervention(model, name):
            results[name] = evaluate(model, dataset, plan, device)
        print(f"completed: {name}", flush=True)
    if before != parameter_hash(model.state_dict()):
        raise RuntimeError("诊断结束后模型参数未正确恢复")
    write_json(output, {
        "completed": True, "checkpoint_sha256": digest(checkpoint),
        "plan_sha256": digest(plan_file), "validation": plan, "results": results,
        "test_accessed": False, "training_performed": False,
        "precision": "FP32, TF32 disabled", "weights_unchanged": True,
        "projection_calibration_unchanged": True,
        "scope": "历史VAL上的条件敏感性诊断；错误条件属于分布外干预，不证明泛化或架构净收益。",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
