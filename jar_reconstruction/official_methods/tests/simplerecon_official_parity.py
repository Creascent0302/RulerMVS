"""独立进程比对原始 SimpleRecon 与薄适配器；仅审计，不报告精度。

官方源码从本地固定 Git 对象导出，禁止隐式联网获取对象；仅拦截预训练下载，
随后严格加载同一份四目随机权重。八目官方任务权重不被裁剪成四目权重。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch
from torch.nn import functional as F

from official_methods.tests.mvsanywhere_official_parity import official_inputs, prepare


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def export_official(output):
    root = Path(__file__).resolve().parents[1] / "methods/simplerecon/upstream"
    environment = dict(os.environ, GIT_NO_LAZY_FETCH="1")

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], env=environment)

    files = git("ls-tree", "-r", "--name-only", "HEAD").decode().splitlines()
    selected = [
        name for name in files
        if (name.endswith(".py") and name.split("/")[0] in {
            "modules", "utils", "experiment_modules", "options.py", "losses.py"
        }) or name in {"LICENSE", "configs/models/hero_model.yaml"}
    ]
    hashes = {}
    for name in selected:
        value = git("show", f"HEAD:{name}")
        destination = output / "pristine" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
        hashes[name] = hashlib.sha256(value).hexdigest()
    write_json(output / "source.json", {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "source": "本地固定 Git 原始 blob，不是工作区已修改文件",
        "sha256": hashes,
    })


def adapter(config):
    from official_methods.methods.simplerecon import Adapter

    return Adapter(config)


def create_fixture(config, output):
    export_official(output)
    prepare(config, output)
    torch.manual_seed(1701)
    model = adapter(config)
    torch.save(model.native_model.state_dict(), output / "shared_state.pt")


def load_original(config, output):
    sys.path.insert(0, str(output / "pristine"))
    import timm
    import yaml
    from experiment_modules import depth_model

    opts = yaml.load(
        (output / "pristine/configs/models/hero_model.yaml").read_text(),
        Loader=yaml.UnsafeLoader,
    )
    opts.image_height, opts.image_width = config["image_size"]
    opts.model_num_views = 4
    opts.min_matching_depth = config["model"]["min_depth"]
    opts.max_matching_depth = config["model"]["max_depth"]
    opts.matching_num_depth_bins = 64
    create_model = timm.create_model
    matching = depth_model.ResnetMatchingEncoder

    def offline_image_encoder(*args, **kwargs):
        return create_model(*args, **dict(kwargs, pretrained=False))

    def offline_matching_encoder(*args, **kwargs):
        return matching(*args, **dict(kwargs, pretrained=False))

    with (
        patch.object(timm, "create_model", offline_image_encoder),
        patch.object(depth_model, "ResnetMatchingEncoder", offline_matching_encoder),
    ):
        return depth_model.DepthModel(opts)


def evaluate_loss(model, case):
    height, width = 32, 48
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    depth = (5 + x * 0.02 + y * 0.01)[None, None]
    source_depth = depth[:, None].expand(-1, 3, -1, -1, -1).clone()
    mask = torch.ones_like(depth, dtype=torch.bool)
    if case == "sparse_current":
        mask.zero_()
        mask[:, :, 8:21, 10:23] = True
        depth = depth.masked_fill(~mask, torch.nan)
    if case == "missing_source":
        source_depth.fill_(torch.nan)
    prediction = (6 + x * 0.024 + y * 0.013)[None, None].requires_grad_()
    identity = torch.eye(4)[None]
    current = {
        "depth_b1hw": depth,
        "mask_b_b1hw": mask,
        "mask_b1hw": mask.float(),
        "normals_b3hw": model.compute_normals(depth, identity),
        "invK_s0_b44": identity,
        "world_T_cam_b44": identity,
    }
    sources = {
        "depth_b1hw": source_depth,
        "K_s0_b44": identity[:, None].expand(-1, 3, -1, -1),
        "cam_T_world_bk44": identity[:, None].expand(-1, 3, -1, -1),
    }
    sources["cam_T_world_b44"] = sources.pop("cam_T_world_bk44")
    outputs = {
        "depth_pred_s0_b1hw": prediction,
        "normals_pred_b3hw": model.compute_normals(prediction, identity),
    }
    for scale in range(4):
        outputs[f"log_depth_pred_s{scale}_b1hw"] = F.interpolate(
            prediction.log(), (height // 2**scale, width // 2**scale), mode="nearest"
        )
    losses = model.compute_losses(current, sources, outputs)
    losses["loss"].backward()
    return {
        "losses": {name: float(value) if torch.isfinite(value) else None for name, value in losses.items()},
        "gradient": prediction.grad.detach(),
    }


def predict(config, output, implementation):
    torch.manual_seed(1701)
    model = load_original(config, output) if implementation == "official" else adapter(config)
    native = model if implementation == "official" else model.native_model
    native.load_state_dict(torch.load(output / "shared_state.pt", weights_only=True), strict=True)
    model.eval()
    predictions = {}
    with torch.inference_mode():
        for scene, inputs in torch.load(output / "inputs.pt", weights_only=True).items():
            if implementation == "adapter":
                predictions[scene] = model(inputs)["native"]
                continue
            predictions[scene] = []
            for reference in range(4):
                current, sources = official_inputs(inputs, reference)
                # SimpleRecon 输出半分辨率；MVSAnywhere helper 使用原图分辨率。
                for fields in (current, sources):
                    for scale in range(5):
                        key = f"K_s{scale}_b44"
                        fields[key] = fields[key].clone()
                        fields[key][..., :2, :] *= 0.5
                        fields[f"invK_s{scale}_b44"] = torch.linalg.inv(fields[key])
                predictions[scene].append(model(
                    "test", current, sources,
                    unbatched_matching_encoder_forward=True, return_mask=True,
                ))
    torch.save(predictions, output / f"{implementation}.pt")
    losses = {
        case: evaluate_loss(native, case)
        for case in ("dense", "sparse_current", "missing_source")
    }
    torch.save(losses, output / f"{implementation}_losses.pt")
    write_json(output / f"{implementation}_metadata.json", {
        "source": sys.modules[type(native).__module__].__file__,
        "parameters": sum(parameter.numel() for parameter in native.parameters()),
        "initialization": "同一份四目随机权重；不是官方八目预训练权重",
    })


def compare(output):
    expected = torch.load(output / "official.pt", weights_only=True)
    actual = torch.load(output / "adapter.pt", weights_only=True)
    comparisons = []
    for scene in expected:
        for reference, (left, right) in enumerate(zip(expected[scene], actual[scene], strict=True)):
            assert left.keys() == right.keys()
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, equal_nan=True)
                comparisons.append({"scene": scene, "reference": reference, "tensor": name})
    report = {
        "status": "passed",
        "exact_tensor_comparisons": len(comparisons),
        "max_difference": 0,
        "scope": "新CARLA一帧VAL+一致合成四目，每组四参考，完整官方图；不验证训练收敛或精度",
        "details": comparisons,
    }
    write_json(output / "parity_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False))
    native_losses = torch.load(output / "official_losses.pt", weights_only=True)
    adapter_losses = torch.load(output / "adapter_losses.pt", weights_only=True)
    loss_report = {}
    for case in native_losses:
        left, right = native_losses[case], adapter_losses[case]
        assert torch.isfinite(right["gradient"]).all()
        if case == "dense":
            assert left["losses"] == right["losses"]
            torch.testing.assert_close(left["gradient"], right["gradient"], rtol=0, atol=0)
        else:
            assert left["losses"]["loss"] is None
            assert right["losses"]["loss"] is not None
        loss_report[case] = {
            "official": left["losses"],
            "adapter": right["losses"],
            "interpretation": "数值及梯度逐值相同" if case == "dense" else "官方空集合NaN，本地跳过无监督项；明确的稀疏标签扩展，不称原版全输入等价",
        }
    write_json(output / "loss_parity_report.json", loss_report)
    (output / "shared_state.pt").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "official", "adapter", "compare"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.phase == "prepare":
        create_fixture(config, args.output)
    elif args.phase == "compare":
        compare(args.output)
    else:
        predict(config, args.output, args.phase)


if __name__ == "__main__":
    main()
