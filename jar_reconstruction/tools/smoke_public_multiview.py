"""用真实公共多视角样例检查自研及 MVSAnywhere 输入/推理，不作精度排名。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from jar.data import CarlaFrameDataset, collate_frames
from jar.models.ruler_direct_model import RulerDirectModel
from jar.models.shared_initializer import SharedPretrainedInitializer
from jar.runtime import move
from official_methods.data import FourCameraDataset, resize_sample


def preview(sample, predictions, output, reference):
    """固定 0–10 米灰度仅用于显示；原始预测另存 float32。"""
    width, height = 448, 256
    canvas = Image.new("RGB", (width * 4, (height + 26) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, rgb in enumerate(sample["inputs"]["rgb"]):
        array = (rgb.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(array).resize((width, height))
        x = index * width
        canvas.paste(image, (x, 26))
        draw.text((x + 8, 5), f"Static view {index}" + (" (reference)" if index == reference else ""), fill="black")
    maps = {"GT optical Z": sample["targets"]["depth"][reference].numpy(), **predictions}
    for index, (name, depth) in enumerate(maps.items()):
        value = np.nan_to_num(depth, nan=0, posinf=10, neginf=0)
        image = Image.fromarray((value.clip(0, 10) / 10 * 255).astype(np.uint8)).convert("RGB")
        x, y = index * width, height + 26
        canvas.paste(image.resize((width, height)), (x, y + 26))
        draw.text((x + 8, y + 5), name + " | 0=black, 10m=white", fill="black")
    draw.text((width * 3 + 8, height + 60), "Data/inference smoke only", fill="black")
    draw.text((width * 3 + 8, height + 80), "Not an unseen-scene benchmark", fill="black")
    canvas.save(output)


def run(config):
    torch.set_num_threads(2)
    torch.manual_seed(47)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config["device"])
    torch.cuda.set_device(device)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    dataset = CarlaFrameDataset(config["root"], config["manifest"], "test")
    unlabeled = CarlaFrameDataset(config["root"], config["manifest"], "test", load_targets=False)
    official = FourCameraDataset({
        "data": {"root": config["root"], "test_manifest": config["manifest"]},
        "image_size": config["image_size"],
    }, "test")
    raw = dataset[0]
    sample = resize_sample(raw, config["image_size"])
    input_only = resize_sample(unlabeled[0], config["image_size"])
    assert not input_only["targets"]
    for key, value in sample["inputs"].items():
        torch.testing.assert_close(value, input_only["inputs"][key], rtol=0, atol=0)
        torch.testing.assert_close(value, official[0]["inputs"][key], rtol=0, atol=0)

    inputs = move(collate_frames([sample])["inputs"], device)
    image_only_inputs = move(collate_frames([input_only])["inputs"], device)
    matching = resize_sample(raw, config["matching_image_size"])["inputs"]
    for target in (inputs, image_only_inputs):
        target["matching_rgb"] = matching["rgb"][None].to(device)
        target["matching_image_mask"] = matching["image_mask"][None].to(device)
    models = {
        "own_scale_transport": RulerDirectModel.from_checkpoint(config["own_checkpoint"]),
        "MVSAnywhere_firstpass": SharedPretrainedInitializer(config["initializer"]),
    }
    reference = config["reference_index"]
    report = {
        "config": config, "samples": len(dataset), "input_shape": list(inputs["rgb"].shape),
        "shared_loader_equal": True, "inputs_identical_without_targets": True, "models": {},
        "notes": [
            "静态场景不同位置的真实图像，不是同步四目驾驶数据。",
            "此场景在 MVSAnywhere 官方训练清单中，不作为未见场景泛化测试。",
            "保留现有 CARLA 权重和搜索范围；只验收接口与推理，没有室内训练或正式排名。",
        ],
    }
    predictions = {}
    for name, model in models.items():
        model = model.to(device).eval()
        assert model.reference_index == reference
        with torch.inference_mode():
            first = model(inputs)
            second = model(image_only_inputs)
        if name == "own_scale_transport":
            depth = first["dense_depth"][0, reference]
            repeated = second["dense_depth"][0, reference]
        else:
            depth = first["depth"][0, 0]
            repeated = second["depth"][0, 0]
        assert list(depth.shape) == config["image_size"]
        assert torch.isfinite(depth).all() and (depth > 0).any()
        torch.testing.assert_close(depth, repeated, rtol=0, atol=0)
        array = depth.detach().cpu().numpy()
        np.save(output / f"{name}_depth.npy", array, allow_pickle=False)
        predictions[name] = array
        report["models"][name] = {
            "forward": "passed", "shape": list(depth.shape),
            "finite_fraction": float(torch.isfinite(depth).float().mean()),
            "positive_fraction": float((depth > 0).float().mean()),
            "depth_min_m": float(depth.min()), "depth_max_m": float(depth.max()),
            "targets_removed_prediction_max_difference_m": float((depth - repeated).abs().max()),
        }
        print(json.dumps({name: report["models"][name]}), flush=True)
        model.cpu()
        del first, second
    preview(sample, predictions, output / "preview.jpg", reference)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"完成：{output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
