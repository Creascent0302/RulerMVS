"""用独立进程对照未经修改的官方仓库与适配器，不共享 forward 实现。

准备、官方计算、适配器计算、对照四个步骤分开运行，避免 Python 模块缓存串用源码。
官方进程仅禁止初始化下载，随后严格加载同一份完整 Hero 权重。
"""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from official_methods.data import FourCameraDataset, collate
from jar.runtime import move, seed_all
from jar.synthetic import make_synthetic_batch

ROOT = Path(__file__).resolve().parents[2]


def prepare(config, output):
    dataset = FourCameraDataset(config, "val", load_targets=False)
    real = collate([dataset[0]])["inputs"]
    height, width = config["image_size"]
    synthetic = make_synthetic_batch(B=1, N=4, H=height, W=width, seed=17)["inputs"]
    torch.save({"carla": real, "synthetic": synthetic}, output / "inputs.pt")


def official_inputs(inputs, reference):
    """独立构造官方半像素中心坐标及源视图顺序，不调用适配器工具。"""
    images = inputs["rgb"]
    batch, views = images.shape[:2]
    world_from_cam = torch.linalg.inv(inputs["T_cam_from_rig"])
    source_ids = torch.tensor(
        [i for i in range(views) if i != reference], device=images.device
    ).expand(batch, -1)
    batch_ids = torch.arange(batch, device=images.device)[:, None]
    relative = inputs["T_cam_from_rig"][:, reference, None] @ world_from_cam[batch_ids, source_ids]
    translation = relative[..., :3, 3].norm(dim=-1)
    trace = relative[..., :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1)
    rotation_squared = (2 * (1 - trace / 3)).clamp_min(0)
    order = (translation.square() + rotation_squared).sqrt().argsort(dim=1, stable=True)
    source_ids = source_ids.gather(1, order)
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
    fields = {
        "image_b3hw": (images - mean) / std,
        "cam_T_world_b44": inputs["T_cam_from_rig"],
        "world_T_cam_b44": world_from_cam,
    }
    for label, scale in [("matching", 0.25)] + [(f"s{i}", 2.0**-i) for i in range(5)]:
        matrix = torch.eye(4, device=images.device).expand(batch, views, 4, 4).clone()
        matrix[..., :3, :3] = inputs["K"]
        matrix[..., :2, 2] += 0.5
        matrix[..., :2, :] *= scale
        fields[f"K_{label}_b44"] = matrix
        fields[f"invK_{label}_b44"] = torch.linalg.inv(matrix)
    current = {key: value[:, reference] for key, value in fields.items()}
    sources = {key: value[batch_ids, source_ids] for key, value in fields.items()}
    return current, sources


def load_official(config, source):
    sys.path.insert(0, str(source / "src"))
    import yaml
    from mvsanywhere.experiment_modules import sr_depth_model

    # 官方 YAML 含自己的 Options 对象标记；仅加载已固定 commit 的可信源码。
    with (source / "configs/models/mvsanywhere_model.yaml").open() as stream:
        opts = yaml.load(stream, Loader=yaml.UnsafeLoader)
    opts.image_height, opts.image_width = config["image_size"]
    opts.model_num_views = 4
    opts.fast_cost_volume = bool(config["model"].get("fast_cost_volume", False))
    opts.da_weights_path = None
    hub_load = torch.hub.load
    encoder = sr_depth_model.ResnetMatchingEncoder

    def local_hub(repository, model, *args, **kwargs):
        return hub_load(
            str(ROOT / "official_methods/upstream/dinov2"),
            model,
            source="local",
            pretrained=False,
        )

    def offline_encoder(*args, **kwargs):
        return encoder(*args, **dict(kwargs, pretrained=False))

    with (
        patch.object(torch.hub, "load", local_hub),
        patch.object(sr_depth_model, "ResnetMatchingEncoder", offline_encoder),
    ):
        model = sr_depth_model.DepthModel(opts)
    payload = torch.load(config["model"]["weights"], map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    regenerated = []
    for name, value in model.state_dict().items():
        if name.endswith("pix_coords_13N") and state[name].shape != value.shape:
            state[name] = value
            regenerated.append(name)
    model.load_state_dict(state, strict=True)
    return model, {"regenerated_pixel_grids": regenerated, "strict": True}


def predict(config, output, implementation, source, device):
    seed_all(17)
    if implementation == "official":
        model, loading = load_official(config, source)
        native = model
    else:
        from official_methods.registry import build_adapter

        config["model"]["reference_indices"] = [0, 1, 2, 3]
        model = build_adapter(config)
        native = model.native_model
        loading = model.weight_loading_report
    model.to(device).eval()
    batches = torch.load(output / "inputs.pt", weights_only=False)
    result = {}
    with torch.inference_mode():
        for name, cpu_inputs in batches.items():
            inputs = move(cpu_inputs, device)
            predictions = []
            if implementation == "official":
                for reference in range(4):
                    current, sources = official_inputs(inputs, reference)
                    prediction = model(
                        "test",
                        current,
                        sources,
                        unbatched_matching_encoder_forward=not config["model"].get(
                            "fast_cost_volume", False
                        ),
                        return_mask=True,
                        estimate_depth_range=True,
                        num_refinement_steps=config["model"]["range_refinement_steps"],
                    )
                    predictions.append(move(prediction, "cpu"))
            else:
                predictions = move(model(inputs)["native"], "cpu")
                clean = torch.cat([item["depth_pred_s0_b1hw"] for item in predictions], 1)
                poisoned = dict(
                    inputs,
                    _training_depth_range=torch.full((1, 4, 2), 99.0, device=device),
                    _geometric_flip=True,
                )
                torch.testing.assert_close(model(poisoned)["depth"].cpu(), clean, rtol=0, atol=0)
            result[name] = predictions
    torch.save(result, output / f"{implementation}.pt")
    report = {
        "implementation": implementation,
        "imported_source": sys.modules[type(native).__module__].__file__,
        "parameter_count": sum(parameter.numel() for parameter in native.parameters()),
        "state_shapes": {key: list(value.shape) for key, value in native.state_dict().items()},
        "weight_loading": loading,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
    }
    (output / f"{implementation}_metadata.json").write_text(json.dumps(report, indent=2))


def compare(output):
    official = torch.load(output / "official.pt", weights_only=False)
    adapter = torch.load(output / "adapter.pt", weights_only=False)
    details = []
    for scene in official:
        for reference, (expected, actual) in enumerate(
            zip(official[scene], adapter[scene], strict=True)
        ):
            assert expected.keys() == actual.keys()
            for key in expected:
                if not torch.is_tensor(expected[key]):
                    assert expected[key] == actual[key]
                    continue
                torch.testing.assert_close(
                    actual[key], expected[key], rtol=0, atol=0, equal_nan=True
                )
                details.append(
                    {"scene": scene, "reference": reference, "tensor": key, "max_difference": 0}
                )
    a, b = [
        json.loads((output / f"{name}_metadata.json").read_text())
        for name in ("official", "adapter")
    ]
    assert a["state_shapes"] == b["state_shapes"]
    report = {
        "status": "passed",
        "exact_tensor_comparisons": len(details),
        "parameters": a["parameter_count"],
        "official_source": a["imported_source"],
        "adapter_source": b["imported_source"],
        "details": details,
        "scope": "完整Hero权重；一帧真实CARLA与一组真实投影合成场景，各轮换4参考图；包含全部原生输出。不是论文精度复现。",
    }
    (output / "parity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "details"}, ensure_ascii=False
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "official", "adapter", "compare"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        prepare(config, args.output)
    elif args.phase == "compare":
        compare(args.output)
    else:
        predict(config, args.output, args.phase, args.official_source, args.device)


if __name__ == "__main__":
    main()
