"""在两个独立进程比较 pristine 官方源码与适配器，不共享模型实例。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def worker(source, output, pristine):
    import torch
    from official_methods.data import FourCameraDataset, collate

    torch.set_num_threads(2)
    config = json.loads((PACKAGE / "configs" / "synthetic.json").read_text())
    batch = collate([FourCameraDataset(config, "train", synthetic=True)[0]])
    inputs = batch["inputs"]
    torch.manual_seed(1701)
    if pristine:
        sys.path.insert(0, str(source))
        from models.networks.DINOv2_mvsformer_model import DINOv2MVSNet

        args = json.loads((source / "config" / "mvsformer++.json").read_text())["arch"]["args"]
        args["vit_path"] = ""
        model = DINOv2MVSNet(args).eval()
        order = [2, 0, 1, 3]
        rgb = inputs["rgb"][:, order]
        mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
        std = rgb.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
        cameras = {}
        for stage, stride in enumerate((8, 4, 2, 1), 1):
            camera = rgb.new_zeros(1, 4, 2, 4, 4)
            camera[:, :, 0] = inputs["T_cam_from_rig"][:, order]
            camera[:, :, 1, :3, :3] = inputs["K"][:, order]
            camera[:, :, 1, :2, :3] /= stride
            cameras[f"stage{stage}"] = camera
        with torch.no_grad():
            result = model((rgb - mean) / std, cameras, torch.linspace(2, 80, 192)[None])
    else:
        from official_methods.methods.mvsformerpp import Adapter

        model = Adapter(config).eval()
        with torch.no_grad():
            result = model(inputs)["native"][0]
    torch.save(result, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="固定提交的未修改官方仓库")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["pristine", "adapted"])
    args = parser.parse_args()
    if args.worker:
        worker(args.source, args.output, args.worker == "pristine")
        return

    import torch

    with tempfile.TemporaryDirectory(prefix="mvsformerpp_parity_") as temporary:
        paths = [Path(temporary) / f"{name}.pt" for name in ("pristine", "adapted")]
        for name, path in zip(("pristine", "adapted"), paths):
            subprocess.run(
                [sys.executable, __file__, "--source", str(args.source),
                 "--output", str(path), "--worker", name], check=True,
            )
        first, second = [torch.load(path, weights_only=True) for path in paths]

    checks = {}

    def compare(first, second, name=""):
        if isinstance(first, dict):
            assert first.keys() == second.keys()
            for key in first:
                compare(first[key], second[key], f"{name}.{key}")
        else:
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            checks[name] = {"shape": list(first.shape), "max_absolute_difference": 0.0}

    compare(first, second)
    report = {"device": "cpu", "random_seed": 1701, "source": str(args.source),
              "tensor_count": len(checks), "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"原始官方与适配器：{len(checks)}个原生张量完全一致")


if __name__ == "__main__":
    main()
