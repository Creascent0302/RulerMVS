"""作者 CasMVSNet 的 CARLA 单卡训练：四目输入、仅参考2有直接监督。"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import comparison_dataset, gt_reference, read_json
from jar.config import load_config
from jar.data import collate_frames
from jar.official_casmvs import OfficialCasMVSNet
from jar.runtime import move, seed_all
from third_party.cas_mvsnet.module import cas_mvsnet_loss


def native_batch(inputs, targets, settings):
    """只改数据与相机顺序；不改变作者特征、代价体或三个预测头。"""
    if inputs["rgb"].shape[0] != 1 or not inputs["view_mask"].all():
        raise ValueError("当前Cas训练协议为单卡batch1完整四目。")
    ref = settings["reference_index"]
    order = [ref] + [i for i in range(4) if i != ref]
    matrices, labels, masks = {}, {}, {}
    target, valid = gt_reference(inputs, targets, settings)
    for stage, stride in enumerate((4, 2, 1), 1):
        key = f"stage{stage}"
        packed = inputs["rgb"].new_zeros(1, 4, 2, 4, 4)
        packed[:, :, 0] = inputs["T_cam_from_rig"][:, order]
        k = inputs["K"][:, order].clone()
        k[:, :, :2] /= stride
        packed[:, :, 1, :3, :3] = k
        matrices[key] = packed
        labels[key] = target[None, ::stride, ::stride]
        masks[key] = valid[None, ::stride, ::stride]
        if not masks[key].any():
            raise ValueError("参考2在某个原生输出尺度没有有效标签；不能向官方loss传空集合。")
    return inputs["rgb"][:, order], matrices, labels, masks


def lr_multiplier(step, settings, frames_per_epoch):
    warmup = settings["warmup_steps"]
    alpha = min(step / max(warmup, 1), 1.0)
    factor = settings["warmup_factor"] * (1 - alpha) + alpha
    milestones = [epoch * frames_per_epoch for epoch in settings["decay_epochs"]]
    return factor * settings["gamma"] ** bisect_right(milestones, step)


def diagnose_failure(model, images, matrices, output, step, batch):
    """失败时只读探针：定位首个非有限代价体/深度，不修改原生forward。"""
    torch.save(
        {
            "model": model.state_dict(),
            "step": step + 1,
            "inputs": {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in batch["inputs"].items()
            },
            "targets": {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in batch["targets"].items()
            },
            "metadata": batch["metadata"],
        },
        output / "failed_state.pt",
    )

    def stats(value):
        finite = torch.isfinite(value)
        clean = value[finite]
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nonfinite": int((~finite).sum()),
            "minimum": float(clean.min()) if len(clean) else None,
            "maximum": float(clean.max()) if len(clean) else None,
            "zero": int((value == 0).sum()),
            "negative": int((value < 0).sum()),
        }

    report = {
        "step": step + 1,
        "metadata": batch["metadata"],
        "images": stats(images),
        "parameters_nonfinite": [
            name for name, p in model.named_parameters() if not torch.isfinite(p).all()
        ],
        "stages": [],
    }
    handles = []

    def depth_begin(module, args, kwargs):
        features, projections = args
        depths = kwargs["depth_values"]
        entry = {
            "features": [stats(value) for value in features],
            "candidates": stats(depths),
            "source_projected_z": [],
        }
        h, w = features[0].shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(h, device=images.device),
            torch.arange(w, device=images.device),
            indexing="ij",
        )
        pixels = torch.stack((x, y, torch.ones_like(x))).float().reshape(3, -1)[None]
        packed = projections[:, :, 0].clone()
        packed[:, :, :3] = projections[:, :, 1, :3, :3] @ projections[:, :, 0, :3]
        for source in range(1, 4):
            relative = packed[:, source] @ torch.linalg.inv(packed[:, 0])
            rays = relative[:, :3, :3] @ pixels
            z = rays[:, 2:3] * depths.flatten(2) + relative[:, 2:3, 3:4]
            entry["source_projected_z"].append(stats(z))
        report["stages"].append(entry)

    def cost_begin(module, args):
        report["stages"][-1]["cost_volume"] = stats(args[0])

    def depth_end(module, args, result):
        report["stages"][-1]["depth"] = stats(result["depth"])

    handles.append(model.net.DepthNet.register_forward_pre_hook(depth_begin, with_kwargs=True))
    handles.append(model.net.DepthNet.register_forward_hook(depth_end))
    handles.extend(
        layer.register_forward_pre_hook(cost_begin) for layer in model.net.cost_regularization
    )
    try:
        with torch.no_grad():
            model.net(images, matrices, model.depth_values[None])
    except Exception as error:
        report["diagnostic_error"] = repr(error)
    finally:
        for handle in handles:
            handle.remove()
    (output / "failure.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args(argv)
    experiment = read_json(args.config)
    protocol = read_json(experiment["protocol"])
    cfg = load_config(experiment["model_config"])
    settings = experiment["training"]
    torch.set_num_threads(protocol["runtime"]["threads"])
    seed_all(settings["seed"])
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = OfficialCasMVSNet(cfg).to(device).train()
    optimizer = torch.optim.Adam(
        model.net.parameters(),
        lr=settings["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=settings["weight_decay"],
    )
    dataset = comparison_dataset(protocol, "train_complete")
    cache = {}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("已有权重；使用新目录或明确--resume。")
    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol or saved["config"] != cfg:
            raise ValueError("续训模型或数据合同改变。")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_step = saved["step"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    (output / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    start = time.perf_counter()
    with (output / "train.jsonl").open("a" if args.resume else "w") as log:
        for step in range(start_step, settings["max_steps"]):
            epoch, offset = divmod(step, len(dataset))
            order = torch.randperm(
                len(dataset), generator=torch.Generator().manual_seed(settings["seed"] + epoch)
            )
            index = int(order[offset])
            if index not in cache:
                cache[index] = collate_frames([dataset[index]])
            batch = move(cache[index], device)
            images, matrices, labels, masks = native_batch(
                batch["inputs"], batch["targets"], protocol["evaluation"]
            )
            lr = settings["learning_rate"] * lr_multiplier(step, settings, len(dataset))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            prediction = model.net(images, matrices, model.depth_values[None])
            loss, final_loss = cas_mvsnet_loss(
                prediction, labels, masks, dlossw=settings["stage_loss_weights"]
            )
            if not torch.isfinite(loss):
                diagnose_failure(model, images, matrices, output, step, batch)
                raise FloatingPointError(f"Cas step {step}: loss不是有限值")
            loss.backward()
            gradients = [p.grad for p in model.net.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError(f"Cas step {step}: gradient不是有限值")
            optimizer.step()
            record = {
                "step": step + 1,
                "epoch": epoch,
                "dataset_index": index,
                "loss": float(loss.detach()),
                "final_loss": float(final_loss.detach()),
                "lr": lr,
                "seconds": time.perf_counter() - start,
                "gradient_tensors": len(gradients),
            }
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if (step + 1) % settings["log_every"] == 0 or step == start_step:
                print(json.dumps(record), flush=True)
            if (step + 1) % settings["checkpoint_every"] == 0 or step + 1 == settings["max_steps"]:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "config": cfg,
                    "protocol": protocol,
                    "experiment": experiment,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                }
                torch.save(state, output / "last.tmp")
                (output / "last.tmp").replace(output / "last.pt")
    # 逐张量严格回读，避免把“文件存在”当checkpoint验证。
    saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    exact = all(
        torch.equal(value.detach().cpu(), saved["model"][key])
        for key, value in model.state_dict().items()
    )
    if not exact:
        raise AssertionError("checkpoint回读不一致")
    (output / "verification.json").write_text(
        json.dumps(
            {
                "steps": saved["step"],
                "checkpoint_state_exact": exact,
                "train_frames": len(dataset),
                "source_model_sha256": hashlib.sha256(
                    (ROOT / "third_party/cas_mvsnet/cas_mvsnet.py").read_bytes()
                ).hexdigest(),
                "source_loss_sha256": hashlib.sha256(
                    (ROOT / "third_party/cas_mvsnet/module.py").read_bytes()
                ).hexdigest(),
                "limitations": [
                    "单卡FP32短预算CARLA从头训练，不是论文DTU精度复现。",
                    "仅参考2有直接标签，其他三台RGB参与原生代价体。",
                    "标签单位米，原生SmoothL1公式不变，默认转折点为1米；原论文DTU数据尺度不同。",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
