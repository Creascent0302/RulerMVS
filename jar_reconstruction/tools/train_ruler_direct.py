"""训练一次生成深度的标尺解码器；只缓存冻结编码器，不运行旧解码链。"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import aggregate_depth, evaluate_depth, read_json
from jar.runtime import move, seed_all
from tools.cache_metric_ruler import load_record
from tools.cache_original_initializer import save_json
from tools.carla_compare import file_sha256
from tools.train_original_initializer import clone_tensors, cpu_state
from jar.models.ruler_direct_losses import direct_loss, stage_loss, subray_distribution, subray_set_loss


def load_direct_record(root, relative_path, contract):
    """新缓存只保存直接编码器输入；旧研究缓存仍可按原合同复现。"""
    if contract.get("version") == "ruler_direct_encoder_v1":
        return torch.load(Path(root) / relative_path, map_location="cpu", weights_only=True, mmap=True)
    return load_record(root, relative_path)


def load_records(plan, device):
    index = read_json(Path(plan["cache_root"]) / "index.json")
    if not index.get("complete"):
        raise ValueError("需要已完成的冻结特征缓存")
    result = {}
    allowed = ("matching_features", "matching_scale", "raw_maps", "K", "T_cam_from_rig",
               "view_mask", "image_mask", "image_size", "reference_index")
    for split in ("train_complete", "dev_complete"):
        rows = []
        for path in index["sets"][split]:
            if "test" in path.lower():
                raise ValueError("训练不允许TEST")
            cached = load_direct_record(plan["cache_root"], path, index["contract"])
            rows.append(clone_tensors(move({
                "sample_id": cached["sample_id"],
                "inputs": {key: cached["inputs"][key] for key in allowed},
                "targets": {key: cached["targets"][key] for key in ("depth", "valid")},
            }, device)))
        result[split] = rows
    return result, index


def configure_training(model, plan):
    """完整学习或只适配已有上采样头；不增加推理层。"""
    checkpoint = plan.get("init_checkpoint", "")
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["plan"]["model"] != plan["model"]:
            raise ValueError("适配起点架构与配置不一致")
        model.load_state_dict(saved["model"], strict=True)
    scope = plan["training"].get("scope", "all")
    if scope == "readout":
        if not checkpoint:
            raise ValueError("只适配上采样头需要训练过的完整解码器")
        model.requires_grad_(False)
        model.upsample_mask.requires_grad_(True)
    elif scope != "all":
        raise ValueError("training.scope只支持all/readout")


@torch.no_grad()
def evaluate(model, records, settings):
    model.eval()
    rows = []
    for row in records:
        result = model(row["inputs"])
        truth, valid = row["targets"]["depth"][0, 0], row["targets"]["valid"][0, 0]
        streams = {}
        for name, stride in (("coarse", 8), ("fine", 4), ("full", 1)):
            depth = result["depth"] if name == "full" else result[name]["depth"]
            if not bool(torch.isfinite(depth).all()):
                raise FloatingPointError("深度存在非有限值")
            streams[name] = evaluate_depth(depth[0, 0], torch.ones_like(valid[::stride, ::stride]),
                                          truth[::stride, ::stride], valid[::stride, ::stride], settings)
        rows.append({"sample_id": row["sample_id"], "streams": streams})
    return aggregate_depth(rows, settings)


def run(plan, device):
    from jar.models.ruler_direct import RulerDirectDecoder

    seed_all(plan["training"]["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = Path(plan["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("实验输出已存在")
    records, index = load_records(plan, device)
    train, dev = records["train_complete"], records["dev_complete"]
    model = RulerDirectDecoder(plan["model"]).to(device)
    configure_training(model, plan)
    settings = plan["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    evaluation = index["contract"]["protocol"]["evaluation"]
    initial = evaluate(model, dev, evaluation)
    best, best_step, best_state = initial, 0, cpu_state(model)
    history = [{"step": 0, "metrics": initial}]
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", plan)
    source_files = ("jar/models/ruler_direct.py", "jar/models/ruler_attention.py", "jar/models/ruler_spatial.py",
                    "jar/models/metric_ruler_network.py", "jar/models/ruler_proposals.py",
                    "jar/models/ruler_direct_losses.py", "tools/train_ruler_direct.py")
    source_hashes = {name: file_sha256(ROOT / name) for name in source_files}
    for name in source_files:
        snapshot = output / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text((ROOT / name).read_text(), encoding="utf-8")
    generator = torch.Generator().manual_seed(settings["seed"])
    order = torch.randperm(len(train), generator=generator).tolist()
    identities, gradients = [], []
    started = time.perf_counter()
    with (output / "training.jsonl").open("w") as stream:
        for step in range(1, settings["steps"] + 1):
            if step > 1 and (step - 1) % len(train) == 0:
                order = torch.randperm(len(train), generator=generator).tolist()
            row = train[order[(step - 1) % len(train)]]
            identities.append(row["sample_id"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = direct_loss(model(row["inputs"]), row["targets"], plan["loss"])
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError("loss非有限")
            losses["total"].backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and (parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())):
                    raise FloatingPointError("梯度缺失/非有限: " + name)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            gradients.append(float(norm))
            stream.write(json.dumps({"step": step, "sample_id": row["sample_id"], "gradient_norm": float(norm),
                                     "loss": {key: float(value.detach()) for key, value in losses.items()}}) + "\n")
            if step % settings["eval_every"] == 0 or step == settings["steps"]:
                stream.flush()
                report = evaluate(model, dev, evaluation)
                history.append({"step": step, "metrics": report})
                if report["full"]["all"]["MAE_m"] < best["full"]["all"]["MAE_m"]:
                    best, best_step, best_state = report, step, cpu_state(model)
                save_json(output / "history.json", history)
                metrics = report["full"]["all"]
                print(json.dumps({"step": step, "DEV_MAE_m": metrics["MAE_m"],
                                  "DEV_RMSE_m": metrics["RMSE_m"], "best_step": best_step,
                                  "seconds": time.perf_counter() - started}), flush=True)
    torch.save({"model": cpu_state(model), "plan": plan, "step": settings["steps"], "optimizer": optimizer.state_dict()}, output / "last.pt")
    model.load_state_dict(best_state, strict=True)
    torch.save({"model": best_state, "plan": plan, "step": best_step}, output / "best.pt")
    restored = RulerDirectDecoder(plan["model"]).to(device).eval()
    restored.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        a, b = model.eval()(dev[0]["inputs"]), restored(dev[0]["inputs"])
        error = float((a["depth"] - b["depth"]).abs().max())
        if error != 0:
            raise AssertionError("checkpoint恢复不一致")
    report = {"plan": plan, "initial_dev": initial, "best_dev": best, "best_step": best_step,
              "train_at_best": evaluate(model, train, evaluation), "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "initial_checkpoint_sha256": file_sha256(plan["init_checkpoint"]) if plan.get("init_checkpoint") else None,
              "sample_order_sha256": hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
              "all_trainable_gradients_finite": True, "checkpoint_reload_max_error": error,
              "source_sha256": source_hashes,
              "command": shlex.join([sys.executable, *sys.argv]), "seconds": time.perf_counter() - started,
              "cache_contract": index["contract"], "no_test_read": True,
              "notes": ["只使用matching/raw/K/T/掩码，未读取旧cost/旧深度/GT作为输入。",
                        "输出连续偏移直接进入完整depth，未接旧K9/GN/DPT。"]}
    save_json(output / "summary.json", report)
    save_json(output / "sample_order.json", identities)
    print(json.dumps({"complete": str(output), "DEV_MAE": best["full"]["all"]["MAE_m"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(read_json(args.config), torch.device(args.device))
