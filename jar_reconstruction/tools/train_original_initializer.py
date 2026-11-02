"""在原自研冻结特征上训练独立初值解码器；仅 TRAIN/DEV，无完整 Hero 前向。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import aggregate_depth, evaluate_depth
from jar.models.dense_decoder import aligned_resample
from jar.models.raw_evidence_initializer import RawEvidenceInitializer
from jar.runtime import move, seed_all


SPLITS = ("train_complete", "dev_complete")


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_tensors(value):
    """避免 inference_mode 产生的缓存张量进入需要保存输入的训练算子。"""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_tensors(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tensors(item) for item in value)
    return value


def load_cache(root, device):
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    if not index.get("complete") or index.get("model_class") != "jar.model.ReconstructionModel":
        raise ValueError("必须先完成原自研 ReconstructionModel 的训练/开发缓存")
    records = {}
    for split in SPLITS:
        paths = index["sets"][split]
        if any("test" in path.lower() for path in paths):
            raise ValueError("训练初值工具禁止读取 TEST 缓存")
        records[split] = [clone_tensors(move(torch.load(
            root / path, map_location="cpu", weights_only=True, mmap=True,
        ), device)) for path in paths]
    train_ids = {record["sample_id"] for record in records["train_complete"]}
    dev_ids = {record["sample_id"] for record in records["dev_complete"]}
    if not train_ids or not dev_ids or train_ids & dev_ids:
        raise ValueError("TRAIN/DEV 必须非空且帧身份无交集")
    return records, index


def initializer_loss(prediction, targets, weights):
    """绝对距离保证米制定位，log项兼顾近距，梯度项保留空间变化与边界。"""
    valid = targets["valid"].bool()
    if not bool(valid.any()):
        raise ValueError("训练样本没有有效深度监督")
    truth = torch.where(valid, targets["depth"].float(), 1.0)
    estimate = prediction.float()
    logarithm, target_log = estimate.clamp_min(1e-6).log(), truth.log()
    absolute = (estimate - truth).abs()[valid].mean() / weights["metric_normalizer_m"]
    log_absolute = (logarithm - target_log).abs()[valid].mean()
    gradient_terms = []
    for axis in (-1, -2):
        first, second = [slice(None)] * 4, [slice(None)] * 4
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        supported = valid[first] & valid[second]
        if bool(supported.any()):
            residual = logarithm[first] - logarithm[second] - target_log[first] + target_log[second]
            gradient_terms.append(residual.abs()[supported].mean())
    gradient = torch.stack(gradient_terms).mean() if gradient_terms else estimate.sum() * 0
    total = absolute + weights["log_weight"] * log_absolute + weights["gradient_weight"] * gradient
    return {"total": total, "metric": absolute, "log": log_absolute, "gradient": gradient}


def predict(model, record, prediction_mode="absolute", residual_log_scale=1.0):
    """监督始终独立；残差模式只额外读取原自研可推理获得的旧初值。"""
    result = model(**record["inputs"])
    return decode_prediction(result, record.get("baseline_initial"), mode=prediction_mode,
                             residual_limit=residual_log_scale)


def decode_prediction(result, base_depth=None, mode="absolute", near=2.0, far=80.0, residual_limit=1.0):
    """训练与真实RGB注入共用；base_depth是旧模型的stride4推理结果。"""
    if mode == "absolute":
        return result["depth"]
    if mode != "log_residual":
        raise ValueError("prediction_mode 必须为 absolute 或 log_residual")
    base = aligned_resample(base_depth, 4, 1, result["logits"].shape[-2:])
    correction = (result["logits"].tanh() * residual_limit).exp()
    return (base * correction).clamp(near, far)


def zero_residual_head(model):
    """仅零初始化最后的输出卷积，使初始残差为零，保留旧深度函数。"""
    final = next(layer for layer in reversed(list(model.depth_decoder.output_convs[0].modules()))
                 if isinstance(layer, torch.nn.Conv2d))
    torch.nn.init.zeros_(final.weight)
    if final.bias is not None:
        torch.nn.init.zeros_(final.bias)


def score_depth(depth, record, settings, stride):
    target, valid = record["targets"]["depth"], record["targets"]["valid"]
    if stride == 4:
        # 原自研初值格点严格为4*i，不用普通resize猜测像素中心。
        depth, target, valid = depth[..., ::4, ::4], target[..., ::4, ::4], valid[..., ::4, ::4]
    predicted = depth[0, 0]
    return evaluate_depth(predicted, torch.ones_like(valid[0, 0]), target[0, 0], valid[0, 0], settings)


@torch.no_grad()
def evaluate(model, records, settings, prediction_options=None):
    model.eval()
    rows = {"full": [], "stride4": []}
    for record in records:
        depth = predict(model, record, **(prediction_options or {}))
        if not bool(torch.isfinite(depth).all()):
            raise FloatingPointError("初值推理存在非有限输出")
        for grid, stride in (("full", 1), ("stride4", 4)):
            rows[grid].append({"sample_id": record["sample_id"], "streams": {
                "initial": score_depth(depth, record, settings, stride),
            }})
    return {grid: aggregate_depth(values, settings)["initial"] for grid, values in rows.items()}


def baseline_metrics(records, settings):
    rows = {"initial_stride4": [], "final_full": []}
    for record in records:
        truth, valid = record["targets"]["depth"][0, 0], record["targets"]["valid"][0, 0]
        for grid, depth, target, mask in (
            ("initial_stride4", record["baseline_initial"][0, 0], truth[::4, ::4], valid[::4, ::4]),
            ("final_full", record["baseline_final"][0, 0], truth, valid),
        ):
            rows[grid].append({"sample_id": record["sample_id"], "streams": {"original": evaluate_depth(
                depth, torch.ones_like(mask), target, mask, settings,
            )}})
    return {grid: aggregate_depth(values, settings)["original"] for grid, values in rows.items()}


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def gradient_statistics(model):
    groups = {"head": [], "geometry": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("可训练参数缺少有限梯度: " + name)
        groups["geometry" if name.startswith("geometry_") else "head"].append(parameter.grad)
    return {name: {"tensors": len(values), "norm": float(torch.stack([
        value.float().square().sum() for value in values
    ]).sum().sqrt()) if values else 0.0} for name, values in groups.items()}


def run(plan, device):
    output = Path(plan["output"])
    if (output / "summary.json").exists() or (output / "training.jsonl").exists():
        raise FileExistsError("已有实验结果，请使用新的输出目录: " + str(output))
    training = plan["training"]
    seed_all(training["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    records, index = load_cache(plan["cache_root"], device)
    settings = index["contract"]["protocol"]["evaluation"]
    train, dev = records["train_complete"], records["dev_complete"]
    warmstart = plan.get("init_checkpoint", "")
    state = torch.load(warmstart, map_location="cpu", weights_only=True, mmap=True)["model"] if warmstart else None
    model = RawEvidenceInitializer(plan["initializer"], bins=192, near=2, far=80, state=state).to(device)
    prediction_options = {"prediction_mode": plan.get("prediction_mode", "absolute"),
                          "residual_log_scale": plan.get("residual_log_scale", 1.0)}
    zero_initialized = prediction_options["prediction_mode"] == "log_residual" and state is None
    if zero_initialized:
        zero_residual_head(model)
    decoder = [parameter for parameter in model.depth_decoder.parameters() if parameter.requires_grad]
    decoder_rate = training.get("decoder_learning_rate", training["head_learning_rate"])
    groups = [{"params": decoder, "lr": decoder_rate, "name": "decoder"}]
    if model.use_geometry:
        geometry = list(model.geometry_encoder.parameters()) + list(model.geometry_projections.parameters())
        groups.append({"params": geometry, "lr": training["geometry_learning_rate"], "name": "geometry"})
    optimizer = torch.optim.AdamW(groups, weight_decay=training["weight_decay"])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    baseline = baseline_metrics(dev, settings)
    initial = evaluate(model, dev, settings, prediction_options)
    step0_parity = None
    if zero_initialized:
        step0_parity = 0.0
        with torch.no_grad():
            for record in dev:
                prediction = predict(model, record, **prediction_options)[..., ::4, ::4]
                original = record["baseline_initial"]
                step0_parity = max(step0_parity, float((prediction - original).abs().max()))
                torch.testing.assert_close(prediction, original, rtol=1e-5, atol=2e-4)
    rmse_limit = baseline["initial_stride4"]["all"]["RMSE_m"]
    best_mae = initial["stride4"]["all"]["MAE_m"]
    best_eligible = initial["stride4"]["all"]["RMSE_m"] <= rmse_limit
    best_step, best_report, best_state = 0, initial, cpu_state(model)
    history = [{"step": 0, "metrics": initial, "eligible": best_eligible}]
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", plan)
    save_json(output / "baseline_metrics.json", baseline)
    save_json(output / "initial_metrics.json", initial)
    generator = torch.Generator().manual_seed(training["seed"])
    order = torch.randperm(len(train), generator=generator).tolist()
    sample_order, first_gradient, last_gradient = [], None, None
    start = time.perf_counter()
    with (output / "training.jsonl").open("w") as stream:
        for step in range(1, training["steps"] + 1):
            if step > 1 and (step - 1) % len(train) == 0:
                order = torch.randperm(len(train), generator=generator).tolist()
            record = train[order[(step - 1) % len(train)]]
            sample_order.append(record["sample_id"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = initializer_loss(predict(model, record, **prediction_options), record["targets"], plan["loss"])
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError("训练损失非有限")
            losses["total"].backward()
            last_gradient = gradient_statistics(model)
            if first_gradient is None:
                first_gradient = last_gradient
            torch.nn.utils.clip_grad_norm_(trainable, training["grad_clip"])
            optimizer.step()
            loss_row = {"step": step, "sample_id": record["sample_id"],
                        "loss": {key: float(value.detach()) for key, value in losses.items()}}
            stream.write(json.dumps(loss_row) + "\n")
            if step == 50 or step % training["eval_every"] == 0 or step == training["steps"]:
                report = evaluate(model, dev, settings, prediction_options)
                metrics = report["stride4"]["all"]
                eligible = metrics["RMSE_m"] <= rmse_limit
                improves = (eligible and not best_eligible) or (
                    eligible == best_eligible and metrics["MAE_m"] < best_mae
                )
                if improves:
                    best_mae, best_step, best_report = metrics["MAE_m"], step, report
                    best_eligible, best_state = eligible, cpu_state(model)
                row = {"step": step, "metrics": report, "eligible": eligible,
                       "last_loss": loss_row["loss"], "gradient": last_gradient}
                history.append(row)
                save_json(output / "history.json", history)
                print(json.dumps({"step": step, "full": report["full"]["all"],
                                  "stride4": metrics, "eligible": eligible,
                                  "best_step": best_step}, ensure_ascii=False), flush=True)
    last_report = evaluate(model, dev, settings, prediction_options)
    torch.save({"model": cpu_state(model), "optimizer": optimizer.state_dict(),
                "plan": plan, "step": training["steps"], "order": order,
                "order_rng": generator.get_state(), "torch_rng": torch.get_rng_state()}, output / "last.pt")
    model.load_state_dict(best_state, strict=True)
    torch.save({"model": best_state, "plan": plan, "step": best_step,
                "eligible": best_eligible}, output / "best.pt")
    # 使用完整state重建模块，确保部署恢复不再依赖原始外部Hero文件。
    restored = RawEvidenceInitializer(plan["initializer"], bins=192, near=2, far=80,
                                     state=best_state).to(device).eval()
    with torch.no_grad():
        expected = predict(model.eval(), dev[0], **prediction_options)
        actual = predict(restored, dev[0], **prediction_options)
        restore_error = float((expected - actual).abs().max())
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    train_report = evaluate(model, train, settings, prediction_options)
    save_json(output / "best_metrics.json", {"train_complete": train_report, "dev_complete": best_report})
    save_json(output / "last_metrics.json", last_report)
    save_json(output / "sample_order.json", sample_order)
    report = {
        "plan": plan, "model_class": "jar.models.raw_evidence_initializer.RawEvidenceInitializer",
        "borrowed_scope": "仅预训练DPT；编码/候选来自原自研。无官方匹配MLP/第二融合ViT/完整Hero前向。",
        "protocol": index["contract"], "baseline": baseline, "initial": initial,
        "best_step": best_step, "best_eligible": best_eligible, "best": best_report, "last": last_report,
        "train_at_best": train_report, "trainable_parameters": sum(p.numel() for p in trainable),
        "selection": "DEV stride4 MAE最低且RMSE不劣原learned初值；若无合格权重，保存未合格最低MAE用于诊断",
        "comparison": "新初值stride4仅和旧learned初值stride4直接比较；旧full final含选面与细化，不冒充初值对照",
        "beats_original_initial_mae": best_mae < baseline["initial_stride4"]["all"]["MAE_m"],
        "no_test_read": True, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "FP32; TF32 disabled", "cache_all_on_device": True,
        "training_seconds": time.perf_counter() - start,
        "first_gradient": first_gradient, "last_gradient": last_gradient,
        "checkpoint_reload_max_error_m": restore_error,
        "step0_original_stride4_max_error_m": step0_parity,
        "prediction_options": prediction_options,
        "sample_order_sha256": sha256(output / "sample_order.json"),
        "source_sha256": {"tool": sha256(__file__), "module": sha256(ROOT / "jar/models/raw_evidence_initializer.py")},
        "init_checkpoint_sha256": sha256(warmstart) if warmstart else None,
    }
    save_json(output / "summary.json", report)
    shutil.copy2(__file__, output / "experiment.py")
    print(json.dumps({"output": str(output), "best_step": best_step, "eligible": best_eligible,
                      "DEV_initial_stride4": best_report["stride4"]["all"]}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), torch.device(args.device))
