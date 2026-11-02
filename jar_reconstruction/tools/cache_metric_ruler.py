"""缓存标尺研究所需的早期逐视图证据；只读取 TRAIN/DEV，不运行重建后端。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import comparison_dataset, gt_reference, read_json
from jar.data import collate_frames
from jar.model import ReconstructionModel
from jar.runtime import move, seed_all
from tools.cache_original_initializer import save_json
from tools.carla_compare import file_sha256, make_model


class FrontendCaptured(Exception):
    """编码已完成；在粗匹配前结束原 forward，避免执行原重型后端。"""


def allowed_splits(splits):
    if not splits or set(splits) != {"train_complete", "dev_complete"}:
        raise ValueError("标尺缓存只允许完整 TRAIN66/DEV25，不允许读取 TEST")


def geometry_key(geometry):
    digest = hashlib.sha256()
    for name in sorted(geometry):
        value = geometry[name]
        digest.update(name.encode())
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def validate_reference_contract(source, expected):
    if not source.get("complete"):
        raise ValueError("旧初值缓存未完成")
    contract = source["contract"]
    for name in ("base_checkpoint_sha256", "model_config_sha256", "protocol_sha256", "reference_index"):
        if contract[name] != expected[name]:
            raise ValueError("旧初值缓存合同不匹配: " + name)
    for split, digest in expected["manifest_sha256"].items():
        if contract["manifest_sha256"][split] != digest:
            raise ValueError("旧初值缓存 manifest 不匹配: " + split)


def validate_record(record):
    inputs = record["inputs"]
    required = {"cost", "hero", "source_cosine", "source_valid", "candidate_valid",
                "matching_features", "matching_scale", "view_mask", "image_mask",
                "image_sizes"}
    if set(inputs) != required:
        raise ValueError("标尺 inputs 字段不符，禁止加入 targets 或旧深度预测")
    if inputs["cost"].shape != (1, 12, 192, 16, 28):
        raise ValueError("当前缓存约定 stride16、192 个深度候选和 12 通道代价")
    if inputs["hero"].shape != (1, 2, 192, 16, 28):
        raise ValueError("匹配统计尺寸错误")
    for name in ("source_cosine", "source_valid"):
        if inputs[name].shape != (1, 4, 192, 16, 28):
            raise ValueError("逐视图证据尺寸错误: " + name)
    if inputs["candidate_valid"].shape != (1, 192, 16, 28):
        raise ValueError("候选掩码尺寸错误")
    if inputs["matching_features"].shape != (4, 16, 128, 224):
        raise ValueError("匹配图特征必须来自 512x896 四目实际前向")
    if inputs["matching_features"].dtype != torch.float16:
        raise ValueError("磁盘匹配特征统一 FP16，读取后转 FP32")
    for name, value in inputs.items():
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise FloatingPointError("inputs 出现非有限张量: " + name)
    if inputs["view_mask"].shape != (1, 4) or inputs["image_mask"].shape != (1, 4, 256, 448):
        raise ValueError("相机/图像掩码尺寸错误")
    for name in ("view_mask", "image_mask", "source_valid", "candidate_valid"):
        if inputs[name].dtype != torch.bool:
            raise ValueError("有效性必须是布尔掩码: " + name)
    if record["targets"]["depth"].shape != (1, 1, 256, 448):
        raise ValueError("标签必须为独立全图光轴米制深度")
    if record["targets"]["valid"].shape != record["targets"]["depth"].shape:
        raise ValueError("标签掩码尺寸错误")
    if record["targets"]["valid"].dtype != torch.bool:
        raise ValueError("标签有效性必须是布尔掩码")


def load_record(root, relative_path, device="cpu", with_references=True):
    """展开共享几何和旧 raw 特征；训练输入不含 targets/baseline。"""
    root = Path(root)
    record = torch.load(root / relative_path, map_location="cpu", weights_only=True, mmap=True)
    validate_record(record)
    geometry = torch.load(root / "geometry" / (record["geometry_key"] + ".pt"),
                          map_location="cpu", weights_only=True, mmap=True)
    if geometry_key(geometry) != record["geometry_key"]:
        raise ValueError("几何文件 hash 不匹配")
    inputs = {**record["inputs"], **geometry}
    inputs["matching_features"] = inputs["matching_features"].float()
    if with_references:
        source = torch.load(record["raw_reference"], map_location="cpu", weights_only=True, mmap=True)
        if source["sample_id"] != record["sample_id"]:
            raise ValueError("旧 raw 特征身份不匹配")
        inputs["raw_maps"] = source["inputs"]["raw_maps"]
        record["baseline_initial"] = source["baseline_initial"]
        record["baseline_final"] = source["baseline_final"]
    labels_path = root / "labels" / relative_path
    if labels_path.exists():
        labels = torch.load(labels_path, map_location="cpu", weights_only=True, mmap=True)
        if labels["sample_id"] != record["sample_id"]:
            raise ValueError("多视图监督身份不匹配")
        record["targets"] = {**record["targets"], **labels["targets"]}
    return move({**record, "inputs": inputs}, device)


@torch.inference_mode()
def run(plan, device):
    allowed_splits(plan["cache_splits"])
    torch.set_num_threads(2)
    seed_all(plan.get("seed", 43))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    protocol = read_json(plan["protocol"])
    reference = protocol["evaluation"]["reference_index"]
    contract = {
        "version": 2,
        "base_checkpoint_sha256": file_sha256(plan["base"]["weights"]),
        "model_config_sha256": file_sha256(plan["base"]["model_config"]),
        "protocol_sha256": file_sha256(plan["protocol"]),
        "manifest_sha256": {split: file_sha256(protocol["data"]["manifests"][split])
                            for split in plan["cache_splits"]},
        "reference_index": reference,
        "protocol": protocol,
        "image_size": [256, 448], "matching_image_size": [512, 896],
        "precision": "编码和证据 FP32，TF32 关闭；matching_features 磁盘 FP16，读取转 FP32",
        "reference_cache": str(Path(plan["reference_cache"]).resolve()),
        "geometry": "K/T/inverse_depths 按标定、分辨率、候选范围与采样网格 hash 共享",
        "raw_centers": "7.5+16*j; candidate centers 16*j; original learned centers 4*j",
        "inference_inputs": "RGB+已知标定；虚拟格点非表面；targets 与 baseline 不进入 inputs",
        "multiview_labels": "labels/<split>/frameNNNN.pt: 独立全四目光轴米制depth/valid；仅用于训练和诊断",
    }
    old_root = Path(plan["reference_cache"]).resolve()
    old_index = read_json(old_root / "index.json")
    validate_reference_contract(old_index, contract)
    old_lookup = {split: dict(zip(old_index["sample_ids"][split], old_index["sets"][split]))
                  for split in plan["cache_splits"]}
    output = Path(plan["cache_root"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry").mkdir(exist_ok=True)
    index_path = output / "index.json"
    if index_path.exists():
        index = read_json(index_path)
        existing_contract = dict(index["contract"])
        # v1升级只补独立标签，图像证据与raw引用完全保留。
        if existing_contract.get("version") == 1:
            existing_contract["version"] = 2
            existing_contract["multiview_labels"] = contract["multiview_labels"]
        if existing_contract != contract:
            raise ValueError("已有标尺缓存合同不同，请更换输出路径")
        index["contract"] = contract
        index["complete"] = False
    else:
        index = {"contract": contract, "sets": {}, "sample_ids": {}, "complete": False}
        save_json(index_path, index)
    model = None
    captured = {}

    def capture_frontend(module, args):
        captured["work"], captured["pyramid"] = args[:2]
        raise FrontendCaptured()

    handle = None
    try:
        for split in plan["cache_splits"]:
            dataset = comparison_dataset(protocol, split, load_targets=True)
            loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=plan.get("num_workers", 2),
                                pin_memory=device.type == "cuda", collate_fn=collate_frames)
            (output / split).mkdir(exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)
            paths, identities = [], []
            for number, batch in enumerate(loader):
                frame = dataset.frames[number]
                sample_id = f"{frame['sequence']}/{frame['frame_id']}"
                if sample_id not in old_lookup[split]:
                    raise ValueError("旧初值缓存缺少同帧身份: " + sample_id)
                raw_reference = old_root / old_lookup[split][sample_id]
                old = torch.load(raw_reference, map_location="cpu", weights_only=True, mmap=True)
                if old["sample_id"] != sample_id:
                    raise ValueError("旧缓存 identity 与 manifest 不符")
                relative = f"{split}/frame{number:04d}.pt"
                destination = output / relative
                if destination.exists():
                    record = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
                    validate_record(record)
                    if record["sample_id"] != sample_id:
                        raise ValueError("已有标尺缓存 identity 不符")
                else:
                    if model is None:
                        model, _ = make_model(plan["base"], plan["base"]["weights"], device)
                        if not isinstance(model, ReconstructionModel) or model.reference_indices != (reference,):
                            raise TypeError("只允许原自研单参考 ReconstructionModel")
                        handle = model.coarse.register_forward_pre_hook(capture_frontend)
                    captured.clear()
                    inputs = move(batch["inputs"], device)
                    if not bool(inputs["view_mask"].all()):
                        raise ValueError("complete 清单出现缺相机帧")
                    try:
                        model(inputs)
                    except FrontendCaptured:
                        pass
                    else:
                        raise RuntimeError("未在粗匹配入口截获前端")
                    work, pyramid = captured["work"], captured["pyramid"]
                    # 直接 forward 避免再次触发主动结束 hook；所有编码仍只执行一次。
                    entry = model.coarse.forward(work, pyramid, stride=16, return_evidence=True)[0][reference]
                    geometry = {
                        "K": inputs["K"].cpu().contiguous(),
                        "T_cam_from_rig": inputs["T_cam_from_rig"].cpu().contiguous(),
                        "inverse_depths": entry["inverse_depths"].cpu().contiguous(),
                        "reference_index": reference, "stride": 16,
                        "image_size": [256, 448], "matching_image_size": [512, 896],
                    }
                    key = geometry_key(geometry)
                    geometry_path = output / "geometry" / (key + ".pt")
                    if not geometry_path.exists():
                        torch.save(geometry, geometry_path)
                    truth, valid = gt_reference(batch["inputs"], batch["targets"], protocol["evaluation"])
                    torch.testing.assert_close(truth[None, None], old["targets"]["depth"], rtol=0, atol=0)
                    torch.testing.assert_close(valid[None, None], old["targets"]["valid"], rtol=0, atol=0)
                    cached_inputs = {name: entry[name].cpu().contiguous() for name in
                                     ("cost", "hero", "source_cosine", "source_valid", "candidate_valid")}
                    cached_inputs.update(
                        matching_features=work["_pretrained_matching_features"][0].half().cpu().contiguous(),
                        matching_scale=work["_pretrained_matching_scale"].cpu().contiguous(),
                        view_mask=work["view_mask"].cpu().contiguous(),
                        image_mask=work["image_mask"].cpu().contiguous(),
                        image_sizes=work.get("image_sizes", torch.tensor([[[256, 448]] * 4], device=device)).cpu().contiguous(),
                    )
                    record = {
                        "sample_id": sample_id, "inputs": cached_inputs, "geometry_key": key,
                        "raw_reference": str(raw_reference),
                        "targets": {"depth": truth[None, None].contiguous(),
                                    "valid": valid[None, None].contiguous()},
                    }
                    validate_record(record)
                    temporary = destination.with_suffix(".tmp")
                    torch.save(record, temporary)
                    temporary.replace(destination)
                    del inputs, work, pyramid, entry
                    captured.clear()
                labels_path = output / "labels" / relative
                if not labels_path.exists():
                    depth = batch["targets"]["depth"].float().cpu().contiguous()
                    valid = batch["targets"]["depth_valid"].bool().cpu().contiguous()
                    if depth.shape != (1, 4, 256, 448) or valid.shape != depth.shape:
                        raise ValueError("多目标签必须为 [1,4,256,448]")
                    if not bool(torch.isfinite(depth[valid]).all()):
                        raise FloatingPointError("多目有效深度出现非有限值")
                    torch.save({"sample_id": sample_id, "targets": {
                        "multiview_depth": depth, "multiview_valid": valid,
                    }}, labels_path)
                paths.append(relative)
                identities.append(sample_id)
                index["sets"][split] = paths
                index["sample_ids"][split] = identities
                index["diagnostic_dev3_ready"] = len(index["sets"].get("dev_complete", [])) >= 3
                save_json(index_path, index)
                if number < 3 or (number + 1) % 20 == 0:
                    print(json.dumps({"split": split, "frame": number + 1, "sample_id": sample_id,
                                      "path": str(destination), "geometry_key": record["geometry_key"]}), flush=True)
            del dataset, loader
    finally:
        if handle is not None:
            handle.remove()
    if set(index["sample_ids"]["train_complete"]) & set(index["sample_ids"]["dev_complete"]):
        raise ValueError("TRAIN/DEV identity 重叠")
    if {name: len(paths) for name, paths in index["sets"].items()} != {"train_complete": 66, "dev_complete": 25}:
        raise ValueError("需要完整 TRAIN66/DEV25 缓存")
    index["complete"] = True
    if model is not None:
        index["checkpoint_load_report"] = getattr(model, "_checkpoint_load_report", None)
    index["geometry_count"] = len(list((output / "geometry").glob("*.pt")))
    index["disk_bytes"] = sum(path.stat().st_size for path in output.rglob("*.pt"))
    save_json(index_path, index)
    print(json.dumps({"complete": True, "cache_root": str(output), "geometry_count": index["geometry_count"],
                      "disk_mib": index["disk_bytes"] / 1024**2}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(read_json(args.config), torch.device(args.device))


if __name__ == "__main__":
    main()
