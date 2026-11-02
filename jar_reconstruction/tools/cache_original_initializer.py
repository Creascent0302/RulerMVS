"""只缓存原自研初值生成所需特征；训练/验证标签在真实前向结束后单独保存。"""

import argparse
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
from tools.carla_compare import file_sha256, make_model


def save_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_record(record, sample_id):
    if record["sample_id"] != sample_id:
        raise ValueError("缓存帧身份与当前manifest不符")
    inputs = record["inputs"]
    if set(inputs) != {"raw_maps", "packet", "coarse_stride", "output_size"}:
        raise ValueError("缓存inputs只允许原始特征及网格元数据，不允许标签或旧预测")
    if len(inputs["raw_maps"]) != 4 or inputs["coarse_stride"] != 16:
        raise ValueError("当前缓存约定四层raw特征及stride16候选")
    if inputs["output_size"] != [256, 448] or inputs["packet"].shape != (1, 773, 16, 28):
        raise ValueError("当前缓存固定256x448/192深度候选配置")
    values = [inputs["packet"], *inputs["raw_maps"],
              record["baseline_initial"], record["baseline_final"]]
    if any(value.shape != (1, 768, 16, 28) for value in inputs["raw_maps"]):
        raise ValueError("raw_maps必须是四个[1,768,16,28]张量")
    if record["baseline_initial"].shape != (1, 1, 64, 112):
        raise ValueError("baseline_initial必须为[1,1,64,112]")
    if record["baseline_final"].shape != (1, 1, 256, 448):
        raise ValueError("baseline_final必须为[1,1,256,448]")
    target = record["targets"]
    if target["depth"].shape != (1, 1, 256, 448) or target["valid"].shape != target["depth"].shape:
        raise ValueError("targets必须是全图[1,1,256,448]")
    if not bool(target["valid"].any()):
        raise ValueError("完整监督帧必须包含有效参考相机GT")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("原模型缓存出现非有限特征或预测")
    if not bool(torch.isfinite(target["depth"][target["valid"]]).all()):
        raise FloatingPointError("有效监督深度必须有限")


@torch.inference_mode()
def run(plan, device):
    splits = plan["cache_splits"]
    if not splits or set(splits) - {"train_complete", "dev_complete"}:
        raise ValueError("初值训练缓存禁止读取TEST；仅允许train_complete/dev_complete")
    torch.set_num_threads(2)
    seed_all(plan.get("seed", 43))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    protocol = read_json(plan["protocol"])
    reference = protocol["evaluation"]["reference_index"]
    contract = {
        "version": 1,
        "base_checkpoint_sha256": file_sha256(plan["base"]["weights"]),
        "model_config": read_json(plan["base"]["model_config"]),
        "model_config_sha256": file_sha256(plan["base"]["model_config"]),
        "protocol": protocol,
        "protocol_sha256": file_sha256(plan["protocol"]),
        "manifest_sha256": {split: file_sha256(protocol["data"]["manifests"][split])
                            for split in splits},
        "reference_index": reference,
        "precision": "FP32; TF32 disabled",
        "raw_centers": "7.5+16*j; learned initial centers 4*j; final centers j",
        "input_keys": ["raw_maps", "packet", "coarse_stride", "output_size"],
    }
    root = Path(plan["cache_root"])
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    if index_path.exists():
        index = read_json(index_path)
        if index["contract"] != contract:
            raise ValueError("缓存权重、模型配置、协议或manifest已变化；禁止复用")
    else:
        index = {"contract": contract, "sets": {}, "sample_ids": {}, "complete": False}
        save_json(index_path, index)
    model, cfg = make_model(plan["base"], plan["base"]["weights"], device)
    if not isinstance(model, ReconstructionModel) or hasattr(model, "initializer"):
        raise TypeError("只能缓存原自研ReconstructionModel，不允许完整MVSAnywhere通路")
    if model.reference_indices != (reference,):
        raise ValueError("缓存要求只执行协议中的单个参考相机")
    captured = {}

    def raw_hook(module, args, outputs):
        captured["raw_maps"] = [value[reference:reference + 1] for value in outputs]

    def packet_hook(module, args):
        captured["packet"] = args[0]

    original_recover = model.dense_decoder._recover_initial

    def capture_initial(depth, valid, inputs):
        captured["baseline_initial"] = depth
        return original_recover(depth, valid, inputs)

    handles = [model.encoder.foundation.register_forward_hook(raw_hook),
               model.dense_decoder.candidate_encoder.register_forward_pre_hook(packet_hook)]
    model.dense_decoder._recover_initial = capture_initial
    try:
        for split in splits:
            dataset = comparison_dataset(protocol, split, load_targets=True)
            loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=plan.get("num_workers", 2),
                                pin_memory=device.type == "cuda", collate_fn=collate_frames)
            paths, identities = [], []
            (root / split).mkdir(exist_ok=True)
            for number, batch in enumerate(loader):
                frame = dataset.frames[number]
                sample_id = f"{frame['sequence']}/{frame['frame_id']}"
                relative = f"{split}/frame{number:04d}.pt"
                destination = root / relative
                if destination.exists():
                    record = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
                    validate_record(record, sample_id)
                else:
                    inputs = move(batch["inputs"], device)
                    if not bool(inputs["view_mask"].all()):
                        raise ValueError("complete清单出现缺相机帧")
                    captured.clear()
                    output = model(inputs)
                    # 此时才根据同帧targets与共同ROI构造监督；绝不送进model(inputs)。
                    truth, valid = gt_reference(batch["inputs"], batch["targets"], protocol["evaluation"])
                    record = {
                        "sample_id": sample_id,
                        "inputs": {
                            "raw_maps": [value.detach().cpu().contiguous() for value in captured["raw_maps"]],
                            "packet": captured["packet"].detach().cpu().contiguous(),
                            "coarse_stride": 16,
                            "output_size": list(inputs["rgb"].shape[-2:]),
                        },
                        "baseline_initial": captured["baseline_initial"].detach().cpu().contiguous(),
                        "baseline_final": output["dense_depth"][:, reference:reference + 1].detach().cpu().contiguous(),
                        "targets": {"depth": truth[None, None].contiguous(),
                                    "valid": valid[None, None].contiguous()},
                    }
                    validate_record(record, sample_id)
                    temporary = destination.with_suffix(".tmp")
                    torch.save(record, temporary)
                    temporary.replace(destination)
                    del output, inputs
                paths.append(relative)
                identities.append(sample_id)
                if number == 0 or (number + 1) % 20 == 0:
                    print(f"{split}: {number + 1}/{len(dataset)} {sample_id}", flush=True)
            index["sets"][split] = paths
            index["sample_ids"][split] = identities
            index["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
            index["checkpoint_load_report"] = getattr(model, "_checkpoint_load_report", None)
            save_json(index_path, index)
        if set(index["sample_ids"]["train_complete"]) & set(index["sample_ids"]["dev_complete"]):
            raise ValueError("训练和开发帧身份重叠")
        index["complete"] = True
        save_json(index_path, index)
        print(json.dumps({"cache": str(root), "counts": {k: len(v) for k, v in index["sets"].items()},
                          "model_class": index["model_class"]}, ensure_ascii=False), flush=True)
    finally:
        model.dense_decoder._recover_initial = original_recover
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(read_json(args.config), torch.device(args.device))
