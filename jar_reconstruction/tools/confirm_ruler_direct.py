"""固定完整权重后一次跨轨迹确认；不训练，不按TEST重新选择配置。"""

import argparse
import json
from pathlib import Path
import shlex
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import aggregate_depth, comparison_dataset, evaluate_depth, gt_reference, read_json
from jar.data import collate_frames
from jar.models.ruler_direct_model import RulerDirectModel
from jar.models.shared_initializer import SharedPretrainedInitializer
from jar.runtime import move
from tools.benchmark_ruler_models import position_caches
from tools.cache_original_initializer import save_json
from tools.carla_compare import file_sha256
from tools.evaluate_ruler_direct import validate_complete_inputs, validate_protocol


def run(plan, device):
    if file_sha256(plan["checkpoint"]) != plan["selected_checkpoint_sha256"]:
        raise ValueError("完整权重与DEV预先选定的SHA256不一致")
    output = Path(plan["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("保留已有确认结果，请勿覆盖或重复选择")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = RulerDirectModel.from_checkpoint(plan["checkpoint"]).to(device).eval()
    initializer = plan["initializer"]
    if any(initializer.get(key) for key in ("readout_weights", "fusion_weights", "decoder_weights", "fusion_skip_indices")):
        raise ValueError("确认对照使用原CARLA baseline，不混入适配权重")
    baseline = SharedPretrainedInitializer(initializer).to(device).eval()
    position_caches(model)
    position_caches(baseline)
    protocol = read_json(plan["protocol"])
    reference = validate_protocol(protocol, {"selected_direct": model, "mvsanywhere_firstpass": baseline})
    dataset = comparison_dataset(protocol, "test_complete", load_targets=True)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", plan)
    rows = {"selected_direct": [], "mvsanywhere_firstpass": []}
    for index in range(len(dataset)):
        batch = move(collate_frames([dataset[index]]), device)
        inputs = batch["inputs"]
        validate_complete_inputs(inputs, protocol)
        truth, valid = gt_reference(inputs, batch["targets"], protocol["evaluation"])
        with torch.inference_mode():
            prediction = model(inputs)["dense_depth"][0, reference]
            comparison = baseline(inputs)["depth"][0, 0]
        identity = f"{dataset.frames[index]['sequence']}/{dataset.frames[index]['frame_id']}"
        for name, depth in (("selected_direct", prediction), ("mvsanywhere_firstpass", comparison)):
            if not bool((torch.isfinite(depth[valid]) & (depth[valid] > 0)).all()):
                raise FloatingPointError("GT范围内必须有有限正深度，不能过滤失败预测")
            row = evaluate_depth(depth, torch.ones_like(valid), truth, valid, protocol["evaluation"])
            rows[name].append({"sample_id": identity, "streams": {"full": row}})
        if (index + 1) % 25 == 0:
            print(f"固定TEST确认 {index + 1}/{len(dataset)}", flush=True)
    report = {
        "plan": plan, "frames": len(dataset), "command": shlex.join([sys.executable, *sys.argv]),
        "selected_checkpoint_sha256": file_sha256(plan["checkpoint"]),
        "baseline_checkpoint_sha256": file_sha256(initializer["weights"]),
        "manifest_sha256": file_sha256(protocol["data"]["manifests"]["test_complete"]),
        "protocol_sha256": file_sha256(plan["protocol"]),
        "models": {name: aggregate_depth(records, protocol["evaluation"])["full"] for name, records in rows.items()},
        "precision": "FP32, TF32关闭", "gpu": str(device),
        "source_sha256": {name: file_sha256(ROOT / name) for name in
                          ("jar/models/ruler_direct.py", "jar/models/ruler_direct_model.py", "tools/confirm_ruler_direct.py")},
        "notes": ["选择规则与权重在DEV固定；本工具不训练、不更新权重、不选架构。",
                  "同一TEST200历史研究轨迹已被查看多次，不能称新的盲测或新地图泛化。",
                  "四目主图256x448、仅上目深度；自研匹配512x896，baseline匹配256x448。",
                  "所有GT评价，unknown不删除；无TEST调参，不以此结果反选训练步。",
                  "本工具不计时；完整网络速度见相同选定权重的DEV交替计时。"],
    }
    for name, records in rows.items():
        save_json(output / (name + "_frames.json"), records)
    save_json(output / "summary.json", report)
    print(json.dumps({name: row["all"] for name, row in report["models"].items()}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    run(read_json(args.config), torch.device(args.device))
