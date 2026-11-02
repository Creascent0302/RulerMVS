"""V5 无标签推理；结构取自checkpoint，bridge可由配置开关作推理消融。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import dataset_config
from jar.pretraining.inference import prepare_inference
from jar.runtime import move
from research.rulermvs_v5_20261031.model import V5Model


def load_model(checkpoint, device="cpu", bridge_enabled=True, inference=None):
    """独立API；返回model和checkpoint元信息，不加载任何数据或标签。"""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved.get("architecture") != "rulermvs_v5" or saved["config"].get("architecture") != "rulermvs_v5":
        raise ValueError("此入口只接受V5完整checkpoint，不能把V2或baseline误作V5")
    # 即使由inference_mode调用，也创建普通参数，保证位置编码缓存能跟踪版本。
    with torch.inference_mode(False), torch.no_grad():
        model = V5Model(saved["config"], initialize=False)
        model.load_state_dict(saved["model"], strict=True)
        model.bridge.enabled = bool(bridge_enabled)
        model.requires_grad_(False).to(device).eval()
        install_position_cache(model.foundation.model)
        prepare_inference(model, **(inference or {}))
    return model, {key: value for key, value in saved.items() if key != "model"}


def run(config):
    torch.set_num_threads(config.get("cpu_threads", 2))
    device = torch.device(config.get("device", "cuda:0"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("输出目录非空，请在配置中指定新目录")
    model, saved = load_model(config["checkpoint"], device,
                              config.get("bridge_enabled", True), config.get("inference", {}))
    data = json.loads(Path(config["data_config"]).read_text()) if config.get("data_config") else saved["config"]
    domain = next(item for item in data["datasets"] if item["name"] == config["dataset"])
    dataset = build_dataset(dataset_config(domain, data), config["split"], load_targets=False)
    output.mkdir(parents=True, exist_ok=True)
    reference = domain["reference_index"]
    with torch.inference_mode():
        for index in config["indices"]:
            batch = collate_frames([dataset[index]])
            inputs = move(batch["inputs"], device)
            prediction = model(inputs, domain, reference)
            record = {key: prediction[key].cpu() for key in
                      ("depth", "confidence", "valid", "support", "unknown", "observed", "state")}
            record["depth"] *= prediction["unit_scale"]
            record.update(
                geometry_units=domain["geometry_units"], reference_index=reference,
                model_name=model.model_name, bridge_enabled=model.bridge.enabled,
                K=inputs["K"][:, reference].cpu(),
                T_cam_from_rig=inputs["T_cam_from_rig"][:, reference].cpu(),
                dataset=config["dataset"], split=config["split"], index=index,
            )
            torch.save(record, output / f"{index:06d}.pt")
            print(f"已保存 {config['dataset']} / {index}，bridge={model.bridge.enabled}", flush=True)
    write_json(output / "config.json", config)
    write_json(output / "audit.json", {
        "checkpoint_sha256": digest(config["checkpoint"]), "load_targets": False,
        "bridge_enabled": model.bridge.enabled,
        "ablation_note": "关闭bridge仅为同权重消融；已训练底座可能变化，不等于原V2权重。",
        "data_config": {"image_size": data["image_size"],
                        "matching_image_size": data["matching_image_size"], "datasets": [domain]},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
