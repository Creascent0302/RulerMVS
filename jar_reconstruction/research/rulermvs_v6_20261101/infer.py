"""独立V6无标签推理，结构/权重严格从checkpoint读取，参数只由配置控制。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.rulermvs_v6_20261101.model import build_from_checkpoint
from tools.smoke_outdoor_datasets import offline_only


def load_model(checkpoint, device="cpu", inference=None):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved["architecture"] != "rulermvs_v6":
        raise ValueError("V6推理入口只加载明确标记rulermvs_v6的完整权重")
    with torch.inference_mode(False), torch.no_grad():
        model, config = build_from_checkpoint(saved)
        model.requires_grad_(False).to(device).eval()
        install_position_cache(model.foundation.model)
        if inference:
            from research.rulermvs_v6_20261101.inference import prepare_fast_inference

            prepare_fast_inference(model, inference)
    return model, config


def run(settings):
    offline_only()
    torch.set_num_threads(2)
    device = torch.device(settings["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("请在配置中指定新的输出目录")
    model, _ = load_model(settings["checkpoint"], device, settings.get("inference"))
    data = json.loads(Path(settings["data_config"]).read_text())
    domains = data.get("validation_datasets", data["datasets"])
    domain = next(d for d in domains if d["name"] == settings["dataset"])
    dataset = build_dataset(dataset_config(domain, data), settings["split"], load_targets=False)
    output.mkdir(parents=True)
    with torch.inference_mode():
        for index in settings["indices"]:
            inputs = move(collate_frames([dataset[index]])["inputs"], device)
            inputs = equal_pixel_inputs(inputs)
            prediction = model(inputs, domain, domain["reference_index"])
            record = {key: prediction[key].cpu() for key in
                      ("depth", "valid", "confidence", "support", "unknown", "observed", "state")}
            record["depth"] *= prediction["unit_scale"]
            record.update(model=model.model_name, units=domain["geometry_units"],
                          reference_index=domain["reference_index"], dataset=domain["name"],
                          index=index, K=inputs["K"].cpu(),
                          T_cam_from_rig=inputs["T_cam_from_rig"].cpu())
            torch.save(record, output / f"{index:06d}.pt")
    write_json(output / "audit.json", {"config": settings, "load_targets": False,
        "checkpoint_sha256": digest(settings["checkpoint"]), "input_mode": "equal_pixels",
        "note": "输出是假设深度，unknown/observed不冒充表面可见性认证；编译首次冷启动不属于稳态延迟"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
