"""公共预训练模型的无标签推理；数据与运行选项均由 JSON 配置。"""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import write_json
from jar.pretraining.data import dataset_config
from jar.pretraining.inference import prepare_inference
from jar.pretraining.model import PretrainingModel
from jar.runtime import move


@torch.inference_mode()
def run(config):
    torch.set_num_threads(config.get("cpu_threads", 2))
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("输出目录非空，请在配置中指定新的输出路径")
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(config["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    training = saved["config"]
    # 数据清单可独立更新；网络结构始终取自所选 checkpoint。
    data = training
    if config.get("data_config"):
        data = json.loads(Path(config["data_config"]).read_text())
    domain = next(item for item in data["datasets"] if item["name"] == config["dataset"])
    model = PretrainingModel(training, initialize=False).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    install_position_cache(model.foundation.model)
    prepare_inference(model, **config["inference"])
    # 推理加载器不加载 depth/semantics/occupancy 等标签。
    dataset = build_dataset(dataset_config(domain, data), config["split"], load_targets=False)
    reference = domain["reference_index"]
    for index in config["indices"]:
        batch = collate_frames([dataset[index]])
        inputs = move(batch["inputs"], device)
        prediction = model(inputs, domain, reference)
        record = {key: prediction[key].cpu() for key in
                  ("depth", "confidence", "valid", "support", "unknown", "observed", "state")}
        record["depth"] *= prediction["unit_scale"]
        record.update(geometry_units=domain["geometry_units"], reference_index=reference,
                      model_name=model.model_name,
                      K=inputs["K"][:, reference].cpu(),
                      T_cam_from_rig=inputs["T_cam_from_rig"][:, reference].cpu(),
                      dataset=config["dataset"], split=config["split"], index=index)
        torch.save(record, output / f"{index:06d}.pt")
        print(f"已保存 {config['dataset']} / {index}：{tuple(record['depth'].shape)}，{record['geometry_units']}", flush=True)
    write_json(output / "config.json", config)
    write_json(output / "data_config.json", {
        "image_size": data["image_size"],
        "matching_image_size": data["matching_image_size"],
        "datasets": [domain],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/current/infer_public.json")
    run(json.loads(Path(parser.parse_args().config).read_text()))
