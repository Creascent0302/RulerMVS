"""V5独立评测工厂及训练后固定TEST；所有指标复用共同评测器。"""

import copy
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining import comparison
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.model import PretrainingModel

FOLDER = Path(__file__).parent
OUTPUT = ROOT / "runs/active/rulermvs_v5_20261031"
TEMPLATES = ROOT / "research/v2_evaluation_20261029"
OLD = ROOT / "runs/active/v2_restart_20261029"


def model_factory(config, initialize=True):
    if config.get("architecture") == "rulermvs_v5":
        from research.rulermvs_v5_20261031.model import V5Model

        return V5Model(config, initialize)
    return PretrainingModel(config, initialize)


# 只在本独立评测进程注册；spawn工作进程会重导入相同入口。
comparison.PretrainingModel = model_factory


def read(filename):
    return json.loads(Path(filename).read_text())


def own(filename, mode):
    return {"kind": "own", "weights": str(filename), "input_mode": mode,
            "inference": {"spatial_layout": True}}


def after_training():
    """两段训练完成后测试best和final；不会按TEST改权重或训练配置。"""
    from research.v2_restart_20261029.post_evaluate import summarize, verify_existing_speed

    for phase in ("public", "carla_new"):
        events = [json.loads(line) for line in (OUTPUT / phase / "events.jsonl").read_text().splitlines()]
        if not any(row.get("event") == "complete" for row in events):
            raise RuntimeError(f"{phase}尚未结束，拒绝提前TEST")
        for name in ("best.pt", "final.pt"):
            if not (OUTPUT / phase / name).is_file():
                raise FileNotFoundError(OUTPUT / phase / name)
    destination = OUTPUT / "evaluation"
    public = read(FOLDER / "configs/train_public.json")
    extension = {str(filename.relative_to(ROOT)): digest(filename)
                 for filename in FOLDER.glob("*.py")}
    plan = {
        "settings": {"output": str(destination)}, "groups": {}, "required_weights": [],
        "selection": "完整VAL决定best，final为实际末步；TEST不回写训练或默认模型",
        "extension_source_sha256": extension,
    }
    for mode in ("native", "equal_pixels"):
        for group in ("public", "carla", "carla_other_refs"):
            phase = "public" if group == "public" else "carla_new"
            name = group + "_" + mode
            config = read(TEMPLATES / (group + ".json"))
            baseline = copy.deepcopy(config["methods"]["mvsanywhere_refined"])
            config.update(
                output=str(destination / name), seed=20261031,
                training_config=str(FOLDER / "configs" / ("train_" + phase + ".json")),
                own_weights=str(OUTPUT / phase / "best.pt"),
                image_size=public["image_size"], matching_image_size=public["matching_image_size"],
                extension_source_sha256=extension,
            )
            config["methods"] = {
                "v5_best": own(OUTPUT / phase / "best.pt", mode),
                "v5_final": own(OUTPUT / phase / "final.pt", mode),
                "original_v2": own(OLD / phase / "best.pt", mode),
                "mvsanywhere_refined": baseline,
            }
            config["protocol_notes"] = [
                plan["selection"],
                "V5严格保留V2候选/解码，仅新增CNN到ViT连接；原V2和baseline按各自架构加载。",
                "V5继承独立V2后另有训练预算，不将全部差异归因新增模块。",
                "相同GT、TRAIN固定搜索范围、完整TEST、FP32/TF32off；缺相机不复制。",
                "native自研匹配RGB分辨率较高；equal_pixels先使用共同主图再上采样匹配图。",
                "scene_unit不解释为米；双目公共数据不冒充同步四目。",
            ]
            filename = destination / "configs" / (name + ".json")
            write_json(filename, config)
            item = {
                "config": str(filename), "output": config["output"],
                "frames": {d["name"]: sum(bool(line.strip()) for line in Path(d["manifests"]["test"]).read_text().splitlines())
                           for d in config["datasets"]},
            }
            # 两套协议计算形状相同，但各自实测；不把独立参考时间乘四作为场景时间。
            if group != "carla_other_refs":
                speed = read(TEMPLATES / (group + "_speed.json"))
                speed.update(data_config=str(filename), output=str(destination / name / "latency"),
                             own_weights=config["own_weights"], methods=config["methods"],
                             extension_source_sha256=extension)
                speed_file = destination / "configs" / (name + "_speed.json")
                write_json(speed_file, speed)
                item["speed_config"] = str(speed_file)
            plan["groups"][name] = item
            plan["required_weights"].extend(m["weights"] for m in config["methods"].values())
    plan["required_weights"] = sorted(set(plan["required_weights"]))
    hashes = {filename: digest(Path(filename) if Path(filename).is_absolute() else ROOT / filename)
              for filename in plan["required_weights"]}
    write_json(destination / "plan.json", plan)
    write_json(destination / "weight_sha256.json", hashes)
    python = str(ROOT / "official_methods/.venv/bin/python")
    wrapper = str(FOLDER / "evaluation.py")

    def call(arguments, log):
        print("执行：", " ".join(map(str, arguments)), flush=True)
        with log.open("a") as stream:
            subprocess.run(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    for name, item in plan["groups"].items():
        call([python, wrapper, "--config", item["config"], "--smoke"], destination / (name + "_smoke.log"))
        call([python, wrapper, "--config", item["config"]], destination / (name + "_test.log"))
    # 所有精度进程退出后再独占GPU0交替计时，避免并行评测污染延迟。
    for name, item in plan["groups"].items():
        if "speed_config" in item and not verify_existing_speed(item["speed_config"]):
            call([python, wrapper, "--config", item["speed_config"], "--speed"], destination / (name + "_speed.log"))
    current_hashes = {filename: digest(Path(filename) if Path(filename).is_absolute() else ROOT / filename)
                      for filename in plan["required_weights"]}
    if current_hashes != hashes or any(digest(ROOT / filename) != expected
                                       for filename, expected in extension.items()):
        raise RuntimeError("评测期间权重或V5实现发生变化，拒绝汇总为固定模型结果")
    summarize(plan, hashes)
    write_json(destination / "completed.json", {"completed": True,
               "groups": list(plan["groups"]), "test_did_not_select_model": True})


if __name__ == "__main__":
    if "--after-training" in sys.argv:
        try:
            after_training()
        except Exception:
            write_json(OUTPUT / "evaluation/failure.json", {"completed": False, "traceback": traceback.format_exc()})
            raise
    elif "--speed" in sys.argv:
        sys.argv.remove("--speed")
        from tools.benchmark_public_speed import main

        main()
    else:
        from research.near_mid_evaluation_20261028.evaluate import main

        main()
