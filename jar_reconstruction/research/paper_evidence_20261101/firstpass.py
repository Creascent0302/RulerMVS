"""补充MVSAnywhere首遍完整TEST及公平速度对照，独立于冻结的主评测计划。"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.v2_restart_20261029.post_evaluate import digest, path, read, write_json


FOLDER = Path(__file__).resolve().parent
WRAPPER = ROOT / "research/rulermvs_v6_20261101/evaluation.py"
PYTHON = ROOT / "official_methods/.venv/bin/python"


def prepare(settings):
    original = read(path(settings["evaluation_plan"]))
    output = path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    if not output.resolve().is_relative_to(Path("/data")):
        raise ValueError("评测输出必须保存数据盘")
    plan = {"settings": settings, "groups": {}, "weights_sha256": {}, "manifests_sha256": {},
            "reuse_audit": [], "notes": [
                "补充官方Hero首遍：仅num_refinement_steps从1改0，不替换架构或任务权重。",
                "全部TEST清单、图像、标定、TRAIN范围、GT掩码、表面采样、精度与主评测一致。",
                "四目是CARLA协议；公共实际双目数据不复制成四目。",
                "本组补充最快原生baseline，不能仅选一次细化版作为速度对手。",
            ]}
    configs = {}
    for phase in ("public", "carla_new"):
        config = copy.deepcopy(read(original["groups"][phase + "_main"]["config"]))
        first = copy.deepcopy(config["methods"]["mvsanywhere_refined"])
        first["refinements"] = 0
        config["methods"] = {"mvsanywhere_firstpass": first}
        config.update(output=str(output / phase), evaluation_split="test")
        config.pop("split", None)
        config["protocol_notes"] += plan["notes"]
        filename = output / "configs" / (phase + "_test.json")
        write_json(filename, config, immutable=True)
        configs[phase] = config
        plan["groups"][phase] = {"config": str(filename), "config_sha256": digest(filename),
                                  "output": config["output"],
                                  "frames": original["groups"][phase + "_main"]["frames"]}
        plan["weights_sha256"][first["weights"]] = digest(path(first["weights"]))
        for domain in config["datasets"]:
            filename = domain["manifests"]["test"]
            plan["manifests_sha256"][filename] = digest(path(filename))

    # 只接受精确历史协议复用；历史相同权重并不自动保证同一指标或帧集合。
    for filename in settings.get("historical_audits", []):
        old = read(path(filename))
        phase = "public" if "public" in Path(filename).parts else "carla_new"
        config = configs[phase]
        differences = []
        for field in ("image_size", "matching_image_size", "surface_queries", "tf32", "position_cache", "fast_cost_volume", "seed"):
            if old["config"].get(field) != config.get(field):
                differences.append("config:" + field)
        lookup = {d["name"]: d for d in config["datasets"]}
        for domain, expected in old["manifest_sha256"].items():
            if domain not in lookup or digest(path(lookup[domain]["manifests"]["test"])) != expected:
                differences.append("test_manifest:" + domain)
        for source in ("research/near_mid_evaluation_20261028/evaluate.py",
                       "research/near_mid_evaluation_20261028/metrics.py", "official_methods/metrics.py",
                       "jar/pretraining/comparison.py", "official_methods/methods/mvsanywhere/adapter.py"):
            if old["source_sha256"].get(source) != digest(ROOT / source):
                differences.append("source:" + source)
        old_weight = old["weights_sha256"].get("mvsanywhere_firstpass")
        weight = configs[phase]["methods"]["mvsanywhere_firstpass"]["weights"]
        if old_weight != plan["weights_sha256"][weight]:
            differences.append("weights")
        plan["reuse_audit"].append({"audit": filename, "exact_reuse_allowed": not differences,
                                    "mismatches": differences,
                                    "decision": "保留历史证据，独立重测；不改写原结果" if differences else "可核验后复用，当前调度仍独立输出"})

    speed = read(path(settings["speed_data"]))
    carla = read(original["groups"]["carla_new_main"]["config"])
    speed["methods"] = {name: copy.deepcopy(carla["methods"][name]) for name in ("v6_best", "mvsanywhere_refined")}
    speed["methods"]["mvsanywhere_firstpass"] = copy.deepcopy(configs["carla_new"]["methods"]["mvsanywhere_firstpass"])
    speed.update(own_weights=carla["own_weights"], training_config=carla["training_config"],
                 extension_source_sha256=carla["extension_source_sha256"], seed=settings["seed"])
    if speed["image_size"] != carla["image_size"] or speed["matching_image_size"] != carla["matching_image_size"]:
        raise ValueError("完整四目测速输入尺度不符合主精度协议")
    for method in speed["methods"].values():
        plan["weights_sha256"][method["weights"]] = digest(path(method["weights"]))
    domain = speed["datasets"][0]
    val = path(domain["manifests"]["val"])
    plan["manifests_sha256"][str(val)] = digest(val)
    # 使用固定完整四目VAL；此处仅读JSON，不解码RGB，更不读深度。
    rows = [json.loads(line) for line in val.read_text().splitlines() if line.strip()]
    selected = [0, len(rows) // 2, len(rows) - 1]
    wide = set(domain["camera"]["wide_ids"])
    for index in selected:
        cameras = {camera["camera_id"]: camera for camera in rows[index]["cameras"]}
        if any(not cameras.get(name, {}).get("rgb") for name in wide):
            raise ValueError("预登记测速VAL样本缺RGB，不允许静默换帧")
    datafile = output / "configs/speed_data.json"
    write_json(datafile, speed, immutable=True)
    timing = {"data_config": str(datafile), "output": str(output / "speed"),
              "gpu": settings["gpu"], "seed": settings["seed"], "warmup": settings["warmup"],
              "repeats": settings["repeats"], "include_v6_fast": settings["include_v6_fast"],
              "v6_checkpoint": carla["methods"]["v6_best"]["weights"],
              "fast": {"compile_source": True, "compile_fusion": True, "cache_dir": str(output / "compile_cache")}}
    speedfile = output / "configs/speed.json"
    write_json(speedfile, timing, immutable=True)
    plan["speed"] = {"config": str(speedfile), "config_sha256": digest(speedfile),
                      "data_config": str(datafile), "data_config_sha256": digest(datafile),
                      "val_indices": selected, "guaranteed_complete_views": 4}
    plan["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in
                             (Path(__file__), FOLDER / "firstpass_speed.py", WRAPPER)}
    write_json(output / "plan.json", plan, immutable=True)
    return plan


def verify(plan):
    for filename, expected in plan["weights_sha256"].items():
        if digest(path(filename)) != expected:
            raise ValueError("固定权重已改变：" + filename)
    for filename, expected in plan["manifests_sha256"].items():
        if digest(path(filename)) != expected:
            raise ValueError("固定清单已改变：" + filename)
    for filename, expected in plan["source_sha256"].items():
        if digest(ROOT / filename) != expected:
            raise ValueError("补充评测源码已改变：" + filename)


def execute(plan):
    verify(plan)
    output = path(plan["settings"]["output"])
    for phase, item in plan["groups"].items():
        with (output / (phase + "_test.log")).open("a") as stream:
            subprocess.run([str(PYTHON), str(WRAPPER), "--config", item["config"]], cwd=ROOT,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    # 计时阶段不允许其他GPU任务重叠；主代理应将整个补充放在其余评测之后。
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("其他GPU进程尚未退出，拒绝污染补充测速：" + active)
    result = output / "speed/results.json"
    if not result.exists():
        with (output / "speed.log").open("a") as stream:
            subprocess.run([str(PYTHON), str(FOLDER / "firstpass_speed.py"), "--config", plan["speed"]["config"]],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    if not read(result).get("completed"):
        raise RuntimeError("补充测速未完整结束，保留现场")
    verify(plan)
    write_json(output / "completed.json", {"completed": True, "groups": list(plan["groups"]),
                                           "new_training": False, "test_selected_model": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(FOLDER / "firstpass.json"))
    parser.add_argument("--prepare", action="store_true", help="仅CPU冻结配置与hash，不读取RGB/深度或运行GPU")
    args = parser.parse_args()
    plan = prepare(read(path(args.config)))
    if args.prepare:
        print(json.dumps({"prepared": True, "frames": {p: sum(v["frames"].values()) for p, v in plan["groups"].items()},
                          "speed": plan["speed"], "reuse_audit": plan["reuse_audit"]}, ensure_ascii=False, indent=2))
    else:
        execute(plan)
