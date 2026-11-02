"""新训练结束后分别测试best/final与固定对照；只调度既有评测器，不选择模型。"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "research/v2_evaluation_20261029"
OLD_PUBLIC = ROOT / "runs/model_releases/RulerMVS-V2/public_recommended/b411275c3203/pretrained.pt"
OLD_CARLA = ROOT / "runs/active/v2_training_20261028/carla_new/pretrained.pt"
PROTECTED = {
    OLD_PUBLIC: "b411275c3203d11e526e3f34142dbe6d3a572e5c1def71156ecb257b12b497bd",
    OLD_CARLA: "b45624ab09a176c37c17f5e78641b106fb6d0be96ee6779e8218730096088f91",
}


def read(path):
    return json.loads(Path(path).read_text())


def path(value):
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def digest(filename):
    value = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(filename, value, immutable=False):
    filename = Path(filename)
    if immutable and filename.exists():
        if read(filename) != value:
            raise ValueError(f"配置已经固定，拒绝覆盖：{filename}")
        return
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, filename)


def own(weights, mode="native"):
    return {"kind": "own", "weights": str(weights), "input_mode": mode,
            "inference": {"spatial_layout": True}}


def prepare(settings, template_dir=TEMPLATES):
    """不加载模型或深度；复制固定清单、生成独立配置，允许权重尚未产生。"""
    output = path(settings["output"])
    training_root = path(settings["training_root"])
    public_config = path(settings["public_training_config"])
    carla_config = path(settings["carla_training_config"])
    public, carla = read(public_config), read(carla_config)
    if public["image_size"] != carla["image_size"] or public["matching_image_size"] != carla["matching_image_size"]:
        raise ValueError("公共与CARLA图像协议不一致，需单独登记评测配置")
    baseline = read(template_dir / "public.json")["methods"]["mvsanywhere_refined"]
    methods = {
        "public": {
            "v2_restart_public_best": own(training_root / "public/best.pt"),
            "v2_restart_public_final": own(training_root / "public/final.pt"),
            "v2_old_public": own(OLD_PUBLIC),
            "mvsanywhere_refined": baseline,
        },
        "carla": {
            "v2_restart_public_best": own(training_root / "public/best.pt"),
            "v2_restart_carla_best": own(training_root / "carla_new/best.pt"),
            "v2_restart_carla_final": own(training_root / "carla_new/final.pt"),
            "v2_previous_carla": own(OLD_CARLA),
            "mvsanywhere_refined": baseline,
        },
    }
    if settings.get("equal_pixels", False):
        for group, method in (("public", "v2_restart_public_best"), ("carla", "v2_restart_carla_best")):
            methods[group][method + "_equal"] = own(methods[group][method]["weights"], "equal_pixels")
    groups = ["public", "carla"]
    if settings.get("include_other_references", True):
        groups.append("carla_other_refs")
    plan = {"groups": {}, "required_weights": [], "settings": settings,
            "selection": "仅完整VAL决定best，final为训练末轮；TEST不回写训练配置或默认模型。"}
    for group in groups:
        config = read(template_dir / (group + ".json"))
        config.update(output=str(output / group), gpus=settings["gpus"],
                      training_config=str(public_config), image_size=public["image_size"],
                      matching_image_size=public["matching_image_size"],
                      own_weights=str(training_root / "public/best.pt"), seed=settings["seed"])
        config["methods"] = copy.deepcopy(methods["public" if group == "public" else "carla"])
        if group == "carla_other_refs":
            config["methods"] = {name: method for name, method in config["methods"].items()
                                 if not name.endswith("_equal")}
        config["protocol_notes"] = [
            plan["selection"],
            "固定旧公共b411与上一轮新CARLA b456，旧模型及baseline不重新训练。",
            "保持20261029的固定TEST、FP32/TF32off、相同TRAIN范围和有效GT；缺相机不复制图。",
            "主图256x448、自研native匹配512x896，不宣称相同像素预算或相同预训练经历。",
            "各参考单独预测，没有场景融合；单位为scene_unit的数据不解释为米。",
        ]
        lookup = {domain["name"]: domain for domain in public["datasets"]}
        for domain in config["datasets"]:
            trained = lookup[domain["name"]] if group == "public" else carla["datasets"][0]
            # 保持既定测试划分，仅从本次TRAIN配置取范围和单位。
            for key in ("near", "far", "depth_scale", "geometry_units"):
                if domain[key] != trained[key]:
                    raise ValueError(f"{domain['name']}的{key}改变；不得静默改变历史比较口径")
            for split in ("val", "test"):
                original = Path(domain["manifests"][split])
                destination = output / "manifests" / f"{domain['name']}_{split}.jsonl"
                content = original.read_bytes()
                if destination.exists() and destination.read_bytes() != content:
                    raise ValueError(f"固定清单改变：{destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                domain["manifests"][split] = str(destination)
        filename = output / "configs" / (group + ".json")
        write_json(filename, config, immutable=True)
        item = {"config": str(filename), "output": config["output"],
                "frames": {d["name"]: sum(bool(line.strip()) for line in Path(d["manifests"]["test"]).read_text().splitlines())
                           for d in config["datasets"]}}
        if group != "carla_other_refs" and settings.get("run_speed", True):
            speed = read(template_dir / (group + "_speed.json"))
            speed.update(data_config=str(filename), output=str(output / group / "latency"),
                         own_weights=config["own_weights"], methods=config["methods"],
                         gpu=settings["timing_gpu"], seed=settings["seed"])
            speed_file = output / "configs" / (group + "_speed.json")
            write_json(speed_file, speed, immutable=True)
            item["speed_config"] = str(speed_file)
        plan["groups"][group] = item
        plan["required_weights"].extend(str(path(m["weights"])) for m in config["methods"].values())
    plan["required_weights"] = sorted(set(plan["required_weights"]))
    write_json(output / "plan.json", plan, immutable=True)
    return plan


def validate_ready(plan):
    """两段训练结束才允许读取TEST；旧保护权重必须匹配登记SHA。"""
    root = path(plan["settings"]["training_root"])
    for stage in ("public", "carla_new"):
        events = [json.loads(line) for line in (root / stage / "events.jsonl").read_text().splitlines() if line.strip()]
        if not any(row.get("event") == "complete" for row in events):
            raise RuntimeError(f"{stage}训练尚未结束，拒绝提前TEST")
    hashes = {filename: digest(filename) for filename in plan["required_weights"]}
    for filename, expected in PROTECTED.items():
        if hashes.get(str(filename)) != expected:
            raise ValueError(f"保护对照权重改变：{filename}")
    return hashes


def call(command, log):
    print("执行：", " ".join(map(str, command)), flush=True)
    with log.open("a") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def verify_existing_speed(filename):
    config = read(filename)
    folder = Path(config["output"])
    if not folder.exists():
        return False
    result, audit = read(folder / "latency.json"), read(folder / "audit.json")
    if not result.get("completed"):
        raise RuntimeError(f"测速未完成，保留现场；请使用新output重测：{folder}")
    for name, expected in audit["checkpoint_sha256"].items():
        if digest(path(name)) != expected:
            raise ValueError(f"测速权重已改变：{name}")
    for name, expected in audit["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"测速实现已改变：{name}")
    if any(audit["config"].get(key) != value for key, value in config.items()):
        raise ValueError("已存测速配置与当前配置不一致")
    return True


def summarize(plan, hashes):
    output = path(plan["settings"]["output"])
    report = {"completed": True, "weights_sha256": hashes, "groups": {},
              "selection": plan["selection"], "prediction_calls": 0}
    rows = {"depth_metrics": [], "stage_metrics": [], "surface_metrics": [], "accuracy_speed": []}
    for group, item in plan["groups"].items():
        config = read(item["config"])
        result = read(Path(item["output"]) / "full_test/summary.json")
        if not result.get("completed") or result["split"] != "test":
            raise RuntimeError(f"{group}完整TEST未完成")
        speed = read(Path(item["output"]) / "latency/latency.json") if "speed_config" in item else None
        if speed is not None and not speed.get("completed"):
            raise RuntimeError(f"{group}测速未完成")
        for dataset, values in result["datasets"].items():
            if set(values["methods"]) != set(config["methods"]):
                raise ValueError(f"{dataset}方法集合不完整")
            denominators = {m["depth"]["full"]["gt_pixels"] for m in values["methods"].values()}
            if len(denominators) != 1:
                raise ValueError(f"{dataset}各方法GT分母不同")
            for method, metrics in values["methods"].items():
                if metrics["frames"] != item["frames"][dataset]:
                    raise ValueError(f"{dataset}/{method}测试帧数不完整")
                report["prediction_calls"] += metrics["frames"]
                common = {"group": group, "dataset": dataset, "method": method,
                          "units": values["units"], "frames": metrics["frames"]}
                rows["depth_metrics"].extend(dict(common, region=region, **metric)
                                             for region, metric in metrics["depth"].items())
                rows["stage_metrics"].extend(dict(common, stage=stage, **metric)
                                             for stage, metric in metrics["stages_stride8"].items())
                rows["surface_metrics"].append(dict(common, **metrics["surface"]))
                joint = dict(common, **metrics["depth"]["full"])
                if speed is not None:
                    timed = speed["datasets"][dataset]
                    latency = timed["summary"][method]["wall_network_ms"]
                    joint.update(network_ms=latency["median_ms"], network_p90_ms=latency["p90_ms"],
                                 single_reference_hz=latency["single_reference_hz"],
                                 **timed["isolated_resources"][method])
                rows["accuracy_speed"].append(joint)
        report["groups"][group] = result
    for name, records in rows.items():
        fields = list(dict.fromkeys(key for record in records for key in record))
        with (output / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    write_json(output / "summary.json", report)
    return report


def execute(plan):
    settings = plan["settings"]
    output = path(settings["output"])
    hashes = validate_ready(plan)
    write_json(output / "weight_audit.json", hashes, immutable=True)
    python = str(path(settings.get("python", "official_methods/.venv/bin/python")))
    evaluator = "research/near_mid_evaluation_20261028/evaluate.py"
    if settings.get("run_smoke", True):
        for group, item in plan["groups"].items():
            call([python, evaluator, "--config", item["config"], "--smoke"], output / (group + "_smoke.log"))
    for group, item in plan["groups"].items():
        call([python, evaluator, "--config", item["config"]], output / (group + "_test.log"))
    # 所有并行精度进程退出后，才在同一GPU上交替计时和逐方法独占测显存。
    for group, item in plan["groups"].items():
        if "speed_config" in item and not verify_existing_speed(item["speed_config"]):
            call([python, "tools/benchmark_public_speed.py", "--config", item["speed_config"]],
                 output / (group + "_speed.log"))
    if validate_ready(plan) != hashes:
        raise RuntimeError("评测期间权重改变，不能汇报为固定权重比较")
    report = summarize(plan, hashes)
    print(f"完成：{report['prediction_calls']}次TEST预测，结果位于{output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true", help="只固定配置和清单，不读取新权重或启动GPU")
    args = parser.parse_args()
    settings = read(path(args.config))
    output = path(settings["output"])
    try:
        plan = prepare(settings)
        if args.prepare_only:
            missing = [filename for filename in plan["required_weights"] if not Path(filename).is_file()]
            print(json.dumps({"prepared": True, "gpu_tests_run": False,
                              "pending_weights": missing, "groups": plan["groups"]}, ensure_ascii=False, indent=2))
            return
        execute(plan)
    except Exception:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "failure.json", {"completed": False, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
