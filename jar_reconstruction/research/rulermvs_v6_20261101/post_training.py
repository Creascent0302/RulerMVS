"""全量训练后固定VAL/TEST对比：只编排已有评测器，不用TEST选模型。"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.v2_restart_20261029.post_evaluate import (
    call, digest, own, path, read, summarize, verify_existing_speed, write_json,
)

FOLDER = Path(__file__).parent
PHASES = ("public", "carla_new")


def require_space(output):
    output = path(output).resolve()
    if not output.is_relative_to(Path("/data")):
        raise ValueError("训练及评测产物必须经软链接存入数据盘")
    parent = output
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 200 * 2 ** 30:
        raise RuntimeError("数据盘不足200GiB，保留已有权重并停止")


def methods_for(settings, phase, transfer):
    if transfer:
        return {name: own(path(item["weights"]), "equal_pixels")
                for name, item in settings["public_transfer_models"].items()}
    root = path(settings["training_root"]) / phase
    methods = {"v6_best": own(root / "best.pt", "equal_pixels"),
               "v6_final": own(root / "final.pt", "equal_pixels")}
    for name, item in settings["historical_models"].items():
        methods[name] = own(path(item["weights"][phase]), "equal_pixels")
    methods.update(copy.deepcopy(settings["baselines"]))
    return methods


def freeze_manifests(config, output):
    """复制小型划分清单；不读RGB/深度，不按标签改变范围或选帧。"""
    counts = {"val": {}, "test": {}}
    for domain in config["datasets"]:
        for split in counts:
            source = path(domain["manifests"][split])
            destination = output / "manifests" / f"{domain['name']}_{split}.jsonl"
            content = source.read_bytes()
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError(f"固定评测清单改变：{destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            domain["manifests"][split] = str(destination)
            counts[split][domain["name"]] = sum(bool(line.strip()) for line in content.splitlines())
    return counts


def prepare_deployment(settings, output):
    """沿用已经验证的部署工具，完整accuracy仍使用eager协议。"""
    root = path(settings["training_root"])
    fast = {"compile_source": True, "compile_fusion": True,
            "cache_dir": str(root / "deploy_cache")}
    speed = read(FOLDER / "configs/speed_final.json")
    speed.update(data_config=str(path(settings["deployment"]["data_config"])),
                 v5_checkpoint=str(path(settings["historical_models"]["v5_best"]["weights"]["carla_new"])),
                 v6_checkpoint=str(root / "carla_new/best.pt"),
                 output=str(output / "deployment_speed"), fast=fast)
    validation = read(FOLDER / "configs/speed_validation.json")
    validation.update(output=str(output / "deployment_validation"), fast=fast,
                      diagnostics=str(path(settings["deployment"]["diagnostics"])))
    validation["phases"] = {
        key: {"v5_checkpoint": str(path(settings["historical_models"]["v5_best"]["weights"][phase])),
              "v6_checkpoint": str(root / phase / "best.pt")}
        for key, phase in (("public", "public"), ("carla", "carla_new"))
    }
    files = {}
    for name, config in (("speed", speed), ("validation", validation)):
        filename = output / "configs" / f"deployment_{name}.json"
        write_json(filename, config, immutable=True)
        files[name] = str(filename)
    return files


def prepare(settings):
    output = path(settings["output"])
    require_space(output)
    training = {phase: read(path(settings["training_configs"][phase])) for phase in PHASES}
    sources = list(FOLDER.glob("*.py")) + [
        ROOT / "research/baseline_full_comparison_20261101/evaluate.py",
        ROOT / "research/v5_comparison_20261101/evaluate.py",
        ROOT / "research/ruler_flow_v2_20261031/model.py",
        ROOT / "research/ruler_flow_v2_20261031/field.py",
        ROOT / "research/rulermvs_v5_20261031/model.py",
        ROOT / "research/rulermvs_v5_20261031/bridge.py",
    ]
    extension = {str(source.relative_to(ROOT)): digest(source) for source in sources}
    plan = {"settings": settings, "groups": {}, "required_weights": [],
            "selection": "训练时仅完整VAL选best；final固定末步，两者都报告，TEST不回写训练或默认模型。",
            "extension_source_sha256": extension,
            "training_config_sha256": {phase: digest(path(filename)) for phase, filename in settings["training_configs"].items()}}
    for phase in PHASES:
        original = read(path(settings["reference_audits"][phase]))["config"]
        trained = training[phase]
        if trained["input_mode"] != "equal_pixels":
            raise ValueError("本轮训练与主评测均需equal_pixels输入协议")
        for transfer in (False, True):
            name = phase + ("_v3_public_transfer" if transfer else "_main")
            config = copy.deepcopy(original)
            config.update(
                output=str(output / name), gpus=settings["gpus"], seed=settings["seed"],
                training_config=str(path(settings["training_configs"][phase])),
                image_size=trained["image_size"], matching_image_size=trained["matching_image_size"],
                own_weights=str(path(settings["training_root"]) / phase / "best.pt"),
                methods=methods_for(settings, phase, transfer), cpu_threads=settings["cpu_threads"],
                loader_workers=settings["loader_workers"], extension_source_sha256=extension,
            )
            config.pop("split", None)
            lookup = {domain["name"]: domain for domain in trained["datasets"]}
            for domain in config["datasets"]:
                source = lookup[domain["name"]] if phase == "public" else trained["datasets"][0]
                for key in ("near", "far", "depth_scale", "geometry_units"):
                    if domain[key] != source[key]:
                        raise ValueError(f"{domain['name']}的{key}不符合冻结比较协议")
            config["protocol_notes"] = [
                plan["selection"],
                "共同RGB256x448；自研匹配图仅由共同RGB上采样，无额外像素；FP32/TF32off。",
                "各域沿用既定TRAIN范围、相同GT和近中远定义，scene_unit不称米；真实双目不复制成四目。",
                "各版本预训练及追加训练预算不同；不把全部差异归因结构，也不宣称等训练预算。",
                "MVSAnywhere使用官方Hero及一次范围细化；PatchmatchNet为DTU初始化后CARLA适配，DiffMVS为随机初始化后CARLA适配。",
                "PatchmatchNet缺失相机只在适配层压紧真实有效视图；不改原生网络。",
                "CARLA TEST四参考各174完整帧；VAL真实为82/109/110/110，保留缺视图样本。",
                "V3仅有公共surface_final权重，CARLA结果是公共权重迁移，单独列组，不冒充新CARLA训练。"
                if transfer else "V2/V4/V5固定历史VAL-best；V6使用本轮公共及新CARLA各自权重。",
            ]
            counts = freeze_manifests(config, output)
            item = {"output": config["output"], "frames": counts["test"], "val_frames": counts["val"]}
            for split in ("val", "test"):
                config["evaluation_split"] = split
                filename = output / "configs" / f"{name}_{split}.json"
                write_json(filename, config, immutable=True)
                item["config" if split == "test" else "val_config"] = str(filename)
            if not transfer and settings["run_speed"]:
                speed = dict(settings["speed"], data_config=item["config"],
                             output=str(output / name / "latency"), methods=config["methods"],
                             own_weights=config["own_weights"], seed=settings["seed"],
                             cpu_threads=settings["cpu_threads"],
                             position_cache=config["position_cache"], fast_cost_volume=config["fast_cost_volume"],
                             extension_source_sha256=extension)
                filename = output / "configs" / f"{name}_speed.json"
                write_json(filename, speed, immutable=True)
                item["speed_config"] = str(filename)
            plan["groups"][name] = item
            plan["required_weights"].extend(str(path(method["weights"])) for method in config["methods"].values())
    plan["required_weights"] = sorted(set(plan["required_weights"]))
    if settings.get("deployment"):
        plan["deployment"] = prepare_deployment(settings, output)
    write_json(output / "plan.json", plan, immutable=True)
    return plan


def validate_training(plan):
    for phase in PHASES:
        folder = path(plan["settings"]["training_root"]) / phase
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines() if line.strip()]
        if not any(event.get("event") == "complete" for event in events):
            raise RuntimeError(f"{phase}未完整结束，不执行训练后TEST")
    return {filename: digest(filename) for filename in plan["required_weights"]}


def execute(plan):
    settings = plan["settings"]
    output = path(settings["output"])
    hashes = validate_training(plan)
    write_json(output / "weight_audit.json", hashes, immutable=True)
    python = str(ROOT / "official_methods/.venv/bin/python")
    wrapper = str(FOLDER / "evaluation.py")
    for name, group in plan["groups"].items():
        require_space(output)
        call([python, wrapper, "--config", group["val_config"], "--smoke"], output / f"{name}_smoke.log")
        call([python, wrapper, "--config", group["val_config"]], output / f"{name}_val.log")
    for name, group in plan["groups"].items():
        require_space(output)
        call([python, wrapper, "--config", group["config"]], output / f"{name}_test.log")
    # 精度评测全部退出后才允许独占GPU0计时；不混入并行训练/验证的速度。
    if settings["run_speed"]:
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
        ], text=True).strip()
        if processes:
            raise RuntimeError(f"仍有GPU计算进程，保留精度结果并拒绝污染测速：{processes}")
    for name, group in plan["groups"].items():
        if "speed_config" in group and not verify_existing_speed(group["speed_config"]):
            call([python, wrapper, "--config", group["speed_config"], "--speed"], output / f"{name}_speed.log")
    if "deployment" in plan:
        speed_config = plan["deployment"]["speed"]
        speed_settings = read(speed_config)
        existing = path(speed_settings["output"]) / "results.json"
        if existing.exists():
            report = read(existing)
            if not report.get("completed") or report["settings"] != speed_settings:
                raise RuntimeError("已有部署测速未完成或配置不同；保留现场，不覆盖")
            if any(digest(speed_settings[name + "_checkpoint"]) != value
                   for name, value in report["checkpoint_sha256"].items()):
                raise RuntimeError("已有部署测速的权重已改变")
        else:
            call([python, str(FOLDER / "speed_final.py"), "--config", speed_config],
                 output / "deployment_speed.log")
        # 不计时的176帧跨域数值/指标检查排在独占测速之后。
        call([python, str(FOLDER / "speed_validation.py"), "--config", plan["deployment"]["validation"]],
             output / "deployment_validation.log")
    if hashes != {filename: digest(filename) for filename in hashes}:
        raise RuntimeError("权重在评测期间改变，拒绝汇总固定模型结果")
    if any(digest(ROOT / filename) != expected for filename, expected in plan["extension_source_sha256"].items()):
        raise RuntimeError("模型或评测扩展代码改变，拒绝混用不同实现")
    summarize(plan, hashes)
    write_json(output / "completed.json", {"completed": True, "groups": list(plan["groups"]),
                                          "test_did_not_select_model": True,
                                          "accuracy_mode": "eager", "speed_mode": "eager",
                                          "deployment": plan.get("deployment"),
                                          "deployment_note": "fast速度与176VAL漂移独立记录，不替代eager完整TEST。"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare", action="store_true", help="仅冻结配置及划分清单，不加载标签或运行GPU")
    args = parser.parse_args()
    settings = read(path(args.config))
    try:
        plan = prepare(settings)
        if not args.prepare:
            execute(plan)
    except Exception:
        write_json(path(settings["output"]) / "failure.json", {"completed": False, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
