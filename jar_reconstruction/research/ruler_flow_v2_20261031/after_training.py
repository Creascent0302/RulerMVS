"""两段训练正常结束后，固定TEST比较新best/final、原V2与官方细化baseline。"""

import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json

FOLDER = Path(__file__).parent
OUTPUT = ROOT / "runs/active/ruler_flow_v2_20261031"
TEMPLATES = ROOT / "research/v2_evaluation_20261029"
OLD = ROOT / "runs/active/v2_restart_20261029"


def read(filename):
    return json.loads(Path(filename).read_text())


def own(filename):
    return {"kind": "own", "weights": str(filename), "input_mode": "native",
            "inference": {"spatial_layout": True}}


def validate_finished():
    for phase in ("public", "carla_new"):
        events = [json.loads(line) for line in (OUTPUT / phase / "events.jsonl").read_text().splitlines()]
        if not any(row.get("event") == "complete" for row in events):
            raise RuntimeError(f"{phase}未完成，不能提前测试")
        for name in ("best.pt", "final.pt"):
            if not (OUTPUT / phase / name).is_file():
                raise FileNotFoundError(OUTPUT / phase / name)


def configurations():
    destination = OUTPUT / "evaluation"
    training = read(FOLDER / "configs/train_public.json")
    extension = {str(filename.relative_to(ROOT)): digest(filename)
                 for filename in FOLDER.glob("*.py")}
    plans = []
    for group in ("public", "carla", "carla_other_refs"):
        config = read(TEMPLATES / (group + ".json"))
        phase = "public" if group == "public" else "carla_new"
        config.update(output=str(destination / group),
                      training_config=str(FOLDER / "configs" / ("train_" + phase + ".json")),
                      own_weights=str(OUTPUT / phase / "best.pt"), seed=20261031,
                      extension_source_sha256=extension,
                      image_size=training["image_size"],
                      matching_image_size=training["matching_image_size"])
        baseline = copy.deepcopy(config["methods"]["mvsanywhere_refined"])
        config["methods"] = {
            "flow_best": own(OUTPUT / phase / "best.pt"),
            "flow_final": own(OUTPUT / phase / "final.pt"),
            "original_v2": own(OLD / phase / "best.pt"),
            "mvsanywhere_refined": baseline,
        }
        config["protocol_notes"] = [
            "新Flow通过独立工厂加载；原V2及baseline原结构不改变。",
            "相同GT分母、TRAIN范围、固定TEST和FP32/TF32off；不按测试结果选权重。",
            "原V2继承了独立训练权重，新Flow有额外训练预算，不声称纯架构因果。",
            "scene_unit不标成米；native匹配输入不同，不宣称等像素预算。",
        ]
        filename = destination / "configs" / (group + ".json")
        write_json(filename, config)
        plans.append((group, filename, config))
    write_json(destination / "plan.json", {
        "groups": {group: str(filename) for group, filename, _ in plans},
        "weight_sha256": {method["weights"]: digest(method["weights"])
                          for _, _, config in plans for method in config["methods"].values()},
        "extension_source_sha256": extension,
        "selected_by": "完整VAL；此脚本不回写训练或默认模型",
    })
    return plans


def call(arguments, log):
    print("执行：", " ".join(map(str, arguments)), flush=True)
    with log.open("a") as stream:
        subprocess.run(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    validate_finished()
    plans = configurations()
    python = str(ROOT / "official_methods/.venv/bin/python")
    wrapper = str(FOLDER / "evaluation.py")
    output = OUTPUT / "evaluation"
    for group, filename, _ in plans:
        call([python, wrapper, "--config", str(filename), "--smoke"], output / (group + "_smoke.log"))
        call([python, wrapper, "--config", str(filename)], output / (group + "_test.log"))
    # 所有精度进程退出后，才开始GPU0独占交替计时。
    for group, filename, config in plans:
        if group == "carla_other_refs":
            continue
        speed = read(TEMPLATES / (group + "_speed.json"))
        speed.update(data_config=str(filename), output=str(output / group / "latency"),
                     own_weights=config["own_weights"], methods=config["methods"],
                     extension_source_sha256=config["extension_source_sha256"])
        speed_file = output / "configs" / (group + "_speed.json")
        write_json(speed_file, speed)
        if (Path(speed["output"]) / "latency.json").exists():
            if not read(Path(speed["output"]) / "latency.json").get("completed"):
                raise RuntimeError("已有测速不完整，保留现场，需独立目录重测")
        else:
            call([python, wrapper, "--config", str(speed_file), "--speed"], output / (group + "_speed.log"))
    # 额外真实四参考调用以及共享匹配编码计时，不能把单参考乘四当测量。
    for phase in ("public", "carla_new"):
        template_name = "latency_public.json" if phase == "public" else "latency_carla.json"
        config = read(FOLDER / "configs" / template_name)
        config.update(original_checkpoint=str(OLD / phase / "best.pt"),
                      flow_checkpoint=str(OUTPUT / phase / "best.pt"),
                      all_references=True,
                      output=str(output / ("multi_reference_" + phase + ".json")))
        filename = output / "configs" / ("multi_reference_" + phase + ".json")
        write_json(filename, config)
        if not Path(config["output"]).exists():
            call([python, str(FOLDER / "latency.py"), "--config", str(filename)],
                 output / ("multi_reference_" + phase + ".log"))
    reports = {group: read(Path(config["output"]) / "full_test/summary.json")
               for group, _, config in plans}
    if not all(report.get("completed") for report in reports.values()):
        raise RuntimeError("完整评测未全部结束")
    write_json(output / "completed.json", {"completed": True, "groups": list(reports),
                                          "test_did_not_select_model": True})


if __name__ == "__main__":
    main()
