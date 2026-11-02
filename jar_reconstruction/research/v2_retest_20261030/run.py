"""复核独立重训的全量结果，并重新推理四参考CARLA、公共VAL和单卡测速。"""

import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from research.v2_restart_20261029.post_evaluate import validate_ready

SOURCE = ROOT / "runs/active/v2_restart_20261029/evaluation"
OUTPUT = ROOT / "runs/active/v2_retest_20261030"


def read(path):
    return json.loads(Path(path).read_text())


def call(script, config, name, *extra):
    command = [sys.executable, str(ROOT / script), "--config", str(config), *extra]
    print("执行：", " ".join(command), flush=True)
    with (OUTPUT / (name + ".log")).open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def compare(first, second):
    """只比较确定性预测统计，不比较计时或不同配置的协议hash。"""
    differences = []
    count = 0
    for domain, values in first["datasets"].items():
        for method, expected in values["methods"].items():
            actual = second["datasets"][domain]["methods"][method]
            count += actual["frames"]
            if actual != expected:
                differences.append({"dataset": domain, "method": method})
    return {"prediction_calls": count, "all_metrics_exactly_equal": not differences,
            "differences": differences}


def main():
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / "configs").mkdir()
    old = read(SOURCE / "summary.json")
    assert old["completed"]
    hashes = validate_ready(read(SOURCE / "plan.json"))
    assert hashes == old["weights_sha256"]
    source_changes = []
    for group in ("public", "carla", "carla_other_refs"):
        audit = read(SOURCE / group / "full_test/audit.json")
        for name, sha in audit["source_sha256"].items():
            if digest(ROOT / name) != sha:
                source_changes.append(name)
        config = read(SOURCE / "configs" / (group + ".json"))
        for domain in config["datasets"]:
            assert digest(domain["manifests"]["test"]) == audit["manifest_sha256"][domain["name"]]
    assert not source_changes, source_changes

    identities = {}
    for name in ("public", "carla_new"):
        folder = SOURCE.parent / name
        best = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        final = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
        assert best["model"].keys() == final["model"].keys()
        changed = [key for key in best["model"] if not torch.equal(best["model"][key], final["model"][key])]
        identities[name] = {"best_final_identical": not changed, "different_tensors": changed}
        del best, final
    write_json(OUTPUT / "audit.json", {"weights_sha256": hashes, "source_changes": source_changes,
                                      "best_final": identities, "prior_prediction_calls": old["prediction_calls"]})

    public = read(SOURCE / "configs/public.json")
    public["output"] = str(OUTPUT / "public")
    public_file = OUTPUT / "configs/public.json"
    write_json(public_file, public)
    carla = read(SOURCE / "configs/carla.json")
    carla["datasets"] += read(SOURCE / "configs/carla_other_refs.json")["datasets"]
    carla["output"] = str(OUTPUT / "carla")
    carla_file = OUTPUT / "configs/carla.json"
    write_json(carla_file, carla)
    evaluator = "research/near_mid_evaluation_20261028/evaluate.py"
    call(evaluator, public_file, "public_val", "--smoke")
    call(evaluator, carla_file, "carla_test")

    public_comparison = compare(read(SOURCE / "public/smoke_val/summary.json"),
                                read(OUTPUT / "public/smoke_val/summary.json"))
    expected = copy.deepcopy(read(SOURCE / "carla/full_test/summary.json"))
    expected["datasets"].update(read(SOURCE / "carla_other_refs/full_test/summary.json")["datasets"])
    carla_comparison = compare(expected, read(OUTPUT / "carla/full_test/summary.json"))
    write_json(OUTPUT / "reproducibility.json", {"public_val": public_comparison, "carla_test": carla_comparison})
    assert public_comparison["all_metrics_exactly_equal"] and carla_comparison["all_metrics_exactly_equal"]

    # 并行精度进程全部退出后，独占GPU0交替计时。
    upper = copy.deepcopy(carla)
    upper["datasets"] = carla["datasets"][:1]
    upper_file = OUTPUT / "configs/carla_upper.json"
    write_json(upper_file, upper)
    speed = read(SOURCE / "configs/carla_speed.json")
    speed.update(data_config=str(upper_file), output=str(OUTPUT / "latency"))
    speed_file = OUTPUT / "configs/speed.json"
    write_json(speed_file, speed)
    call("tools/benchmark_public_speed.py", speed_file, "speed")

    four = copy.deepcopy(upper)
    four["methods"] = {name: value for name, value in upper["methods"].items()
                       if name in ("v2_restart_carla_best", "v2_previous_carla", "mvsanywhere_refined")}
    four["result_file"] = str(OUTPUT / "four_reference.json")
    four_file = OUTPUT / "configs/four_reference.json"
    write_json(four_file, four)
    call("research/v2_retest_20261030/four_reference.py", four_file, "four_reference")
    assert validate_ready(read(SOURCE / "plan.json")) == hashes
    write_json(OUTPUT / "completed.json", {"completed": True, "public_val": public_comparison,
                                           "carla_test": carla_comparison, "weights_unchanged": True})
    print("复测完成：", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
