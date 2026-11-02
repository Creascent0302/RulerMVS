"""只读核验本轮V4训练完整性、VAL选模及旧资源保持，不执行推理或训练。"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/active/ruler_flow_v2_20261031"


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def beijing(value):
    return timestamp(value).astimezone(ZoneInfo("Asia/Shanghai")).isoformat()


def same(first, second):
    if first.keys() != second.keys():
        raise ValueError("模型state键集合不同")
    differences = {}
    for name, value in first.items():
        other = second[name]
        if not torch.equal(value, other):
            differences[name] = float((value.double() - other.double()).abs().max())
    return {"all_tensors_exact": not differences, "changed_tensors": len(differences),
            "max_abs": max(differences.values(), default=0.), "differences": differences}


def phase_audit(phase, plan):
    directory = OUTPUT / phase
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    config = read(directory / "config.json")
    best, final, last = [load(directory / name) for name in ("best.pt", "final.pt", "last.pt")]
    exit_code = int((directory / "exit_code.txt").read_text())
    if exit_code != 0:
        raise RuntimeError(f"{phase}退出码非0")
    start = next(row for row in events if row["event"] == "start")
    finish = next(row for row in reversed(events) if row["event"] == "complete")
    full = [row for row in events if row["event"] == "validation" and row["full"]]
    expected_epochs = [(stage["number"], epoch) for stage in config["stages"]
                       for epoch in range(1, stage["max_epochs"] + 1)]
    if [(row["stage"], row["epoch"]) for row in full] != expected_epochs:
        raise RuntimeError("完整VAL未覆盖配置中的全部阶段/轮数")
    selected = min(full, key=lambda row: row["selection_risk"])
    if (best["stage"], best["epoch"], best["selection_score"]) != (
            selected["stage"], selected["epoch"], selected["selection_risk"]):
        raise RuntimeError("best不是本轮完整VAL最低预登记风险")
    state = last["state"]
    if state["global_step"] != plan["total_updates"] or final["global_step"] != plan["total_updates"]:
        raise RuntimeError("实际训练更新数与预登记预算不符")
    if state["exposures"] != plan["total_exposures"] or state["global_epoch"] != plan["epochs"]:
        raise RuntimeError("实际训练曝光或轮数与预算不符")
    if start["world_size"] != 8 or state["stage_index"] != len(config["stages"]):
        raise RuntimeError("训练非八卡或阶段未完成")
    if any(row["event"] in ("early_stop", "stage_skipped", "final_test", "zero_shot_test") for row in events):
        raise RuntimeError("正式训练出现未经计划的提前停止、阶段跳过或TEST")
    if not all(row["selection"] == "last_parameters_no_rollback"
               for row in events if row["event"] == "stage_complete"):
        raise RuntimeError("阶段间并非连续训练")
    if best["architecture"] != "ruler_flow_v2" or final["architecture"] != "ruler_flow_v2":
        raise RuntimeError("新模型架构ID不符")
    finite = all(bool(torch.isfinite(value).all()) for value in final["model"].values())
    if not finite:
        raise RuntimeError("final模型含非有限state")
    initial = load(config["initial_model_checkpoint"])
    prefix = "decoder.flow_field."
    initial_head = {key: value for key, value in initial["model"].items() if key.startswith(prefix)}
    best_head = {key: value for key, value in best["model"].items() if key.startswith(prefix)}
    head_updates = same(initial_head, best_head)
    if head_updates["all_tensors_exact"]:
        raise RuntimeError("新速度场未更新")
    durations = []
    for stage in config["stages"]:
        begin = next(row for row in events if row["event"] == "stage_start" and row["stage"] == stage["number"])
        end = next(row for row in events if row["event"] == "stage_complete" and row["stage"] == stage["number"])
        duration = (timestamp(end["time_utc"]) - timestamp(begin["time_utc"])).total_seconds()
        durations.append({"stage": stage["number"], "epochs": stage["max_epochs"],
                          "trainable_parameters": begin["trainable_parameters"],
                          "start_beijing": beijing(begin["time_utc"]),
                          "end_beijing": beijing(end["time_utc"]), "seconds": duration})
    result = {
        "completed": True, "exit_code": exit_code, "world_size": start["world_size"],
        "start_beijing": beijing(start["time_utc"]), "end_beijing": beijing(finish["time_utc"]),
        "event_elapsed_seconds": (timestamp(finish["time_utc"]) - timestamp(start["time_utc"])).total_seconds(),
        "stages": durations, "global_epoch": state["global_epoch"], "global_step": state["global_step"],
        "exposures": state["exposures"], "full_validation_epochs": len(full),
        "best": {key: best[key] for key in ("stage", "epoch", "global_step", "selection_score", "selected")},
        "best_validation_split": best["validation"]["split"],
        "best_final_comparison": same(best["model"], final["model"]),
        "final_last_comparison": same(final["model"], last["model"]),
        "head_initial_best_comparison": head_updates,
        "head_parameters": sum(value.numel() for value in best_head.values()),
        "final_state_finite": finite,
        "flow": config["flow"],
        "checkpoint_sha256": {name: sha256(directory / name) for name in ("best.pt", "final.pt", "last.pt")},
        "train_manifests_unchanged": {
            domain["name"]: sha256(domain["manifests"]["train"]) == domain["train_manifest_sha256"]
            for domain in config["datasets"]},
        "initial_model_checkpoint": config["initial_model_checkpoint"],
        "initial_model_sha256": sha256(config["initial_model_checkpoint"]),
    }
    if not all(result["train_manifests_unchanged"].values()):
        raise RuntimeError("训练清单在训练后改变")
    return result


def main():
    torch.set_num_threads(2)
    plan = read(OUTPUT / "training_plan.json")
    phases = {phase: phase_audit(phase, plan[phase]) for phase in ("public", "carla_new")}
    old = {filename: sha256(filename) == expected for filename, expected in plan["original_weights"].items()}
    if not all(old.values()):
        raise RuntimeError("原模型文件SHA改变")
    queue_code = int((OUTPUT / "queue.exit_code.txt").read_text())
    if queue_code != 0:
        raise RuntimeError("完整队列未正常结束")
    if phases["carla_new"]["initial_model_sha256"] != phases["public"]["checkpoint_sha256"]["best.pt"]:
        raise RuntimeError("CARLA迁移源不是当前新公共best")
    report = {"completed": True, "display_version": "V4", "architecture": "ruler_flow_v2",
              "queue_exit_code": queue_code, "phases": phases, "original_weights_unchanged": old,
              "limits": "训练完整不证明精度提升；best按完整VAL选取，未用TEST选模型。该审计仅读取已有训练记录与权重。"}
    destination = OUTPUT / "verification" / "training_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(destination), "phases": {
        name: {key: value for key, value in result.items() if key in (
            "global_step", "global_epoch", "exposures", "best", "start_beijing", "end_beijing",
            "event_elapsed_seconds", "head_parameters")}
        for name, result in phases.items()}, "old_weights_unchanged": all(old.values())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
