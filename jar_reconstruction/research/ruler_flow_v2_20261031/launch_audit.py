"""正式八卡启动前汇总已运行证据、冻结源码与原权重指纹。"""

import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json

FOLDER = Path(__file__).parent
OUTPUT = ROOT / "runs/active/ruler_flow_v2_20261031"


def read(filename):
    return json.loads(Path(filename).read_text())


def main():
    plan = read(OUTPUT / "training_plan.json")
    original = {filename: digest(filename) for filename in plan["original_weights"]}
    if original != plan["original_weights"]:
        raise RuntimeError("原V2权重被改变")
    smokes = {}
    for phase, expected in (("public", 21), ("carla_new", 2)):
        directory = OUTPUT / ("smoke_" + phase)
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        updates = [event for event in events if event["event"] == "update"]
        if not any(event["event"] == "complete" for event in events) or len(updates) != expected:
            raise RuntimeError(f"八卡{phase}冒烟未完整结束")
        if read(directory / "config.json")["train"]["precision"] != "fp32":
            raise RuntimeError("smoke精度模式不一致")
        starts = [event for event in events if event["event"] == "start"]
        if starts[-1]["world_size"] != 8:
            raise RuntimeError("必须是实际八卡冒烟")
        smokes[phase] = {"updates": len(updates), "exposures": updates[-1]["exposures"],
                         "peak_rank0_vram_gib": max(event["peak_vram_gib"] for event in updates),
                         "stages": sorted({event["stage"] for event in updates})}
    results = {name: read(OUTPUT / name) for name in (
        "pilot_summary.json", "resume_verification.json", "initial_metadata_update.json",
        "latency_public.json", "latency_carla.json")}
    if not results["pilot_summary.json"]["completed"]:
        raise RuntimeError("短试未完成")
    adaptation = read(OUTPUT / "adapter_validation/smoke_val/summary.json")
    if not adaptation["completed"] or adaptation["split"] != "val":
        raise RuntimeError("新架构评测适配未经过真实VAL验证")
    free = shutil.disk_usage(OUTPUT).free / 2 ** 30
    if free < 204:
        raise RuntimeError("剩余空间不足以保留200GiB")
    sources = list(FOLDER.rglob("*.py")) + list((FOLDER / "configs").glob("*.json"))
    sources += list(FOLDER.glob("*.md")) + list(FOLDER.glob("*.sh"))
    sources += list((ROOT / "jar").rglob("*.py"))
    for relative in ("research/v2_training_20261028", "research/v2_restart_20261029",
                     "research/near_evidence_20261030", "research/near_mid_20261027",
                     "research/near_mid_evaluation_20261028"):
        sources += list((ROOT / relative).glob("*.py"))
    sources += [ROOT / name for name in (
        "tools/with_server_network.sh", "tools/benchmark_public_speed.py",
        "tools/compare_public_pretrained.py", "tools/smoke_outdoor_datasets.py",
        "docs/research_memory.md")]
    archive_path = OUTPUT / "source_snapshot.zip"
    source_hashes = {str(filename.relative_to(ROOT)): digest(filename) for filename in sorted(set(sources))}
    if archive_path.exists():
        raise FileExistsError("正式源码快照已存在，不能覆盖")
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in source_hashes:
            archive.write(ROOT / name, name)
    write_json(OUTPUT / "launch_audit.json", {
        "eight_gpu_smokes": smokes, "original_weights_unchanged": original,
        "resume": results["resume_verification.json"],
        "pilot_complete": True, "adapter_validation_complete": True,
        "source_sha256": source_hashes, "free_gib": free,
        "training_plan": plan,
        "limits": "短试与程序验证不保证全量训练优于旧V2或baseline；不自动覆盖旧默认模型。",
    })
    print(json.dumps({"smokes": smokes, "free_gib": free, "source_files": len(source_hashes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
