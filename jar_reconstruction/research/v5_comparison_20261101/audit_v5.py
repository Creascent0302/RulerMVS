"""复核已完成的V5全TEST证据，不重新选择权重或改变测试划分。"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    folder = ROOT / "runs/active/rulermvs_v5_20261031"
    evaluation = folder / "evaluation"
    hashes, cache = read(evaluation / "weight_sha256.json"), {}

    def verify(path, expected):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        if str(path) not in cache:
            cache[str(path)] = digest(path)
        if cache[str(path)] != expected:
            raise RuntimeError(f"证据文件已变化：{path}")

    for path, expected in hashes.items():
        verify(path, expected)
    plan, summary = read(evaluation / "plan.json"), read(evaluation / "summary.json")
    counts = {}
    for name, group in plan["groups"].items():
        audit = read(Path(group["output"]) / "full_test/audit.json")
        for path, expected in audit["source_sha256"].items():
            verify(path, expected)
        for path, expected in audit["config"]["extension_source_sha256"].items():
            verify(path, expected)
        report = summary["groups"][name]
        for domain in audit["config"]["datasets"]:
            verify(domain["manifests"]["test"], audit["manifest_sha256"][domain["name"]])
            records = report["datasets"][domain["name"]]["methods"]
            if any(record["frames"] != group["frames"][domain["name"]] for record in records.values()):
                raise RuntimeError("测试帧数与固定清单不符")
            for region in ("full", "near", "mid", "far", "gradient_band", "smooth"):
                if len({record["depth"][region]["gt_pixels"] for record in records.values()}) != 1:
                    raise RuntimeError("各方法GT分母不一致")
        counts[name] = {d: next(iter(v["methods"].values()))["frames"]
                        for d, v in report["datasets"].items()}
    equality, selected = {}, {}
    for phase in ("public", "carla_new"):
        best = torch.load(folder / phase / "best.pt", map_location="cpu", mmap=True, weights_only=False)
        final = torch.load(folder / phase / "final.pt", map_location="cpu", mmap=True, weights_only=False)
        equality[phase] = all(torch.equal(value, final["model"][key]) for key, value in best["model"].items())
        events = [json.loads(line) for line in (folder / phase / "events.jsonl").read_text().splitlines()]
        selected[phase] = [event for event in events if event["event"] == "best_updated"][-1]
        if not any(event["event"] == "complete" for event in events):
            raise RuntimeError("训练阶段未完成")
        del best, final
    write_json(ROOT / "runs/active/v5_comparison_20261101/v5_audit.json", {
        "completed": True, "time_utc": datetime.now(timezone.utc).isoformat(),
        "files_hashed": len(cache), "counts": counts, "prediction_calls": summary["prediction_calls"],
        "best_final_tensor_equality": equality, "selected_by_val": selected,
        "weights_sha256": hashes,
        "note": "复核历史完整运行及当前权重/源码，不把本核查视为重新执行101608次预测。",
    })
    print(json.dumps({"completed": True, "files_hashed": len(cache), "counts": counts,
                      "best_final_tensor_equality": equality, "selected_by_val": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
