"""核验补充等像素 TEST 的完整性、GT覆盖和重复 baseline 的数值一致性。"""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    base = ROOT / "runs/active/ruler_flow_v2_20261031"
    result = {}
    for group in ("public", "carla", "carla_other_refs"):
        directory = base / "verification/equal_pixels" / group / "full_test"
        audit = read(directory / "audit.json")
        summary = read(directory / "summary.json")
        protocol = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
        assert summary["completed"] and summary["split"] == "test"
        assert summary["protocol"] == protocol
        methods = audit["config"]["methods"]
        for name, settings in methods.items():
            assert digest(settings["weights"]) == audit["weights_sha256"][name]
        old_directory = base / "evaluation" / group / "full_test"
        old_audit = read(old_directory / "audit.json")
        assert audit["manifest_sha256"] == old_audit["manifest_sha256"]
        assert audit["weights_sha256"]["v4_equal_pixels"] == old_audit["weights_sha256"]["flow_best"]
        assert audit["weights_sha256"]["v2_equal_pixels"] == old_audit["weights_sha256"]["original_v2"]
        for name in ("mvsanywhere_firstpass", "mvsanywhere_refined"):
            assert audit["weights_sha256"][name] == old_audit["weights_sha256"]["mvsanywhere_refined"]
        domains = {}
        for domain in audit["config"]["datasets"]:
            name = domain["name"]
            expected = len(Path(domain["manifests"]["test"]).read_text().splitlines())
            sequences = {split: {json.loads(line)["sequence"] for line in Path(path).read_text().splitlines()}
                         for split, path in domain["manifests"].items()}
            sequence_overlap = {f"{a}_{b}": len(sequences[a] & sequences[b])
                                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
            assert not any(sequence_overlap.values())
            indices = []
            fingerprints = 0
            max_baseline_absolute_error = 0.0
            for filename in sorted((directory / name).glob("part_*.json")):
                part = read(filename)
                old = read(old_directory / name / filename.name)
                assert part["completed"] and part["protocol"] == protocol
                assert len(part["frames"]) == len(old["frames"])
                for frame, old_frame in zip(part["frames"], old["frames"]):
                    indices.append(frame["index"])
                    assert frame["index"] == old_frame["index"]
                    assert frame["valid_views"] == old_frame["valid_views"]
                    assert set(frame["methods"]) == set(methods)
                    if "input_sha256" in frame:
                        assert frame["input_sha256"] == old_frame["input_sha256"]
                        fingerprints += 1
                    for region in ("full", "near", "mid", "far", "gradient_band", "smooth"):
                        values = [m["depth"][region] for m in frame["methods"].values()]
                        assert len({v["gt_pixels"] for v in values}) == 1
                        assert all(v["predicted_pixels"] == v["gt_pixels"] for v in values)
                        previous = old_frame["methods"]["mvsanywhere_refined"]["depth"][region]
                        repeated = frame["methods"]["mvsanywhere_refined"]["depth"][region]
                        assert previous["gt_pixels"] == repeated["gt_pixels"]
                        # 所有统计均累加为 double；仅容许GPU算法的浮点级微差。
                        for key in previous:
                            delta = abs(previous[key] - repeated[key])
                            assert delta <= 2e-5 * max(1.0, abs(previous[key])), (name, frame["index"], key, delta)
                        count = max(1, previous["predicted_pixels"])
                        max_baseline_absolute_error = max(
                            max_baseline_absolute_error,
                            abs(previous["abs_sum"] - repeated["abs_sum"]) / count,
                        )
            assert sorted(indices) == list(range(expected))
            assert fingerprints == min(3, expected)
            assert all(m["frames"] == expected for m in summary["datasets"][name]["methods"].values())
            domains[name] = {"frames": expected, "input_fingerprints_equal": fingerprints,
                             "sequence_counts": {split: len(values) for split, values in sequences.items()},
                             "sequence_overlap": sequence_overlap,
                             "all_gt_covered": True, "all_gt_denominators_equal": True,
                             "maximum_repeated_baseline_frame_MAE_difference": max_baseline_absolute_error,
                             "MAE": {region: {method: value["depth"][region]["MAE"]
                                              for method, value in summary["datasets"][name]["methods"].items()}
                                     for region in ("full", "near")}}
        wins = {other: {region: sum(d["MAE"][region]["v4_equal_pixels"] < d["MAE"][region][other]
                                   for d in domains.values())
                       for region in ("full", "near")}
                for other in ("v2_equal_pixels", "mvsanywhere_firstpass", "mvsanywhere_refined")}
        result[group] = {"completed": True, "frames": sum(d["frames"] for d in domains.values()),
                         "method_predictions": 4 * sum(d["frames"] for d in domains.values()),
                         "MAE_wins": wins, "domains": domains}
    report = {"completed": True, "groups": result,
              "note": "相同256x448原始RGB信息；自研matching仍在512x896上计算，不是低分辨率重新训练。"}
    output = base / "verification/equal_pixels/completed_audit.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
