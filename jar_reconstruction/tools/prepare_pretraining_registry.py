"""固定公共预训练划分并检查所有视图的文件交叉；不修改原始数据或模型。

参数集中在 JSON。已有六套划分原样引用；DynamicReplica 的原 valid 清单
另存为明确命名的本项目开发划分，不覆盖或改称官方 train/test。
"""

import argparse
import hashlib
import json
import shutil
from itertools import combinations
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_or_identical(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"已有清单与本次不同，不覆盖：{path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def prepare_development_split(entry):
    specification = entry["development_split"]
    source = Path(specification["source_manifest"])
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    assignment = {}
    for split, groups in specification["sequences"].items():
        for group in groups:
            if group in assignment:
                raise ValueError(f"序列重复分配：{group}")
            assignment[group] = split
    actual = {record["source_sequence"] for record in records}
    if actual != set(assignment):
        raise ValueError("开发划分必须完整覆盖所选原始序列，不能丢弃或添加序列")
    for split, destination in entry["manifests"].items():
        selected = []
        for record in records:
            if assignment[record["source_sequence"]] == split:
                selected.append({
                    **record,
                    "project_split": split,
                    "split_origin": "project_development_from_official_valid",
                })
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
        write_new_or_identical(Path(destination), text)
    return {"source_manifest": str(source), "sha256": sha256(source)}


def audit_dataset(entry):
    root = Path(entry["root"]).resolve()
    report, resources = {}, {}
    for split, name in entry["manifests"].items():
        manifest = Path(name)
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        identifiers, paths, groups = set(), set(), set()
        for row in rows:
            identifier = (row["sequence"], str(row["frame_id"]))
            if identifier in identifiers:
                raise ValueError(f"样本标识重复：{manifest} {identifier}")
            identifiers.add(identifier)
            groups.add(row["sequence"])
            cameras = row["cameras"]
            if len(cameras) != entry["num_views"]:
                raise ValueError(f"视图数与配置不同：{identifier}")
            if row.get("geometry_units", "meters") != entry["geometry_units"]:
                raise ValueError(f"单位与配置不同：{identifier}")
            if row.get("reference_slot", entry["reference_index"]) != entry["reference_index"]:
                raise ValueError(f"参考槽位与配置不同：{identifier}")
            for camera in cameras:
                targets = camera.get("targets", {})
                for name in [camera["rgb"], *[targets[k] for k in
                             ("depth", "depth_valid", "semantic") if k in targets]]:
                    path = (root / name).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError(f"样本引用了数据根目录外的文件：{path}")
                    paths.add(path)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{entry['name']}/{split} 缺文件：{missing[:5]}")
        report[split] = {
            "manifest": str(manifest), "sha256": sha256(manifest),
            "samples": len(rows), "sequence_count": len(groups),
            "sequences": sorted(groups), "unique_referenced_files": len(paths),
        }
        resources[split] = paths
    overlaps = {}
    for first, second in combinations(resources, 2):
        shared = resources[first] & resources[second]
        overlaps[f"{first}:{second}"] = len(shared)
        if shared:
            raise ValueError(f"{entry['name']} 的 {first}/{second} 共用图像或标签：{list(shared)[:5]}")
    return {"splits": report, "cross_split_file_overlaps": overlaps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["storage_root"])
    free = shutil.disk_usage(root.parent).free
    if free < (config["minimum_free_gib"] + 1) * 2 ** 30:
        raise RuntimeError("可用空间不足以保留指定下限和1GiB写入余量")
    report = {
        "config": str(config_path), "config_sha256": sha256(config_path),
        "scope": "清单与实际文件存在性检查；不证明内容去重、几何正确或训练精度",
        "free_bytes_before": free, "datasets": {},
    }
    for entry in config["datasets"]:
        provenance = prepare_development_split(entry) if "development_split" in entry else None
        checked = audit_dataset(entry)
        checked["source_provenance"] = provenance
        report["datasets"][entry["name"]] = checked
        print(entry["name"], {s: r["samples"] for s, r in checked["splits"].items()}, flush=True)
    report["totals"] = {
        split: sum(item["splits"][split]["samples"] for item in report["datasets"].values())
        for split in ("train", "val", "test")
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "split_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Audit:", root / "split_audit.json", "Totals:", report["totals"])


if __name__ == "__main__":
    main()
