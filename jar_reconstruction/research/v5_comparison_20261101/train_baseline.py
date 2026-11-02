"""单方法独立队列：原生训练→VAL-best/final固定TEST；参数只读配置。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    method = config["method"]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training/last.pt").exists():
        raise FileExistsError("队列不覆盖旧训练；续训请使用该方法train.sh --checkpoint last.pt")
    package = ROOT / f"official_methods/methods/{method}"
    files = list(package.rglob("*.py")) + [ROOT / "official_methods/cli.py",
            ROOT / "official_methods/data.py", ROOT / "official_methods/training_validation.py"]
    sources = {str(path.relative_to(ROOT)): digest(path) for path in files}
    audit = {
        "time_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "config_sha256": digest(args.config), "source_sha256": sources,
        "manifest_sha256": {split: digest(config["data"][f"{split}_manifest"])
                            for split in ("train", "val", "test")},
        "initial_weights_sha256": digest(config["model"]["weights"]) if config["model"]["weights"] else None,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "world_size": int(os.environ.get("GPUS", "1")),
        "protocol": config["experiment"],
    }
    write_json(output / "audit.json", audit)
    command = ["bash", str(package / "train.sh"), "--config", str(Path(args.config).resolve()),
               "--output", str(output / "training")]
    print("TRAIN", command, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if any(digest(ROOT / path) != value for path, value in sources.items()):
        raise RuntimeError("训练期间原生源码改变，拒绝自动测试")
    comparison = json.loads((Path(__file__).parent / "configs/carla_complete_equal_pixels.json").read_text())
    comparison.update(output=str(output / "evaluation"), gpus=list(range(audit["world_size"])))
    comparison["methods"] = {
        f"{method}_{name}": {"kind": method, "weights": str(output / f"training/{name}.pt"),
                             "project_checkpoint": True,
                             "model": {"variant": config["model"].get("variant", method)}}
        for name in ("best", "final")
    }
    comparison["extension_source_sha256"].update(sources)
    comparison["protocol_notes"].append(config["experiment"]["comparability"])
    path = output / "evaluation_config.json"
    write_json(path, comparison)
    command = [sys.executable, str(Path(__file__).with_name("evaluate.py")), "--config", str(path)]
    print("TEST", command, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    write_json(output / "completed.json", {"completed": True, "time_utc": datetime.now(timezone.utc).isoformat(),
                                           "test_did_not_select_checkpoint": True})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
