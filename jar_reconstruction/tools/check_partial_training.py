"""生成有真实投影关系的部分标签训练夹具，供共享单卡/DDP入口短测。"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jar.synthetic import SyntheticRigDataset
from jar.config import load_config
from jar.carla_capture import WIDE_IDS, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).absolute()
    out.mkdir(parents=True, exist_ok=True)
    ds = SyntheticRigDataset(length=2, image_size=(32, 48), num_cameras=4, seed=43)
    rows = []
    for i, views in enumerate(([0, 2], [0, 1, 2])):
        sample = ds[i]
        a = sample["inputs"]
        t = sample["targets"]
        row = {"sequence": "analytic_partial", "frame_id": i, "timestamp": float(i), "cameras": []}
        for view in views:
            rgb = f"{i}_{view}.png"
            Image.fromarray(
                (a["rgb"][view].permute(1, 2, 0).numpy() * 255)
                .round()
                .clip(0, 255)
                .astype(np.uint8)
            ).save(out / rgb)
            camera = {
                "camera_id": WIDE_IDS[view],
                "frame_id": i,
                "timestamp": float(i),
                "rgb": rgb,
                "K": a["K"][view].tolist(),
                "T_cam_from_rig": a["T_cam_from_rig"][view].tolist(),
            }
            if i == 0 and view == 2:
                depth = "depth.npy"
                np.save(out / depth, t["depth"][view].numpy())
                np.save(out / "valid.npy", t["depth_valid"][view].numpy())
                camera["targets"] = {
                    "frame_id": i,
                    "depth": depth,
                    "depth_valid": "valid.npy",
                    "depth_type": "optical_z",
                }
            row["cameras"].append(camera)
        rows.append(row)
    write_jsonl(out / "train.jsonl", rows)
    cfg = load_config("configs/mf_et_jar_wide4_tiny.json")
    cfg["camera"] = {
        "mode": "wide4",
        "wide_ids": list(WIDE_IDS),
        "tele_ids": ["tele_left", "tele_right"],
        "allow_missing": True,
    }
    cfg["data"] = {
        "root": str(out),
        "train_manifest": "train.jsonl",
        "val_manifest": "",
        "test_manifest": "train.jsonl",
    }
    cfg["loss"]["enforce_reference_depth_range"] = True
    cfg["eval"]["observed_only"] = True
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(out / "config.json")


if __name__ == "__main__":
    main()
