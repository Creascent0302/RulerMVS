"""对真实DDP冒烟checkpoint检查七域无标签推理和共同VAL统计。"""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import torch

from jar.pretraining.comparison import write_json
from jar.runtime import setup_runtime, cleanup_runtime
from research.overnight_v6_v7_mvs_20261103.common_data import SharedDataset
from research.overnight_v6_v7_mvs_20261103.train import build_model, validate
from tools.smoke_outdoor_datasets import offline_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    offline_only()
    runtime = setup_runtime("cuda")
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = build_model(config, initialize=False)
        saved = torch.load(Path(config["output"]) / "last.pt", map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(saved["model"], strict=True)
        model.to(runtime.device).eval()
        result = validate(model, SharedDataset(config, "val"), config, runtime, limit=2)
        if runtime.is_main:
            write_json(Path(config["output"]) / "validation_smoke.json", dict(completed=True, **result))
            print(json.dumps({"completed": True, "method": config["method"],
                              "frames": sum(d["frames"] for d in result["domains"])}), flush=True)
    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
