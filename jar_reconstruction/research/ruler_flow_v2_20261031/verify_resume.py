"""两卡断点恢复：同时记录逐位差异和真实RGB预测差，不混称精确恢复。"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move
from research.ruler_flow_v2_20261031.pilot import load_model


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = ROOT / "runs/active/ruler_flow_v2_20261031"
    device = torch.device("cuda:0")
    a, first = load_model(folder / "resume_check/last.pt", device)
    b, second = load_model(folder / "resume_reference/last.pt", device)
    differences = {key: float((value.float() - second["model"][key].float()).abs().max())
                   for key, value in first["model"].items()
                   if not torch.equal(value, second["model"][key])}
    close = all(torch.allclose(value, second["model"][key], atol=1e-7, rtol=1e-6)
                for key, value in first["model"].items())
    old_exact = all(key.startswith("decoder.flow_field.") for key in differences)
    dataset = MixedDataset(first["config"], "val")
    predictions = []
    for domain_index in (0, 1):
        domain = dataset.domains[domain_index]
        batch = move(collate_mixed([dataset[(domain_index, 0, -1)]]), device)
        with torch.inference_mode():
            x = a(batch["inputs"], domain)["depth"] * domain["depth_scale"]
            y = b(batch["inputs"], domain)["depth"] * domain["depth_scale"]
        predictions.append({"dataset": domain["name"], "exact": bool(torch.equal(x, y)),
                            "max_abs": float((x - y).abs().max()),
                            "mean_abs": float((x - y).abs().mean()),
                            "units": domain["geometry_units"]})
    write_json(folder / "resume_verification.json", {
        "resumed_step": first["state"]["global_step"],
        "continuous_step": second["state"]["global_step"],
        "all_model_tensors_exact": not differences,
        "all_model_tensors_close": close, "atol": 1e-7, "rtol": 1e-6,
        "old_model_tensors_exact": old_exact, "differences": differences,
        "prediction_comparison": predictions,
        "interpretation": "loss和采样日志相同，新增7张量有FP32微差；未定位唯一浮点原因，不称逐位恢复。",
    })
    if not close or not old_exact:
        raise RuntimeError("恢复参数偏差超过登记容差")
    print(json.dumps(predictions, ensure_ascii=False))


if __name__ == "__main__":
    main()
