"""四个固定TRAIN样本的真实再投影位移审计；不执行模型、不读取TEST。"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from research.v7_domain_20261102.synthetic_pair_bilinear import synthesize_pair


def run():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(202611027)
    records = []
    for phase in ("public", "carla"):
        config = json.loads((ROOT / f"research/rulermvs_v6_20261101/configs/data_{phase}.json").read_text())
        config["datasets"] = config["datasets"][:1]
        config.pop("validation_datasets", None)
        config.update(seed=20261102, augmentation=False)
        config["datasets"][0]["manifests"] = {"train": config["datasets"][0]["manifests"]["train"]}
        data = MixedDataset(config, "train")
        batch = collate_mixed([data[(0, index, -1)] for index in (0, 1)])
        inputs = equal_pixel_inputs(batch["inputs"])
        augmented, supervision, record = synthesize_pair(inputs, batch["targets"], data.domains[0],
                                                        generator=generator)
        width = inputs["rgb"].shape[-1]
        for sample in range(2):
            mapping = supervision["mapping"]
            valid = mapping["weight"][sample] > 0
            indices = mapping["reference_index"][sample, valid]
            # 同一参考点可能投放到两个相邻像素；按参考点去重避免加倍计数。
            indices = torch.unique(indices)
            depth = batch["targets"]["depth"][sample, 0].flatten()[indices]
            translation = (augmented["T_cam_from_rig"][sample, 1, 0, 3]
                           - augmented["T_cam_from_rig"][sample, 0, 0, 3]).abs()
            shift = inputs["K"][sample, 0, 0, 0] * translation / depth
            exact_shift = (mapping["projected_u"][sample, valid]
                           - (mapping["reference_index"][sample, valid] % width)).abs()
            expected = inputs["K"][sample, 0, 0, 0] * translation / (
                batch["targets"]["depth"][sample, 0].flatten()[mapping["reference_index"][sample, valid]])
            error = float((exact_shift - expected).abs().max())
            assert error < 1e-4
            bands = {}
            for name, lower, upper in (("0-.5px", 0., .5), (".5-2px", .5, 2.),
                                       ("2-8px", 2., 8.), (">=8px", 8., float("inf"))):
                bands[name] = float(((shift >= lower) & (shift < upper)).float().mean())
            records.append({
                "dataset": data.domains[0]["name"], "split": "train", "index": sample,
                "valid_reference_points_reaching_valid_source": len(indices),
                "baseline_data_units": float(translation),
                "source_coverage": float(record["source_coverage"][sample]),
                "shift_quantiles_pixels_p10_p50_p90": torch.quantile(shift, torch.tensor([.1, .5, .9])).tolist(),
                "fractions": bands, "exact_horizontal_projection_error_pixels": error,
            })
    output = ROOT / "runs/active/v7_domain_20261102/synthetic_pair_bilinear_shift_audit.json"
    write_json(output, {
        "completed": True, "test_accessed": False, "model_run": False, "device": "cpu",
        "samples": records, "source_sha256": digest(Path(__file__)),
        "limits": "仅四个固定TRAIN样本的有效重投影点；不代表全部训练曝光或实际网络收益。",
    })
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
