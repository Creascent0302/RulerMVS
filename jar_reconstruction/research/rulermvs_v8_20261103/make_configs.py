"""生成一次事前登记的四路配对短试；正式配置在选择结论后另存。"""

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STORE = ROOT / "runs/active/rulermvs_v8_20261103"


def main():
    original = json.loads((ROOT / "research/overnight_v6_v7_mvs_20261103/configs/v6.json").read_text())
    config = copy.deepcopy(original)
    config["initial_model_checkpoint"] = str(ROOT / "runs/active/overnight_v6_v7_mvs_20261103/v6/best.pt")
    config["protocol_notes"] = [
        "V6公共VAL-best暖起点；非从零或同原论文预算重训，不加载Hero。",
        "四路均使用同TRAIN顺序、种子、globalB32、冻结DA和相同学习率。",
        "改变因素分别为训练增强、单位一致几何、源数先验；ETH/Tanks不参与选择。",
        "短试只选研究候选，不证明完整训练泛化收益。"
    ]
    stage = dict(original["stages"][2])
    stage.update(number=1, epochs=1, foundation_blocks=0, foundation_lr=0.,
                 decoder_lr=2e-5, handoff_lr=2e-5, bridge_lr=1e-5,
                 fpn_lr=1e-5, matching_lr=2e-6, correspondence_weight=.02)
    config["stages"] = [stage]
    config["train"].update(batch_size=8, accumulation=2, global_batch_size=32,
                           smoke_updates_per_epoch=256, workers=2,
                           initial_validation_limit=64, validation_limit=64,
                           validation_samples_per_domain=64, full_validation_every=99,
                           warmup_updates=20, checkpoint_every=256, log_every=20)
    config["photometric_augmentation"] = dict(enabled=True, clean_probability=.4,
                                                source_dropout_probability=.1)
    config["range_augmentation"] = dict(enabled=True, nominal_probability=.4,
                                        broad_probability=.2, max_log_padding=1.5,
                                        gt_occupancy_bounds=[.25, .85], broad_multipliers=[.25, 4.],
                                        unit_jitter_probability=.5, unit_scale_bounds=[.5, 2.])
    variants = {
        "v6_control": ("v6", "legacy", {}),
        "v6_augmentation": ("v6", "generalization", {}),
        "v8_units": ("v8", "generalization", dict(information_mode="log_depth", information_scale=10.,
                                                   relative_validity=True, physical_thresholds=True)),
        "v8_units_count": ("v8", "generalization", dict(information_mode="log_depth", information_scale=10.,
                                                         relative_validity=True, physical_thresholds=True,
                                                         source_count_strength=1.)),
    }
    folder = HERE / "configs"
    folder.mkdir(exist_ok=True)
    for index, (name, (method, recipe, options)) in enumerate(variants.items()):
        current = copy.deepcopy(config)
        current.update(method=method, architecture="rulermvs_" + method,
                       training_recipe=recipe, v8=options,
                       output=str(STORE / "pilots" / name))
        current["launch"] = dict(gpus=[2 * index, 2 * index + 1])
        if recipe == "legacy":
            current["range_augmentation"] = original["range_augmentation"]
        if method == "v8":
            current["loss"]["unit_invariant_relative_floor"] = True
        (folder / (name + ".json")).write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
