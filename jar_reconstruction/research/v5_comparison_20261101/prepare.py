"""固定本轮比较和独立CARLA训练配置；不根据TEST选择配置。"""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json

HERE = Path(__file__).parent
OUTPUT = ROOT / "runs/active/v5_comparison_20261101"
SUITE = ROOT / "runs/active/baseline_audit_20261031/suite"
V5 = ROOT / "runs/active/rulermvs_v5_20261031/evaluation/configs"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    patch = ROOT / "official_methods/methods/patchmatchnet/upstream/checkpoints/model_000007.ckpt"
    extension = {str(path.relative_to(ROOT)): digest(path)
                 for path in [HERE / "evaluate.py"]
                 + list((ROOT / "official_methods/methods/patchmatchnet").rglob("*.py"))}
    config = read(V5 / "public_equal_pixels.json")
    config.update(output=str(OUTPUT / "patchmatchnet_public"), gpus=[4, 5, 6, 7], shard_size=128)
    config["methods"] = {"patchmatchnet_dtu": {"kind": "patchmatchnet", "weights": str(patch)}}
    config["extension_source_sha256"].update(extension)
    write_json(HERE / "configs/patchmatchnet_public.json", config)

    config = read(HERE / "configs/speed_data_equal_pixels.json")
    config["methods"].pop("v5_final")
    config["methods"]["patchmatchnet_dtu"] = {"kind": "patchmatchnet", "weights": str(patch)}
    config["methods"]["fast_mvsnet_dtu"] = {
        "kind": "fast_mvsnet",
        "weights": str(ROOT / "official_methods/methods/fast_mvsnet/upstream/outputs/pretrained.pth"),
    }
    config.update(output=str(OUTPUT / "carla_complete_equal_pixels"), gpus=[0, 1, 2, 3], shard_size=48)
    config["extension_source_sha256"].update(extension)
    write_json(HERE / "configs/carla_complete_equal_pixels.json", config)

    for method in ("patchmatchnet", "diffmvs"):
        config = read(SUITE / f"configs/{method}/CARLA_new_upper.json")
        config["seed"] = 20261101
        config["training"].update(
            epochs=32, max_steps=384, batch_size=2, gradient_accumulation_steps=4,
            num_workers=2, checkpoint_every=32, validation_every_epochs=1,
            minimum_free_gib=200,
        )
        # 四GPU×每卡2×累积4=32；403帧每轮末尾另有短批次，日志记录实际曝光。
        config["evaluation"]["reference_indices"] = [2]
        config["experiment"] = {
            "purpose": "独立CARLA适配先导试验，非与V5等预训练预算的架构对照",
            "selection": "仅完整VAL选best；final独立保留；TEST只在训练结束后使用",
            "training_reference": "上相机；三个真实源；完整403帧，标签不进入推理",
            "initialization": "官方DTU任务权重" if method == "patchmatchnet" else "随机初始化，尚无官方任务权重",
            "unit_note": "CARLA使用米；保留原生损失。PatchmatchNet原DTU使用毫米，SmoothL1拐点的物理尺度随单位变化，不能称原论文训练协议。",
            "comparability": "与V5训练域、初始化、参考轮换和训练曝光不同，结果只作为适配基线；不据此宣称架构普遍胜出。",
        }
        if method == "patchmatchnet":
            config["training"].update(lr=0.0001, lr_milestones=[20, 28], scheduler_interval="epoch")
        else:
            config["model"].update(weights="", allow_random_init=True)
        path = ROOT / f"official_methods/methods/{method}/configs/train_carla_20261101.json"
        write_json(path, config)
        smoke = copy.deepcopy(config)
        smoke["training"].update(max_steps=2, num_workers=0, validation_every_epochs=0,
                                 gradient_accumulation_steps=1, checkpoint_every=1)
        write_json(HERE / f"configs/{method}_smoke.json", smoke)


if __name__ == "__main__":
    main()
