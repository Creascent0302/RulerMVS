"""冻结共同训练数据和三份独立配置；不启动训练。"""

import copy
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA = Path("/data/liuzonghao_data/dataset/public_expansion_20261102")
OUTPUT = DATA / "overnight_v6_v7_mvs_20261103"
CONFIGS = Path(__file__).parent / "configs"


def own_stage(number, epochs, matching_lr, blocks, decoder_lr, foundation_lr):
    prefixes = ["decoder.", "bridge.", "matching.lateral.", "matching.smooth."]
    if matching_lr:
        prefixes.append("matching.backbone.")
    return dict(number=number, epochs=epochs, trainable_prefixes=prefixes,
                foundation_blocks=blocks, decoder_lr=decoder_lr, handoff_lr=decoder_lr,
                bridge_lr=decoder_lr, fpn_lr=decoder_lr, matching_lr=matching_lr,
                foundation_lr=foundation_lr, bridge_detach_backbone=False,
                correspondence_weight=.02 if number > 1 else 0.)


def main():
    base = json.loads((DATA / "train_v6_expanded.json").read_text())
    base.update(seed=20261103, initial_model_checkpoint="", augmentation=True)
    base["train"].update(batch_size=8, accumulation=2, global_batch_size=32,
                         workers=4, precision="bf16", checkpoint_every=400, log_every=20,
                         validation_samples_per_domain=128, full_validation_every=3,
                         warmup_updates=100, minimum_free_gib=208)
    base["stages"] = [own_stage(1, 1, 0., 0, 1e-4, 0.),
                      own_stage(2, 2, 2e-5, 4, 1e-4, 1e-6),
                      own_stage(3, 6, 1e-5, 8, 5e-5, 1e-6)]
    base["range_augmentation"] = dict(
        enabled=True, nominal_probability=.5, near_multiplier=[.5, 1.], far_multiplier=[1., 2.],
        unit_jitter_probability=.5, unit_scale_bounds=[.5, 2.])
    base["protocol_notes"] = [
        "三模型均从通用预训练初始化，不加载旧V6或Hero任务权重，不蒸馏。",
        "共同七域TRAIN64198组、相同重放倍率、全局batch32与样本键顺序。",
        "RGB共同256x448；自研匹配支路512x896仅由共同RGB插值，无额外图像信息。",
        "共同亮度增强；不主动删源；MVS原生几何翻转、范围扰动和原生损失保留。",
        "V6/V7共同使用TRAIN监督范围扩展和内部单位增强；推理只用无标签固定范围。",
        "完整VAL每三轮及末轮选本次best；子集VAL仅监测，不回滚历史权重。",
        "这是有限共同数据预算重训，不是原论文145k步与原八域训练指标复现。",
        "本轮不训练CARLA、不读取ETH/RMVD TEST，以保留目标域外测。",
    ]
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for method in ("v6", "v7", "mvsanywhere"):
        config = copy.deepcopy(base)
        config.update(method=method, output=str(OUTPUT / method))
        config["launch"] = {"gpus": {"v6": [0, 1], "v7": [2, 3], "mvsanywhere": [4, 5, 6, 7]}[method]}
        if method == "v7":
            config["architecture"] = "rulermvs_v7_refusion"
            config["refusion"] = dict(enabled=True, bands=8, hidden=32, rounds=2)
        if method == "mvsanywhere":
            config["architecture"] = "mvsanywhere_retrained"
            config["train"].update(batch_size=8, accumulation=1)
            config["stages"] = [dict(number=1, epochs=9)]
            config["initialization"].update(
                dinov2=str(DATA / "mvs_initialization/dinov2_vitb14_pretrain.pth"),
                matching_antialiased=str(DATA / "mvs_initialization/resnet18_lpf4_finetune-8cc58f59.pth"))
            config["mvsanywhere"] = dict(schedule="budget_scaled", matching_lr=1e-4,
                                         encoder_lr=5e-6, decoder_lr=5e-5,
                                         geometric_flip=True, refinements=1,
                                         estimate_depth_range=False)
            config["range_augmentation"]["enabled"] = False
        (CONFIGS / (method + ".json")).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"三配置已写入 {CONFIGS}；训练尚未启动")


if __name__ == "__main__":
    main()
