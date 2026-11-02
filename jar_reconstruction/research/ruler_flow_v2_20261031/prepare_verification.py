"""为V4完整评测补充相同RGB信息量对照；固定权重，不训练、不选模。"""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json


def main():
    base = ROOT / "runs/active/ruler_flow_v2_20261031"
    folder = Path(__file__).parent
    destination = folder / "configs/verification"
    files = {}
    for group in ("public", "carla", "carla_other_refs"):
        config = json.loads((base / "evaluation/configs" / f"{group}.json").read_text())
        own = copy.deepcopy(config["methods"]["flow_best"])
        old = copy.deepcopy(config["methods"]["original_v2"])
        first = copy.deepcopy(config["methods"]["mvsanywhere_refined"])
        refined = copy.deepcopy(first)
        own["input_mode"] = old["input_mode"] = "equal_pixels"
        first["refinements"] = 0
        config["methods"] = {"v4_equal_pixels": own, "v2_equal_pixels": old,
                             "mvsanywhere_firstpass": first, "mvsanywhere_refined": refined}
        config["output"] = str(base / "verification/equal_pixels" / group)
        config["protocol_notes"] = [
            "固定完整VAL选择的V4 best、旧V2 best和官方Hero权重；不以TEST选择配置或训练。",
            "共同RGB信息为256x448；自研matching仅由该主图双线性放大到既有512x896分支，不读取额外高分辨率RGB。",
            "自研网络分支尺寸保留，故相同输入信息不等于相同内部算力、参数量或训练预算。",
            "原baseline第一遍与一轮范围细化两种官方推理分别报告，起始范围统一来自TRAIN。",
            "同帧、标定、真实N、GT有效区域、单位与距离段定义保持原完整TEST协议；不补假视角、不增加历史帧。",
            "匹配分支上采样引入的分布变化属于固定权重输入预算干预，不是重新训练的低分辨率架构。",
            "native完整TEST结果另外保留，不按数据域挑选native/equal_pixels组合。",
        ]
        config["extension_source_sha256"] = {
            str(p.relative_to(ROOT)): digest(p) for p in folder.glob("*.py")}
        filename = destination / f"equal_pixels_{group}.json"
        write_json(filename, config)
        files[group] = str(filename)
    print(json.dumps(files, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
