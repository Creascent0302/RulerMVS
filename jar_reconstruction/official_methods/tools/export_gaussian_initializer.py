"""将共享第一阶段checkpoint导出为官方分布初始化器的权重格式。"""

import argparse
from pathlib import Path
import torch
from mmengine.logging import MMLogger
from official_methods.methods.gaussianformer2 import Adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    if (
        saved.get("method") != "gaussianformer2"
        or config["model"].get("training_stage") != "initializer"
    ):
        raise ValueError("必须输入GF2 initializer阶段的共享checkpoint")
    config = dict(config, _loading_shared_checkpoint=True)
    MMLogger.get_instance("gf_initializer_export", log_level="ERROR")
    adapter = Adapter(config)
    adapter.load_state_dict(saved["model"], strict=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter.export_initializer(output)
    print(f"已导出官方初始化器键名：{output}")


if __name__ == "__main__":
    main()
