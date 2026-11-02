"""统一启动入口；模型、相机和训练参数只从 JSON 配置读取。"""

import argparse
from pathlib import Path


def build_parser(mode):
    parser = argparse.ArgumentParser(
        description=f"JAR {mode}：参数见 JSON；合成测试须显式 --synthetic"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/mf_et_jar_wide4.json"),
        help="模型、数据与运行参数的 JSON 配置",
    )
    parser.add_argument("--synthetic", action="store_true", help="显式启用解析合成数据")
    parser.add_argument(
        "--device", default="auto", help="auto/cpu/cuda/cuda:0；torchrun 按 LOCAL_RANK 绑定 GPU"
    )
    parser.add_argument("--output", default=f"runs/{mode}")
    if mode == "train":
        weights = parser.add_mutually_exclusive_group()
        weights.add_argument("--init-checkpoint", default="", help="仅初始化模型权重")
        weights.add_argument("--resume", default="", help="恢复模型、优化器与完整训练状态")
    else:
        parser.add_argument("--checkpoint", default="", help="模型权重路径")
        parser.add_argument(
            "--split", choices=["val", "test"], default="test" if mode == "infer" else "val"
        )
        parser.add_argument("--limit", type=int, help="本次最多处理的帧数，便于短测")
        if mode == "eval":
            parser.add_argument(
                "--save-predictions", action="store_true", help="评估时额外导出 NPZ"
            )
        else:
            parser.set_defaults(save_predictions=True)
    return parser


def main(mode):
    args = build_parser(mode).parse_args()
    if mode == "train":
        return train(args)
    from .inference import run_prediction

    return run_prediction(args, evaluate=mode == "eval")


def train(args):
    from .training import train as implementation

    return implementation(args)


def evaluate(args):
    from .inference import run_prediction

    return run_prediction(args, evaluate=True)
