"""新旧架构的隔离加载适配；完整复用原共同指标和同卡计时实现。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining import comparison
from jar.pretraining.model import PretrainingModel


def model_factory(config, initialize=True):
    if config.get("architecture") == "ruler_flow_v2":
        from research.ruler_flow_v2_20261031.model import EvidenceFlowModel

        return EvidenceFlowModel(config, initialize)
    return PretrainingModel(config, initialize)


# 仅此独立评测进程改变工厂。spawn子进程重导入此入口，保留同一工厂。
comparison.PretrainingModel = model_factory


if __name__ == "__main__":
    if "--speed" in sys.argv:
        sys.argv.remove("--speed")
        from tools.benchmark_public_speed import main
    else:
        from research.near_mid_evaluation_20261028.evaluate import main
    main()
