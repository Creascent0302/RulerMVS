"""独立标尺消融训练入口；复用已验V6训练循环，不修改生产代码。"""

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from research.paper_ablation_20261102.geometry import DESCRIPTIONS, build_geometry_model
from research.paper_ablation_20261102.handoff_evaluate import evaluate
from research.rulermvs_v6_20261101 import pilot


SOURCES = [
    Path(__file__), Path(__file__).with_name("geometry.py"),
    Path(__file__).with_name("handoff_evaluate.py"),
    ROOT / "research/rulermvs_v6_20261101/pilot.py",
    ROOT / "research/rulermvs_v6_20261101/model.py",
    ROOT / "research/rulermvs_v6_20261101/objective.py",
    ROOT / "jar/pretraining/model.py", ROOT / "jar/models/ruler_proposals.py",
]


@contextmanager
def geometry_builder(ablation):
    """进程内只替换模型工厂；训练、恢复与损失仍走相同实现。"""
    def factory(saved, variant):
        if variant != "parent":
            raise ValueError("标尺消融统一使用V6 parent，不混入交接消融")
        return build_geometry_model(saved, ablation)

    with patch.object(pilot, "build_from_checkpoint", factory), patch.object(pilot, "evaluate", evaluate):
        yield


def run(settings):
    settings = copy.deepcopy(settings)
    ablation = settings["geometry_ablation"]
    if ablation not in DESCRIPTIONS:
        raise ValueError(f"未知消融: {ablation}")
    if settings.get("variant") != "parent":
        raise ValueError("四组必须统一V6 parent及grouped_loss")
    if not settings.get("grouped_loss", False):
        raise ValueError("不能在几何消融里同时更改训练目标")
    output = Path(settings["output"])
    if not output.resolve().is_relative_to(Path("/data")):
        raise ValueError("训练产物必须经项目软链接存入/data")
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in SOURCES}
    with geometry_builder(ablation):
        pilot.run(settings)
    # 先由原训练器完成真实落盘恢复/隔离核验，再保存本扩展的来源记录。
    if source_hashes != {str(path.relative_to(ROOT)): digest(path) for path in SOURCES}:
        raise RuntimeError("运行期间依赖源码发生变化，须保留结果并审计，不能标为同版本完成")
    write_json(output / "geometry_audit.json", {
        "completed": True,
        "geometry_ablation": ablation,
        "description": DESCRIPTIONS[ablation],
        "checkpoint_sha256": digest(settings["checkpoint"]),
        "source_sha256": source_hashes,
        "selection": "fixed_last_step",
        "test_accessed": False,
        "causal_scope": "共同V5任务权重起点的冻结编码器适配，不是从零训练对照",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
