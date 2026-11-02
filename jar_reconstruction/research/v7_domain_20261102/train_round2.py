"""第二轮独立包装器：复用冻结第一轮训练循环，仅显式替换模型工厂。"""

import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from research.v7_domain_20261102 import model, train
from research.v7_domain_20261102.model_round2 import build_model
from research.v7_domain_20261102.retention_round2 import retention_penalty


def run(settings):
    settings = copy.deepcopy(settings)
    if settings["variant"] not in ("fixed", "coordinate"):
        raise ValueError("第二轮仅比较fixed和coordinate内容先验")
    if not settings.get("domain_augmentation"):
        raise ValueError("第二轮两组共同使用预登记域增强，不夹带数据条件差异")
    sources = [Path(__file__), Path(__file__).with_name("model_round2.py"),
               Path(__file__).with_name("coordinate_prior.py"), Path(model.__file__),
               Path(train.__file__), Path(__file__).with_name("prior.py"),
               Path(__file__).with_name("augmentation.py"),
               Path(__file__).with_name("retention_round2.py")]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    output = Path(settings["output"])
    with patch.object(model, "build_model", build_model), \
            patch.object(train, "retention_penalty", retention_penalty):
        train.run(settings)
    completed = json.loads((output / "completed.json").read_text())
    if completed["prior_gradient_max"] <= 0:
        raise RuntimeError("粗层先验没有获得非零梯度")
    if hashes != {str(path.relative_to(ROOT)): digest(path) for path in sources}:
        raise RuntimeError("第二轮源文件运行中改变，保留产物但不标完整通过")
    for path in sources:
        (output / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    write_json(output / "round2_audit.json", {
        "completed": True, "test_accessed": False, "source_sha256": hashes,
        "variant": settings["variant"], "training_condition": settings["training_condition"],
        "factory_override": "进程内显式替换，原model.py/train.py文件未修改",
        "scope": "同V6起点、同TRAIN/VAL预算；new_only与joint_keep是两种训练条件，不独立分离解冻与保持损失",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
