"""相同完整VAL计算，只显式注册第二轮严格模型工厂。"""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.v7_domain_20261102 import fullval
from research.v7_domain_20261102.model_round2 import build_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    settings = json.loads(Path(parser.parse_args().config).read_text())
    if settings.get("merge"):
        fullval.merge(settings)
    else:
        with patch.object(fullval, "build_model", build_model):
            fullval.run(settings)
