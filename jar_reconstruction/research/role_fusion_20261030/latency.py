"""沿用交替计时协议，只替换本轮checkpoint加载器。"""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.early_geometry_20261030 import latency as engine
from research.role_fusion_20261030.pilot import load_model, path


def run(settings):
    with patch.object(engine, "load_model", load_model):
        engine.run(settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
