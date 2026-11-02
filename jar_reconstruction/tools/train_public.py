"""公共多域分阶段训练。超参数全部在配置文件中。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jar.pretraining.runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.resume or None)


if __name__ == "__main__":
    main()
