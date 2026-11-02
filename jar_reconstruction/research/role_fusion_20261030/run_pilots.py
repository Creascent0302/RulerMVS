"""配置列出的独立GPU研究队列；失败停止该GPU后续任务，保留全部日志。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(settings):
    output = ROOT / settings["output"]
    output.mkdir(parents=True, exist_ok=True)
    code_file = output / f"{settings['name']}.exit_code.txt"
    if code_file.exists():
        raise FileExistsError(f"队列已执行：{code_file}")
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    python = ROOT / "official_methods/.venv/bin/python"

    def worker(configs):
        for name in configs:
            filename = ROOT / name
            label = filename.stem
            log_file = output / f"{label}.log"
            with log_file.open("x") as log:
                result = subprocess.run(
                    [str(python), "research/role_fusion_20261030/pilot.py", "--config", str(filename)],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                )
            (output / f"{label}.exit_code.txt").write_text(str(result.returncode) + "\n")
            if result.returncode:
                return result.returncode
        return 0

    with ThreadPoolExecutor(max_workers=len(settings["workers"])) as pool:
        results = list(pool.map(worker, settings["workers"]))
    code = int(any(results))
    code_file.write_text(str(code) + "\n")
    if code:
        raise SystemExit(code)
    print(f"队列 {settings['name']} 完成", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads((ROOT / parser.parse_args().config).read_text()))
