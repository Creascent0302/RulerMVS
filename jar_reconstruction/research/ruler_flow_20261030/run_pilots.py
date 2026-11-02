"""每个GPU一条配置队列；失败停该队列，保留退出码和完整日志。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def run(settings):
    output = ROOT / settings["output"]
    output.mkdir(parents=True, exist_ok=True)
    code_file = output / f"{settings['name']}.exit_code.txt"
    if code_file.exists():
        raise FileExistsError(f"队列已执行：{code_file}")
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")

    def worker(configs):
        for filename in configs:
            label = Path(filename).stem
            with (output / f"{label}.log").open("x") as log:
                result = subprocess.run([
                    str(ROOT / "official_methods/.venv/bin/python"),
                    "research/ruler_flow_20261030/pilot.py", "--config", filename,
                ], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads((ROOT / parser.parse_args().config).read_text()))
