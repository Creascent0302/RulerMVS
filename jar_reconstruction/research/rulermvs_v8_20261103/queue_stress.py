"""短训正常结束后，各自复用空出的GPU运行事前固定VAL压力测试。"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "runs/active/rulermvs_v8_20261103"


def worker(name, gpu):
    directory = STORE / "pilots" / name
    while True:
        try:
            status = json.loads((directory / "job_status.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(5)
            continue
        if status["state"] == "failed":
            raise RuntimeError(f"{name}训练失败，不伪造压力结果")
        if status["state"] == "completed":
            break
        time.sleep(5)
    jobs = [(name, directory / "final.pt")]
    if name == "v6_control":
        jobs.insert(0, ("v6_original", ROOT / "runs/active/overnight_v6_v7_mvs_20261103/v6/best.pt"))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for label, checkpoint in jobs:
        with (STORE / "logs" / ("stress_" + label + ".log")).open("w") as stream:
            subprocess.run([sys.executable, "research/rulermvs_v8_20261103/stress.py",
                            "--checkpoint", str(checkpoint), "--output",
                            str(STORE / "diagnostics" / ("stress_" + label + ".json"))],
                           cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    names = ["v6_control", "v6_augmentation", "v8_units", "v8_units_count"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, name, index * 2) for index, name in enumerate(names)]
        for future in futures:
            future.result()
    (STORE / "diagnostics/stress_completed.json").write_text('{"completed": true}\n')


if __name__ == "__main__":
    main()
