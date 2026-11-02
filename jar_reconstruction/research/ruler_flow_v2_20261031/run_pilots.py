"""六组短试按GPU0–5并行，所有输出写数据盘；不启动正式八卡训练。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).parent
OUTPUT = ROOT / "runs/active/ruler_flow_v2_20261031"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    processes = []
    logdir = OUTPUT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    names = [phase + "_" + variant for phase in ("public", "carla_new")
             for variant in ("control", "direct", "meanflow")]
    for gpu, name in enumerate(names):
        stem = ("pilot_smoke_" if args.smoke else "pilot_") + name
        filename = FOLDER / "configs" / (stem + ".json")
        config = json.loads(filename.read_text())
        if (Path(config["output"]) / "completed.json").exists():
            print("已完成，保留：", stem, flush=True)
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                   PYTHONUNBUFFERED="1")
        log = (logdir / (stem + ".log")).open("a")
        command = [str(ROOT / "tools/with_server_network.sh"),
                   str(ROOT / "official_methods/.venv/bin/python"),
                   str(FOLDER / "pilot.py"), "--config", str(filename)]
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((stem, process, log))
        print(json.dumps({"started": stem, "gpu": gpu, "pid": process.pid}), flush=True)
    failures = []
    while processes:
        active = []
        for name, process, log in processes:
            status = process.poll()
            if status is None:
                active.append((name, process, log))
                continue
            log.close()
            (logdir / (name + ".exit_code.txt")).write_text(str(status) + "\n")
            print(json.dumps({"finished": name, "exit_code": status}), flush=True)
            if status:
                failures.append(name)
        processes = active
        if processes:
            time.sleep(2)
    if failures:
        raise RuntimeError("短试失败，现场保留：" + ", ".join(failures))


if __name__ == "__main__":
    main()
