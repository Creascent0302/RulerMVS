"""供tmux调用的单模型启动器；显卡和训练参数仅从配置读取。"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    source = Path(args.config).resolve()
    config = json.loads(source.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("已有训练权重，必须显式提供resume；禁止静默覆盖")
    gpus = config["launch"]["gpus"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2",
                       TZ="Asia/Shanghai", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={len(gpus)}", "research/overnight_v6_v7_mvs_20261103/train.py",
               "--config", str(source)]
    if args.resume:
        command += ["--resume", args.resume]
    record = dict(method=config["method"], gpus=gpus, command=command, config=str(source),
                  started=datetime.now().astimezone().isoformat(), state="running")
    status = output / "job_status.json"
    status.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    with (output / "console.log").open("a", buffering=1) as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record["pid"] = process.pid
        status.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        code = process.wait()
    record.update(state="completed" if code == 0 else "failed", exit_code=code,
                  finished=datetime.now().astimezone().isoformat())
    status.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
