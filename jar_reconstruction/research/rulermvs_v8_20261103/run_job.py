"""单实验独立启动器，参数和GPU全部来自配置，适合tmux后台执行。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


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
        raise FileExistsError("已有last.pt；须显式resume，禁止覆盖实验")
    gpus = config["launch"]["gpus"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={len(gpus)}", "research/rulermvs_v8_20261103/train.py",
               "--config", str(source)]
    if args.resume:
        command.extend(["--resume", args.resume])
    status = dict(state="running", gpus=gpus, command=command,
                  started=datetime.now(timezone.utc).isoformat())
    path = output / "job_status.json"

    def record():
        path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")

    with (output / "console.log").open("a", buffering=1) as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT)
        status["pid"] = process.pid
        record()
        code = process.wait()
    status.update(state="completed" if code == 0 else "failed", exit_code=code,
                  finished=datetime.now(timezone.utc).isoformat())
    record()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
