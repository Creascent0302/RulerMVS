"""等待冻结测速结束，独占分配每卡一个单进程任务；训练后完整VAL。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import write_json


def wait_for_release():
    root = ROOT / "runs/active/rulermvs_v6_20261101/full_training"
    while True:
        main = root / "evaluation_recovery/queue.exit_code.txt"
        speed = root / "paper_firstpass/finish.exit_code.txt"
        done = root / "paper_firstpass/completed.json"
        if all(path.exists() for path in (main, speed, done)):
            if main.read_text().strip() != "0" or speed.read_text().strip() != "0":
                raise RuntimeError("主评测或独占速度未成功，不启动训练")
            if json.loads(done.read_text()).get("completed"):
                result = subprocess.run(["nvidia-smi", "-i", "0,1,2,3,4,5,6", "--query-compute-apps=pid", "--format=csv,noheader"],
                                        capture_output=True, text=True, check=True)
                if not result.stdout.strip():
                    return
        print(json.dumps({"status": "waiting_main_and_firstpass_speed_release"}), flush=True)
        time.sleep(30)


def run(settings):
    output = Path(settings["output"])
    state = {"completed": False, "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "stages": {}, "errors": []}
    logdir = output / "queue_logs"
    logdir.mkdir(exist_ok=True)
    wait_for_release()
    python = str(ROOT / "official_methods/.venv/bin/python")
    stages = (("smoke_train", settings["smoke_jobs"]), ("smoke_val", settings["smoke_jobs"]),
              ("train", settings["train_jobs"]), ("shared_val", settings["shared_validation_jobs"]))
    for stage, jobs in stages:
        training = stage.endswith("train")
        shared = stage == "shared_val"
        pending = list(jobs)
        active = {}
        rows = state["stages"][stage] = []
        while pending or active:
            for gpu in settings["gpus"]:
                if gpu in active or not pending:
                    continue
                job = pending.pop(0)
                destination = Path(job["output"])
                audit = "geometry_audit.json" if job["name"].startswith("geometry_") else "handoff_audit.json"
                marker = destination / ("completed.json" if shared else audit if training else "full_val.json")
                if marker.exists() and json.loads(marker.read_text()).get("completed"):
                    rows.append({"name": job["name"], "status": "already_complete"})
                    continue
                if not training and not shared and not (destination / audit).exists():
                    rows.append({"name": job["name"], "status": "skipped_failed_training"})
                    continue
                script = job["script"] if training or shared else str(Path(__file__).with_name("handoff_fullval.py"))
                logfile = logdir / f"{stage}_{job['name']}.log"
                handle = logfile.open("a")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
                process = subprocess.Popen([python, script, "--config", job["config"]], cwd=ROOT,
                                           env=env, stdout=handle, stderr=subprocess.STDOUT)
                record = {"name": job["name"], "gpu": gpu, "pid": process.pid,
                          "status": "running", "log": str(logfile), "started_epoch": time.time()}
                rows.append(record)
                active[gpu] = (process, handle, record)
                print(json.dumps({"stage": stage, **record}), flush=True)
            for gpu, (process, handle, record) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                record.update(status="complete" if code == 0 else "failed", exit_code=code,
                              elapsed_seconds=time.time() - record["started_epoch"])
                if code:
                    state["errors"].append(dict(stage=stage, **record))
                del active[gpu]
                print(json.dumps({"stage": stage, **record}), flush=True)
            write_json(output / "queue_progress.json", state)
            if pending or active:
                time.sleep(2)
        if stage.startswith("smoke") and state["errors"]:
            write_json(output / "queue_completed.json", state)
            (output / "queue.exit_code.txt").write_text("1\n")
            raise RuntimeError("GPU预检失败；不启动正式消融，失败证据保留")
    if not state["errors"]:
        from research.paper_ablation_20261102.handoff_sharedval import merge
        merge(settings)
    state["completed"] = not state["errors"]
    state["ended_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(output / "queue_completed.json", state)
    (output / "queue.exit_code.txt").write_text("1\n" if state["errors"] else "0\n")
    if state["errors"]:
        raise RuntimeError(f"{len(state['errors'])}个任务失败，详见queue_completed.json；成功结果保留")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
