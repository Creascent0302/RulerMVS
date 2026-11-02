"""六卡完成固定末步完整VAL；无需人工逐个启动，不访问TEST。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name("configs") / "fullval_plan.json"


def main(plan_path=PLAN):
    plan = json.loads(Path(plan_path).read_text())
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    script = plan.get("evaluator", str(Path(__file__).with_name("fullval.py")))
    for phase, merge in plan["phases"].items():
        jobs = [job for job in plan["jobs"] if job["phase"] == phase]
        for job in jobs:
            settings = json.loads(Path(job["config"]).read_text())
            for model in settings["models"]:
                if model.get("completion"):
                    if not json.loads(Path(model["completion"]).read_text())["completed"]:
                        raise RuntimeError("短训未完成，不提前验证")
        processes = []
        for job in jobs:
            log = (output / f"{phase}_rank{job['rank']}.log").open("w")
            process = subprocess.Popen([sys.executable, script, "--config", job["config"]],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes.append((process, log))
        codes = []
        for process, log in processes:
            codes.append(process.wait())
            log.close()
        if any(codes):
            raise RuntimeError(f"{phase}分片失败: {codes}")
        subprocess.run([sys.executable, script, "--config", merge], cwd=ROOT, check=True)
    (output / "completed.json").write_text(json.dumps({
        "completed": True, "test_accessed": False, "plan": str(plan_path), "time": time.time()
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PLAN))
    main(parser.parse_args().config)
