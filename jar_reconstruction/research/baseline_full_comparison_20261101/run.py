"""独占测速后运行完整验证、四参考TEST、随机重复和公共跨域TEST。"""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import write_json

HERE = Path(__file__).parent
OUTPUT = ROOT / "runs/active/baseline_full_comparison_20261101"
PYTHON = str(ROOT / "official_methods/.venv/bin/python")


def main():
    stages = [
        ("single_speed", "benchmark.py", []), ("four_speed", "benchmark.py", ["--four"]),
        ("validation", "evaluate.py", []), ("carla", "evaluate.py", []),
        ("repeats", "evaluate.py", []), ("public_transfer", "evaluate.py", []),
    ]
    completed = []
    for name, script, flags in stages:
        write_json(OUTPUT / "progress.json", {"completed": False, "active": name, "finished": completed,
                   "time_utc": datetime.now(timezone.utc).isoformat()})
        command = [PYTHON, str(HERE / script), "--config", str(HERE / f"configs/{name}.json"), *flags]
        print("START", name, command, flush=True)
        with (OUTPUT / f"{name}.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        completed.append(name)
        print("COMPLETE", name, flush=True)
        if name in ("repeats", "public_transfer"):
            subprocess.run([PYTHON, str(HERE / "report.py")], cwd=ROOT, check=True)
    write_json(OUTPUT / "progress.json", {"completed": True, "active": None, "finished": completed,
               "time_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_json(OUTPUT / "failure.json", {"completed": False, "traceback": traceback.format_exc()})
        raise
