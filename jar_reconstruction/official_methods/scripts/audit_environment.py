"""只读记录真实软硬件，不将 import 成功当作模型测试。"""

import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import torch


def command(args):
    try:
        run = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {
            "command": args,
            "returncode": run.returncode,
            "stdout": run.stdout,
            "stderr": run.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "error": str(exc)}


def main():
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible": torch.cuda.is_available(),
        "torch_arch_list": torch.cuda.get_arch_list(),
        "packages": {
            item.metadata["Name"]: item.version for item in importlib.metadata.distributions()
        },
        "commands": [
            command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,driver_version",
                    "--format=csv",
                ]
            ),
            command(["nvcc", "--version"]),
            command(["/usr/local/cuda-12.8/bin/nvcc", "--version"]),
            command(["g++-10", "--version"]),
            command(["cmake", "--version"]),
        ],
    }
    result["gpus"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i)),
            "total_bytes": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ]
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "official_methods/artifacts/environment.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(path)


if __name__ == "__main__":
    main()
