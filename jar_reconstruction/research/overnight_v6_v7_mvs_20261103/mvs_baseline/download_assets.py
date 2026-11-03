"""官方通用权重的小型补齐；只使用服务器eno5，并在tmux运行。"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from research.data_expansion_20261102.direct_http import fetch


def main():
    if not os.environ.get("TMUX"):
        raise RuntimeError("下载必须在tmux内运行")
    root = Path("/data/liuzonghao_data/dataset/public_expansion_20261102/mvs_initialization")
    url = "https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf4_finetune-8cc58f59.pth"
    total = 46910513
    block = math.ceil(total / 16)

    def chunk(index):
        start, stop = index * block, min((index + 1) * block, total)
        path = root / "parts" / str(index)
        if not path.exists() or path.stat().st_size != stop - start:
            fetch(url, path, byte_range=(start, stop - 1), max_bytes=stop - start, timeout=600)
        print("ready", index, flush=True)
        return path

    with ThreadPoolExecutor(max_workers=16) as pool:
        parts = list(pool.map(chunk, range(16)))
    destination = root / "resnet18_lpf4_finetune-8cc58f59.pth"
    temporary = destination.with_suffix(".complete")
    with temporary.open("wb") as stream:
        for part in parts:
            stream.write(part.read_bytes())
    sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if not sha.startswith("8cc58f59") or temporary.stat().st_size != total:
        raise RuntimeError("官方ResNet完整SHA前缀或大小校验失败")
    temporary.replace(destination)
    for part in parts:
        part.unlink()
    dino = root / "dinov2_vitb14_pretrain.pth"
    report = {
        "resnet": {"url": url, "path": str(destination), "sha256": sha},
        "dinov2": {"url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
                   "path": str(dino), "sha256": hashlib.sha256(dino.read_bytes()).hexdigest()},
        "network": "eno5; proxy disabled; per-request proof in ../network.jsonl",
    }
    (root / "sources.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
