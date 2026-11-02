"""在相同完整四目VAL输入上测四参考顺序输出；不读取标签，不包含融合。"""

import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import Models, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from tools.benchmark_public_speed import prepare_model, summarize
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import fingerprint, offline_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = json.loads(Path(parser.parse_args().config).read_text())
    offline_only()
    device = configure(config, 0)
    domain = config["datasets"][0]
    dataset = build_dataset(dataset_config(domain, config), "val", load_targets=False)
    rows = [json.loads(line) for line in Path(domain["manifests"]["val"]).read_text().splitlines()]
    complete = [i for i, row in enumerate(rows) if row["valid_camera_count"] == 4]
    selected = [complete[0], complete[len(complete) // 2], complete[-1]]
    domains = [dict(domain, reference_index=i) for i in range(4)]
    models = Models(config, device)
    names = list(config["methods"])
    frames = []
    with torch.inference_mode():
        for index in selected:
            with patch("numpy.load", side_effect=AssertionError("测速不允许读取深度标签")):
                cpu = collate_frames([dataset[index]])["inputs"]
            inputs = move(cpu, device)
            for name in names:
                prepare_model(models, name, config["methods"][name], domain, inputs)
            for _ in range(8):
                for name in names:
                    for d in domains:
                        models.convert(models.forward(name, d, inputs), inputs)
            samples = {name: [] for name in names}
            for iteration in range(20):
                order = names[iteration % len(names):] + names[:iteration % len(names)]
                for name in order:
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    results = [models.convert(models.forward(name, d, inputs), inputs) for d in domains]
                    torch.cuda.synchronize(device)
                    samples[name].append((time.perf_counter() - begin) * 1000)
                    del results
            frames.append({"val_index": index, "input_sha256": fingerprint(cpu), "samples_ms": samples})
    summary = {name: summarize([v for row in frames for v in row["samples_ms"][name]]) for name in names}
    for row in summary.values():
        row["four_reference_hz"] = row.pop("single_reference_hz")
    write_json(config["result_file"], {"completed": True, "summary": summary, "frames": frames,
        "protocol": "GPU0独占，FP32/TF32off，共同冻结位置缓存；VAL三完整帧各8轮预热20次交替计时。",
        "scope": "顺序输出四张参考深度图，包含输出格点转换，不含读盘/H2D/点云融合；没有跨参考编码复用。"})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
