"""固定权重、真实 VAL 编码的融合重排对照，不训练、不读取 TEST。"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import write_json
from jar.pretraining.model import CalibratedDecoder
from jar.runtime import move


@torch.no_grad()
def run(config):
    torch.set_num_threads(2)
    torch.cuda.set_device(config["gpu"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", config["gpu"])
    saved = torch.load(config["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    decoder = CalibratedDecoder(saved["config"]["model"]).to(device).eval()
    state = {key.removeprefix("decoder."): value for key, value in saved["model"].items()
             if key.startswith("decoder.")}
    decoder.load_state_dict(state, strict=True)
    index = json.loads(Path(config["cache"]).read_text())
    result = {"config": config, "domains": {}, "test_accessed": False}
    for domain in index["domains"]:
        if domain["name"] not in config["datasets"]:
            continue
        path = index["splits"]["confirmation"][domain["name"]][0]
        work = move(torch.load(path, map_location="cpu", weights_only=False)["work"], device)
        decoder.set_range(domain["near"] / domain["depth_scale"], domain["far"] / domain["depth_scale"])
        times = {False: [], True: []}
        outputs = {}
        for flag in times:
            decoder.settings["factorized_fusion"] = flag
            for _ in range(config["warmup"]):
                outputs[flag] = decoder(work)
        for iteration in range(config["repetitions"]):
            for flag in ((False, True) if iteration % 2 == 0 else (True, False)):
                decoder.settings["factorized_fusion"] = flag
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = decoder(work)
                torch.cuda.synchronize()
                times[flag].append((time.perf_counter() - start) * 1000)
                del output
        difference = (outputs[False]["depth"] - outputs[True]["depth"]).abs() * domain["depth_scale"]
        row = {"decoder_ms": {str(flag): statistics.median(values) for flag, values in times.items()},
               "depth_difference_mean": difference.mean().item(),
               "depth_difference_max": difference.max().item(), "units": domain["geometry_units"],
               "sample": path}
        result["domains"][domain["name"]] = row
        print(json.dumps({domain["name"]: row}), flush=True)
    write_json(Path(config["output"]) / "benchmark.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
