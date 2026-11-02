"""V5最终权重验收补测：同卡真实四参考，含官方首遍与一次细化。

仅读取VAL RGB与标定。保留各模型原算法，不给baseline添加未验证缓存。
"""

import argparse
import copy
import gc
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import Models, digest, write_json, equal_pixel_inputs
from jar.pretraining.data import dataset_config
from jar.runtime import move, seed_all
from research.rulermvs_v5_20261031 import evaluation  # 独立新架构工厂
from research.ruler_flow_v2_20261031.latency import shared_original
from tools.smoke_outdoor_datasets import fingerprint, offline_only


def summary(samples):
    median = float(np.median(samples))
    return {"median_ms": median, "mean_ms": float(np.mean(samples)),
            "p95_ms": float(np.percentile(samples, 95)), "hz": 1000 / median,
            "calls": len(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    config = json.loads(Path(settings["data_config"]).read_text())
    output = Path(settings["output"])
    if output.exists():
        raise FileExistsError(output)
    config["methods"].pop("v5_final")
    config["methods"]["mvsanywhere_first"] = copy.deepcopy(config["methods"]["mvsanywhere_refined"])
    config["methods"]["mvsanywhere_first"]["refinements"] = 0
    offline_only()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(20261101)
    models = Models(config, device)
    domain = config["datasets"][0]
    dataset = build_dataset(dataset_config(domain, config), "val", load_targets=False)
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    durations = {}
    fingerprints = {}
    exact = []
    generator = random.Random(20261101)
    for index in indices:
        cpu_inputs = collate_frames([dataset[index]])["inputs"]
        if not bool(cpu_inputs["view_mask"].all()):
            raise RuntimeError("预登记测速帧不是完整四目，不静默更换样例")
        fingerprints[str(index)] = fingerprint(cpu_inputs)
        inputs = move(cpu_inputs, device)
        with torch.inference_mode(False):
            original = models.get("original_v2", domain)
            updated = models.get("v5_best", domain)
            baseline = models.get("mvsanywhere_refined", domain)

        def serial(name):
            return {reference: models.forward(name, dict(domain, reference_index=reference), inputs)
                    for reference in range(4)}

        def shared(model, new):
            work = equal_pixel_inputs(inputs) if config["methods"]["v5_best"]["input_mode"] == "equal_pixels" else inputs
            native = (model.forward_references(work, domain, range(4)) if new else
                      shared_original(model, work, domain, range(4)))
            return {reference: {"depth": row["depth"] * row["unit_scale"], "valid": row["valid"]}
                    for reference, row in native.items()}

        methods = {}
        for name in config["methods"]:
            methods[name + "_single"] = lambda name=name: models.forward(name, domain, inputs)
            methods[name + "_serial_four"] = lambda name=name: serial(name)
        methods["original_v2_shared_four"] = lambda: shared(original, False)
        methods["v5_best_shared_four"] = lambda: shared(updated, True)
        with torch.inference_mode():
            for name, model, new in (("original_v2", original, False), ("v5_best", updated, True)):
                a, b = serial(name), shared(model, new)
                fields = [torch.equal(a[reference][field], b[reference][field])
                          for reference in range(4) for field in ("depth", "valid")]
                if not all(fields):
                    raise RuntimeError("同权重共享匹配入口输出不一致")
                exact.append({"index": index, "method": name, "depth_valid_exact": True})
            for forward in methods.values():
                for _ in range(settings["warmup"]):
                    forward()
            caches = [original.foundation.model, updated.foundation.model,
                      baseline.native_model.encoder.model, baseline.native_model.cost_volume_net.model]
            if any(model._frozen_position_cache.key is None for model in caches):
                raise RuntimeError("位置编码缓存没有实际命中")
            for _ in range(settings["repeats"]):
                order = list(methods)
                generator.shuffle(order)
                for name in order:
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    prediction = methods[name]()
                    torch.cuda.synchronize(device)
                    durations.setdefault(name, []).append(1000 * (time.perf_counter() - start))
                    del prediction
        print(json.dumps({"frame_done": index, "methods": len(methods)}, ensure_ascii=False), flush=True)
        del inputs, cpu_inputs
    write_json(output, {
        "completed": True, "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "settings": settings, "config": config, "indices": indices,
        "input_sha256": fingerprints, "methods": {key: summary(value) for key, value in durations.items()},
        "samples_ms": durations, "shared_equality": exact,
        "checkpoint_sha256": {name: digest(value["weights"]) for name, value in config["methods"].items()},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
                           ROOT / "jar/pretraining/comparison.py", ROOT / "research/rulermvs_v5_20261031/model.py", ROOT / "research/rulermvs_v5_20261031/bridge.py"]},
        "precision": "FP32，TF32关闭；相同位置缓存、同卡交替、10次预热/帧/方法、40次测量/帧/方法",
        "scope": "GPU常驻输入；实际single或四参考输出；含原生输入适配，不含IO/H2D/预处理/场景融合。",
        "limits": "输入模式见config；native自研匹配512x896，equal_pixels只使用共同256x448图再上采样；baseline未添加额外共享编码。单参考与四参考不跨列比较。",
        "test_labels_accessed": False,
        "memory": "多模型共驻，本补测不把共驻显存解释为方法独占显存。",
    })
    models.loaded.clear()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
