"""启动前记录旧权重保护、通用初始化、实际训练参数量和源码快照。"""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from research.v2_restart_20261029.training import RestartHooks


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    torch.set_num_threads(2)
    folder = Path(__file__).parent
    output = ROOT / "runs/active/v2_restart_20261029"
    configs = {name: json.loads((folder / "configs" / ("train_" + name + ".json")).read_text())
               for name in ("public", "carla_new")}
    protected = {
        "runs/model_releases/RulerMVS-V2/public_recommended/b411275c3203/pretrained.pt":
            "b411275c3203d11e526e3f34142dbe6d3a572e5c1def71156ecb257b12b497bd",
        "runs/model_releases/RulerMVS-V2/carla_recommended/7888b8b637d1/pretrained.pt":
            "7888b8b637d16753e35f65a0ce3674d482f48f6d00c5bcd54573e76f08efec5d",
        "runs/active/v2_training_20261028/carla_new/pretrained.pt":
            "b45624ab09a176c37c17f5e78641b106fb6d0be96ee6779e8218730096088f91",
    }
    for path, expected in protected.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"旧保护权重SHA不符：{path}")
    public = configs["public"]
    torch.manual_seed(public["seed"])
    model = RestartHooks(public).model_factory(public, initialize=True)
    old = torch.load(ROOT / next(iter(protected)), map_location="cpu", weights_only=False, mmap=True)
    changed = [name for name, tensor in model.state_dict().items()
               if not torch.equal(tensor, old["model"][name])]
    if not any(name.startswith("decoder.") for name in changed):
        raise RuntimeError("候选解码器没有重新随机初始化")
    source = torch.load(public["initialization"]["foundation"], map_location="cpu", weights_only=True, mmap=True)
    for name, tensor in model.foundation.model.state_dict().items():
        if not torch.equal(tensor, source["pretrained." + name]):
            raise RuntimeError(f"通用ViT初始化不符：{name}")
    del source, old
    source = torch.load(public["initialization"]["matching"], map_location="cpu", weights_only=True, mmap=True)
    initialized_counters = []
    for name, tensor in model.matching.backbone.state_dict().items():
        # 旧版官方ResNet文件没有BN计数器；PyTorch加载器按兼容规则补零。
        if name not in source and name.endswith("num_batches_tracked") and int(tensor) == 0:
            initialized_counters.append(name)
            continue
        if name not in source or not torch.equal(tensor, source[name]):
            raise RuntimeError(f"ImageNet匹配初始化不符：{name}")
    del source
    stages = []
    for stage in public["stages"]:
        model.set_stage(stage["number"])
        optimizer = RestartHooks.optimizer_factory(model, stage, public)
        stages.append({"stage": stage["number"], "groups": [
            {"name": group["group_name"], "lr": group["lr"],
             "parameters": sum(p.numel() for p in group["params"])} for group in optimizer.param_groups]})
    del model, optimizer
    manifests = {}
    for config in configs.values():
        for domain in config["datasets"] + config.get("validation_datasets", []):
            for path in domain["manifests"].values():
                manifests[path] = digest(path)
            if manifests[domain["manifests"]["train"]] != domain["train_manifest_sha256"]:
                raise RuntimeError("训练清单SHA改变")
    files = list((ROOT / "jar").rglob("*.py")) + list(folder.rglob("*.py"))
    files += list(folder.rglob("*.json")) + list(folder.glob("*.sh"))
    files += list((ROOT / "research/v2_training_20261028").glob("*.py"))
    files += [ROOT / "research/near_mid_20261027/training.py", ROOT / "tools/benchmark_public_speed.py"]
    files += list((ROOT / "research/near_mid_evaluation_20261028").glob("*.py"))
    snapshot = output / "launch_source.zip"
    with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(files)):
            archive.write(file, file.relative_to(ROOT))
    result = {"configs": configs, "protected_weights": protected, "manifest_sha256": manifests,
              "fresh_vs_old_different_tensors": len(changed), "generic_initialization_verified": True,
              "compatibility_initialized_bn_counters": initialized_counters,
              "trainable_stages": stages, "snapshot_sha256": digest(snapshot),
              "free_gib": shutil.disk_usage(output).free / 2 ** 30,
              "no_old_model_warmstart_or_teacher": True}
    (output / "launch_protocol.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in ("fresh_vs_old_different_tensors",
          "generic_initialization_verified", "trainable_stages", "free_gib")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
