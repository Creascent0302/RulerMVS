"""真实新增TRAIN上的原生损失、梯度、初始化与checkpoint烟雾检查。"""

import json
from pathlib import Path
import sys
import time

import torch

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move
from research.overnight_v6_v7_mvs_20261103.mvs_baseline.model import MVSAnywhereTrainingModel
from tools.smoke_outdoor_datasets import offline_only


def main():
    offline_only()
    torch.set_num_threads(2)
    torch.cuda.set_device(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261103)
    config = json.loads(Path("/data/liuzonghao_data/dataset/public_expansion_20261102/train_v6_expanded.json").read_text())
    config["augmentation"] = False
    directory = Path("/data/liuzonghao_data/dataset/public_expansion_20261102/mvs_initialization")
    config["initialization"].update(
        dinov2=str(directory / "dinov2_vitb14_pretrain.pth"),
        matching_antialiased=str(directory / "resnet18_lpf4_finetune-8cc58f59.pth"),
    )
    config["mvsanywhere"] = {"schedule": "budget_scaled", "refinements": 0}
    dataset = MixedDataset(config)
    model = MVSAnywhereTrainingModel(config).cuda(6)
    optimizer, scheduler = model.configure_optimizer(100)
    report = {"completed": False, "initialization": model.initialization_report,
              "records": [], "optimizer_steps": 0, "precision": "bf16 autocast"}
    for domain_index, domain in enumerate(config["datasets"]):
        # 新扩容域明确使用新增训练样本；其他域使用原TRAIN首组。
        index = {0: 3056, 2: 597}.get(domain_index, 0)
        sample = dataset[(domain_index, index, 0)]
        batch = move(collate_mixed([sample, sample]), torch.device("cuda", 6))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch["inputs"], domain, targets=batch["targets"])
        if not torch.isfinite(output["loss"]):
            raise RuntimeError("非有限原生损失")
        output["loss"].backward()
        norms = {}
        for name, parameters in (("matching", model.native_model.matching_model.parameters()),
                                 ("metadata", model.native_model.cost_volume.parameters()),
                                 ("reference_vit", model.native_model.encoder.parameters()),
                                 ("cost_volume_vit", model.native_model.cost_volume_net.parameters()),
                                 ("decoder", model.native_model.depth_decoder.parameters())):
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise RuntimeError(f"{name}分支梯度缺失或非有限")
            norms[name] = float(torch.sqrt(sum(g.float().square().sum() for g in gradients)))
            if norms[name] == 0:
                raise RuntimeError(f"{name}分支零梯度")
        optimizer.step()
        scheduler.step()
        report["optimizer_steps"] += 1
        torch.cuda.synchronize()
        record = dict(dataset=domain["name"], batch=2, views=batch["inputs"]["rgb"].shape[1],
                      seconds=time.monotonic() - started,
                      peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                      loss=float(output["loss"].detach()), gradient_norms=norms)
        report["records"].append(record)
        print(json.dumps(record), flush=True)
        (directory / "smoke_progress.json").write_text(json.dumps(report, indent=2) + "\n")
        del output

    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        first = model(batch["inputs"], domain)
        changed = {key: torch.zeros_like(value) if torch.is_tensor(value) else value
                   for key, value in batch["targets"].items()}
        second = model(batch["inputs"], domain, targets=changed)
        if not torch.equal(first["depth"], second["depth"]):
            raise RuntimeError("eval受targets影响")
        current, sources, _ = model.adapter.prepare_native(batch["inputs"], 0)
        current["min_depth"], current["max_depth"] = domain["near"], domain["far"]
        native = model.native_model("test", current, sources, return_mask=True,
                                    estimate_depth_range=False, num_refinement_steps=0)
        if not torch.equal(native["depth_pred_s0_b1hw"], first["depth"]):
            raise RuntimeError("包装输出与直接官方调用不一致")
    checkpoint = directory / "smoke_checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": config}, checkpoint)
    restored = MVSAnywhereTrainingModel(config, initialize=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    restored.load_state_dict(state["model"], strict=True)
    del state
    restored = restored.cuda(6).eval()
    with torch.inference_mode():
        restored_output = restored(batch["inputs"], domain)
    if not torch.equal(first["depth"], restored_output["depth"]):
        raise RuntimeError("checkpoint恢复输出不一致")
    checkpoint.unlink()
    report.update(completed=True, targets_isolated=True, native_forward_exact=True,
                  checkpoint_restore_exact=True, checkpoint_removed=True)
    (directory / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print("completed", flush=True)


if __name__ == "__main__":
    main()
