"""固定16 VAL/域的逐帧度量与候选交接诊断，标签只用于评分。"""

import torch
from torch.utils.data import DataLoader

from jar.pretraining.data import collate_mixed
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import ground_truth, regions
from research.rulermvs_v6_20261101.diagnose import cpu_prediction, finish, update_record


@torch.no_grad()
def evaluate(model, dataset, val_plan, device):
    model.eval()
    report = {"completed": False, "split": "val", "test_accessed": False, "datasets": {}}
    for domain_index, domain in enumerate(dataset.domains):
        plan = val_plan[domain["name"]]
        # MixedDataset已经把参考相机重排到槽0；标定和GT跟随同一重排。
        local_domain = dict(domain, reference_index=0)
        keys = [(domain_index, index, -1) for index in plan["indices"]]
        loader = DataLoader(dataset, batch_size=1, sampler=keys, collate_fn=collate_mixed, num_workers=0)
        totals, rows = {}, []
        for index, batch in zip(plan["indices"], loader):
            prediction = model(move(batch["inputs"], device), domain, reference=0)
            scale = float(prediction["unit_scale"])
            prediction = cpu_prediction(prediction)
            truth, valid = ground_truth(batch, local_domain)
            masks = regions(truth, valid, domain)
            record = {}
            update_record(record, prediction, truth, masks, scale, False)
            update_record(totals, prediction, truth, masks, scale, False)
            rows.append({"index": index, "valid_views": int(batch["inputs"]["view_mask"].sum()),
                         "metrics": finish(record), "raw": record})
        report["datasets"][domain["name"]] = {
            "units": domain["geometry_units"], "indices": plan["indices"],
            "manifest_sha256": plan["manifest_sha256"], "metrics": finish(totals),
            "raw": totals, "frames": rows,
        }
    report["completed"] = True
    return report
