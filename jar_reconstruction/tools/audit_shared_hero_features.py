"""核对自研 ViT 原始特征是否与原始 Hero 相同，不读取任何标签。"""

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import comparison_dataset, read_json
from jar.data import collate_frames
from jar.runtime import move
from tools.carla_compare import file_sha256, make_model


@torch.inference_mode()
def run(plan, device):
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    own, _ = make_model(plan["own"], plan["own"]["weights"], device)
    raw = torch.load(
        plan["pretrained_weights"], map_location="cpu", weights_only=False, mmap=True
    )["state_dict"]
    reference_model = copy.deepcopy(own.encoder.foundation.model)
    prefix = "encoder.model."
    reference_model.load_state_dict(
        {key[len(prefix):]: value for key, value in raw.items() if key.startswith(prefix)},
        strict=True,
    )
    protocol = read_json(plan["protocol"])
    reference = protocol["evaluation"]["reference_index"]
    hero_layers = [0, 2, 4, 5, 8, 11]
    results = []
    for split, indices in plan["splits"].items():
        index = indices[0]
        dataset = comparison_dataset(protocol, split, load_targets=False)
        inputs = move(collate_frames([dataset[index]]), device)["inputs"]
        inputs = dict(inputs, _reference_indices=(reference,))
        batch, views, _, height, width = inputs["rgb"].shape
        if height % 16 or width % 16:
            raise ValueError("本对照使用两边均无填充的16整数倍图像")
        features = own.encoder.foundation(inputs)
        image = inputs["rgb"][:, reference].float()
        image = (image - own.encoder.foundation.mean) / own.encoder.foundation.std
        image = F.interpolate(image, scale_factor=14 / 16, mode="bilinear", align_corners=False)
        hero_features = reference_model.get_intermediate_layers(
            image, hero_layers, return_class_token=True
        )
        differences = []
        for field, layer in zip(features, own.encoder.foundation.indices):
            own_tokens = field.unflatten(0, (batch, views))[:, reference].flatten(2).transpose(1, 2)
            original = hero_features[hero_layers.index(layer)][0]
            differences.append({
                "block_one_based": layer + 1,
                "shape": list(own_tokens.shape),
                "max_abs": float((own_tokens - original).abs().max()),
                "exact_equal": torch.equal(own_tokens, original),
            })
        results.append({"split": split, "frame_index": index, "layers": differences})
    output = Path(plan["output"]).parent / "shared_feature_equivalence.json"
    output.write_text(json.dumps({
        "same_raw_hero_weights_and_same_rgb": results,
        "script_sha256": file_sha256(__file__),
        "note": "比较压缩、门控及重采样之前的原始参考图patch特征；双方均不读取标签。",
    }, ensure_ascii=False, indent=2) + "\n")
    print(output)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(read_json(args.config), torch.device(args.device))
