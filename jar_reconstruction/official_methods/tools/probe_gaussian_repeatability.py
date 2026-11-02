"""固定标定/RGB/RNG诊断官方GF2的重复坐标及数值可重复性，不改变预测算法。"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mmengine.logging import MMLogger
from official_methods.methods.gaussianformer2 import Adapter
from jar.synthetic import make_synthetic_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="official_methods/configs/gaussianformer2_carla.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conv-algo", choices=["default", "native", "implicit"], default="default")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    MMLogger.get_instance("gf_repeatability", log_level="ERROR")
    config = json.loads(Path(args.config).read_text())
    config["model"]["synthetic_random_init"] = True
    model = Adapter(config).cuda().eval()
    from spconv.pytorch import SubMConv3d
    from spconv.core import ConvAlgo

    convs = [module for module in model.modules() if isinstance(module, SubMConv3d)]
    original_algorithms = sorted({str(module.algo) for module in convs})
    if args.conv_algo != "default":
        for module in convs:
            module.algo = (
                ConvAlgo.Native if args.conv_algo == "native" else ConvAlgo.MaskImplicitGemm
            )
    height, width = config["image_size"]
    inputs = {
        key: value.cuda()
        for key, value in make_synthetic_batch(N=4, H=height, W=width, seed=17)["inputs"].items()
    }
    lifter_outputs, sparse_inputs, sparse_outputs = [], [], []

    def lifter_hook(module, inputs, output):
        lifter_outputs.append(output["representation"].detach().cpu())

    def sparse_pre_hook(module, arguments):
        sparse_inputs.append(
            (arguments[0].features.detach().cpu(), arguments[0].indices.detach().cpu())
        )

    def sparse_hook(module, arguments, output):
        sparse_outputs.append(output.features.detach().cpu())

    model.native.lifter.register_forward_hook(lifter_hook)
    convs[0].register_forward_pre_hook(sparse_pre_hook)
    convs[0].register_forward_hook(sparse_hook)
    with torch.no_grad():
        torch.manual_seed(41)
        np.random.seed(41)
        adapted = model(inputs)
        torch.manual_seed(41)
        np.random.seed(41)
        native = model.native(**model.native_inputs(inputs))
    probability_a, probability_b = adapted["native"]["pred_occ"][-1], native["pred_occ"][-1]
    indices = sparse_inputs[0][1]
    unique, counts = indices.unique(dim=0, return_counts=True)
    report = dict(
        config=args.config,
        random_init=True,
        gaussians=int(indices.shape[0]),
        image_size=[height, width],
        gpu=torch.cuda.get_device_name(),
        roi=model.roi,
        requested_conv_algorithm=args.conv_algo,
        original_conv_algorithms=original_algorithms,
        used_conv_algorithms=sorted({str(module.algo) for module in convs}),
        fixed_rgb_calibration_rng=True,
        lifter_representation_max_abs_difference=float(
            (lifter_outputs[0] - lifter_outputs[1]).abs().max()
        ),
        first_spconv_input_feature_max_abs_difference=float(
            (sparse_inputs[0][0] - sparse_inputs[1][0]).abs().max()
        ),
        first_spconv_input_indices_identical=torch.equal(sparse_inputs[0][1], sparse_inputs[1][1]),
        unique_first_spconv_indices=int(len(unique)),
        duplicate_entries=int(len(indices) - len(unique)),
        duplicate_entry_fraction=float((len(indices) - len(unique)) / len(indices)),
        entries_in_colliding_cells=int(counts[counts > 1].sum()),
        first_spconv_output_max_abs_difference=float(
            (sparse_outputs[0] - sparse_outputs[1]).abs().max()
        ),
        probability_max_abs_difference=float((probability_a - probability_b).abs().max()),
        probability_mean_abs_difference=float((probability_a - probability_b).abs().mean()),
        semantic_disagreement_fraction=float(
            (adapted["native"]["final_occ"] != native["final_occ"]).float().mean()
        ),
        strict_native_adapter_equality=bool(torch.equal(probability_a, probability_b)),
        limitation="官方稀疏卷积输入含重复坐标；本探针不去重、不平均高斯、不更换网络。",
    )
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
