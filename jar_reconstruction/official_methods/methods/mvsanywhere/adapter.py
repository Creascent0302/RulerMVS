"""原始 MVSAnywhere Hero，保留两个 ViT-B、视图无关元数据和自适应范围。"""

import sys
import torch
from ..niantic_common import NianticAdapter, UPSTREAM, configure_options


class Adapter(NianticAdapter):
    prediction_scale = 1.0

    def __init__(self, config):
        super().__init__(config)
        height = self.settings.get("image_height", 480)
        width = self.settings.get("image_width", 640)
        if any(not isinstance(size, int) or size < 16 or size % 16 for size in (height, width)):
            raise ValueError(
                "MVSAnywhere 的 image_height/image_width 必须是正整数且为 16 的倍数；请在数据接口同步缩放 RGB 与内参。"
            )
        bins = self.settings.get("depth_bins", 64)
        if not isinstance(bins, int) or bins < 1:
            raise ValueError("MVSAnywhere 的 depth_bins 必须是正整数；官方 Hero 使用 64 个候选。")
        resume = config.get("_loading_shared_checkpoint", False)
        source = str(UPSTREAM / "mvsanywhere" / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from mvsanywhere.options import Options
        from mvsanywhere.experiment_modules.sr_depth_model import DepthModel

        opts = configure_options(
            Options(),
            config,
            dict(
                feature_volume_type="view_agnostic_mlp_feature_volume",
                matching_encoder_type="resnet",
                image_encoder_name="dinov2_vitb14",
                cv_encoder_type="vit_encoder",
                depth_decoder_name="dpt",
                matching_scale=0.25,
                prediction_scale=1.0,
                model_num_views=4,
                image_height=480,
                image_width=640,
                fast_cost_volume=bool(self.settings.get("fast_cost_volume", False)),
            ),
        )
        if (
            not resume
            and opts.pretrained
            and not self.settings.get("weights")
            and not self.settings.get("da_weights")
        ):
            raise ValueError(
                "Hero 官方训练初始化需要 model.da_weights（Depth Anything V2 Base）；仅 DINOv2 不能冒充此预训练。合成管线可显式 pretrained=false。"
            )
        if resume or self.settings.get("weights"):
            opts.pretrained = False
        opts.dinov2_source = str(UPSTREAM / "dinov2")
        opts.da_weights_path = None  # 下方严格检查加载，避免上游 try/except 静默忽略错误。
        opts.lr_da_encoder = float(
            config.get("training", {}).get("lr_da_encoder", opts.lr_da_encoder)
        )
        opts.lr_da_decoder = float(
            config.get("training", {}).get("lr_da_decoder", opts.lr_da_decoder)
        )
        self.native_model = DepthModel(opts)
        if self.settings.get("da_weights") and not resume:
            state = torch.load(self.settings["da_weights"], map_location="cpu", weights_only=False)
            encoder = {
                k.replace("pretrained", "model"): v
                for k, v in state.items()
                if k.startswith("pretrained.")
            }
            if not encoder:
                raise ValueError("所给文件不含 Depth Anything V2 pretrained.* 权重。")
            self.native_model.encoder.load_state_dict(encoder, strict=True)
            decoder = {
                k.replace("depth_head.", ""): v
                for k, v in state.items()
                if k.startswith("depth_head.") and "output_conv" not in k
            }
            result = self.native_model.depth_decoder.load_state_dict(decoder, strict=False)
            invalid_missing = [
                key
                for key in result.missing_keys
                if not key.startswith(("output_convs.", "sky_convs."))
            ]
            if result.unexpected_keys or invalid_missing:
                raise ValueError(
                    f"Depth Anything 解码器权重不完整：missing={invalid_missing}, unexpected={result.unexpected_keys}"
                )
        if self.settings.get("weights") and not resume:
            self.load_native_weights(self.settings["weights"])

    def call_native(self, cur, src, phase):
        if phase != "train" or "min_depth" not in cur:
            baseline = (src["cam_T_world_b44"] @ cur["world_T_cam_b44"][:, None])[..., :3, 3].norm(
                dim=-1
            )
            if (baseline <= 1e-7).any():
                raise ValueError("MVSAnywhere 原生启发式范围在零基线时退化；不能生成可信米制深度。")
        return self.native_model(
            phase,
            cur,
            src,
            unbatched_matching_encoder_forward=(
                phase != "train" and not self.settings.get("fast_cost_volume", False)
            ),
            return_mask=True,
            estimate_depth_range=phase != "train" or "min_depth" not in cur,
            num_refinement_steps=(
                0 if phase == "train" else int(self.settings.get("range_refinement_steps", 1))
            ),
        )

    def prepare_training_inputs(self, inputs, targets):
        """仅训练路径复制标签范围；eval forward 无条件用标定启发式覆盖它。"""
        if not self.training:
            raise RuntimeError("训练范围增强只能在 train() 模式调用。")
        inputs = super().prepare_training_inputs(inputs, targets)
        depth, valid = targets["depth"], targets["depth_valid"]
        if depth.ndim == 5:
            depth, valid = depth.squeeze(2), valid.squeeze(2)
        valid = valid.bool() & torch.isfinite(depth) & (depth > 0)
        minimum = depth.masked_fill(~valid, torch.inf).flatten(2).amin(-1)
        maximum = depth.masked_fill(~valid, -torch.inf).flatten(2).amax(-1)
        minimum = torch.where(torch.isfinite(minimum), minimum, torch.full_like(minimum, 10.0))
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.full_like(maximum, 10.0))
        minimum = minimum * (torch.rand_like(minimum) * 0.5 + 0.5)
        maximum = maximum * (torch.rand_like(maximum) + 1.0)
        return dict(inputs, _training_depth_range=torch.stack((minimum, maximum), -1))

    def configure_optimizer(self):
        # 三组参数不重叠；合并 AdamW 与逐组 LambdaLR 与官方三优化器逐步更新等价。
        net, opts = self.native_model, self.native_model.run_opts
        groups = [
            dict(
                params=list(net.cost_volume.parameters()) + list(net.matching_model.parameters()),
                lr=opts.lr,
            ),
            dict(params=net.encoder.parameters(), lr=opts.lr_da_encoder),
            dict(
                params=list(net.depth_decoder.parameters())
                + list(net.cost_volume_net.parameters()),
                lr=opts.lr_da_decoder,
            ),
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=opts.wd)

        def matching(step):
            return 1.0 if step < opts.lr_steps[0] else (0.1 if step < opts.lr_steps[1] else 0.01)

        def linear(step):
            return 1.0 - 0.999 * min(step, 70000) / 70000

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, [matching, linear, linear])


def create_adapter(config):
    return Adapter(config)
