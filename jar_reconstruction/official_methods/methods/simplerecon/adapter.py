"""原始 SimpleRecon Hero：EfficientNetV2-S + 元数据 MLP + ResNet 匹配。"""

import pickle
import types
from ..niantic_common import NianticAdapter, configure_options


class _OfficialUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "options":
            module = "official_methods.upstream.simplerecon.options"
        return super().find_class(module, name)


_checkpoint_pickle = types.ModuleType("simplerecon_checkpoint_pickle")
_checkpoint_pickle.__dict__.update(pickle.__dict__)
_checkpoint_pickle.Unpickler = _OfficialUnpickler


class Adapter(NianticAdapter):
    ddp_find_unused_parameters = False
    checkpoint_pickle_module = _checkpoint_pickle

    def __init__(self, config):
        super().__init__(config)
        resume = config.get("_loading_shared_checkpoint", False)
        from official_methods.upstream.simplerecon.options import Options
        from official_methods.upstream.simplerecon.experiment_modules.depth_model import DepthModel

        opts = configure_options(
            Options(),
            config,
            dict(
                feature_volume_type="mlp_feature_volume",
                matching_encoder_type="resnet",
                image_encoder_name="efficientnet",
                cv_encoder_type="multi_scale_encoder",
                depth_decoder_name="unet_pp",
                matching_scale=1,
                model_num_views=4,
            ),
        )
        if resume or self.settings.get("weights"):
            opts.pretrained = False  # 完整checkpoint覆盖全部可训练层，无需再次下载骨干。
        self.native_model = DepthModel(opts)
        from torchvision.transforms import ColorJitter

        self.color_transform = ColorJitter(0.2, 0.2, 0.2, 0.2)
        if self.settings.get("weights") and not resume:
            # 八目 Hero MLP 的输入宽度不同；strict=True 明确拒绝偷偷裁剪/补零权重。
            self.load_native_weights(self.settings["weights"])


def create_adapter(config):
    return Adapter(config)
