"""完整 checkpoint 恢复不得依赖已移动的外部预训练文件。"""

from unittest.mock import Mock, patch

import torch

from official_methods.methods.simplerecon import Adapter


def test_resume_skips_external_pretraining():
    from official_methods.upstream.simplerecon.experiment_modules import depth_model

    constructor = Mock(return_value=torch.nn.Identity())
    config = {
        "_loading_shared_checkpoint": True,
        "model": {"pretrained": True, "weights": "/missing/original_hero.ckpt"},
    }
    with (
        patch.object(depth_model, "DepthModel", constructor),
        patch.object(Adapter, "load_native_weights") as load_native,
    ):
        Adapter(config)
    load_native.assert_not_called()
    assert constructor.call_args.args[0].pretrained is False
