"""可选部署开关必须保留旧checkpoint的默认计算选择及参数。"""

import pytest
import torch
from torch import nn

from jar.pretraining.inference import prepare_inference
from jar.pretraining.model import CalibratedDecoder


def model():
    module = nn.Module()
    module.decoder = CalibratedDecoder({
        "channels": 16,
        "raw_channels": 32,
        "depth_bins": 12,
        "peak_count": 2,
        "near": 1.0,
        "far": 30.0,
        "use_scene": False,
        "local_kind": "scale_space",
        "projective_ruler": True,
        "compact_sources": True,
    })
    return module.eval()


def test_omitted_geometry_options_keep_checkpoint_settings():
    instance = model()
    prepare_inference(instance)
    assert instance.decoder.settings["projective_ruler"] is True
    assert instance.decoder.settings["compact_sources"] is True
    prepare_inference(instance, projective_ruler=False, compact_sources=False)
    assert instance.decoder.settings["projective_ruler"] is False
    assert instance.decoder.settings["compact_sources"] is False


def test_fast_options_preserve_all_parameters_and_state_keys():
    instance = model()
    parameters = {name: id(value) for name, value in instance.named_parameters()}
    state = {name: value.clone() for name, value in instance.state_dict().items()}
    prepare_inference(instance, spatial_layout=True, factorized_fusion=True,
                      hoist_reference=True, compact_pair=True,
                      projective_ruler=True, compact_sources=True)
    assert parameters == {name: id(value) for name, value in instance.named_parameters()}
    assert instance.state_dict().keys() == state.keys()
    assert all(torch.equal(value, state[name]) for name, value in instance.state_dict().items())
    assert instance.decoder.settings["factorized_fusion"] is True
    assert instance.decoder.source.hoist_reference is True
    assert instance.decoder.source.compact_pair is True


def test_inference_preparation_rejects_training_mode():
    with pytest.raises(ValueError, match="eval"):
        prepare_inference(model().train(), projective_ruler=True)
