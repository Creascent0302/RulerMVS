"""加载原 checkpoint 后，显式开启不删层、不降分辨率的推理计算优化。"""

from .fast_spatial import install_spatial_layout


def prepare_inference(model, *, spatial_layout=False, factorized_fusion=False,
                      hoist_reference=False, compact_pair=False,
                      projective_ruler=None, compact_sources=None):
    """原位准备 eval 模型，参数和 state_dict 接口保持不变。

    默认关闭以便复现旧结果。建议由配置文件设置，在模型加载完成后调用一次。
    本轮部署验证使用 FP32；浮点重排不承诺所有输入逐位相同。
    """
    if model.training:
        raise ValueError("请在加载权重并调用 model.eval() 后准备推理")
    model.decoder.settings["factorized_fusion"] = factorized_fusion
    model.decoder.source.hoist_reference = hoist_reference
    model.decoder.source.compact_pair = compact_pair
    if projective_ruler is not None:
        model.decoder.settings["projective_ruler"] = projective_ruler
    if compact_sources is not None:
        model.decoder.settings["compact_sources"] = compact_sources
    if spatial_layout:
        install_spatial_layout(model)
    return model
