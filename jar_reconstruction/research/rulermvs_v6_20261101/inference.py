"""可选的V5/V6部署准备：保留参数，只复用计算和编译两个纯融合段。"""

import os
from pathlib import Path

import torch

from jar.pretraining.inference import prepare_inference


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / 'runs/active/rulermvs_v6_20261101/inference_cache'


def prepare_fast_inference(model, options=None):
    """只接受已冻结的eval实例；重复同配置无操作，改配置需重新加载模型。

    三项代数复用默认开启；source/fusion编译显式选择，解析投影默认关闭。
    编译会按视图数、形状和范围特化。首次编译开销不属于稳态延迟。
    不改state_dict或训练默认；部署实例不用于恢复训练，训练重新加载原权重。
    """
    settings = {
        'spatial_layout': True,
        'factorized_fusion': True,
        'hoist_reference': True,
        'compact_pair': True,
        'projective_ruler': False,
        'compile_source': False,
        'compile_fusion': False,
        'cache_dir': str(DEFAULT_CACHE),
        'compile_threads': 2,
    }
    options = dict(options or {})
    unknown = set(options) - set(settings)
    if unknown:
        raise ValueError(f'未知推理选项：{sorted(unknown)}')
    settings.update(options)
    settings['cache_dir'] = str(Path(settings['cache_dir']).expanduser().resolve())
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('先执行model.eval().requires_grad_(False)，再准备部署计算')
    previous = getattr(model, '_v6_fast_inference_options', None)
    if previous is not None:
        if previous != settings:
            raise ValueError('部署配置已经安装；更改选项请重新加载原始checkpoint')
        return model

    keys = tuple(model.state_dict())
    prepare_inference(model, **{key: settings[key] for key in (
        'spatial_layout', 'factorized_fusion', 'hoist_reference', 'compact_pair', 'projective_ruler')})
    if settings['compile_source'] or settings['compile_fusion']:
        cache = Path(settings['cache_dir'])
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
        os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = str(settings['compile_threads'])
        # 七域各有粗/细两种形状与范围；默认8次不足，不静默退回eager。
        torch._dynamo.config.recompile_limit = 32
        torch._dynamo.config.suppress_errors = False
        if settings['compile_source']:
            model.decoder.source.forward = torch.compile(
                model.decoder.source.forward, fullgraph=True, dynamic=False)
        if settings['compile_fusion']:
            model.decoder._tokens_from_source = torch.compile(
                model.decoder._tokens_from_source, fullgraph=True, dynamic=False)
    if keys != tuple(model.state_dict()):
        raise RuntimeError('部署准备不应改变checkpoint的键')
    model._v6_fast_inference_options = settings
    return model
