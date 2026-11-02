"""冻结 ViT 位置编码的单项推理缓存，不缓存图像特征、不改变权重格式。"""

import torch


class FrozenPositionCache:
    """复用原插值函数，仅在 eval 且位置参数冻结时复用最近一次结果。

    缓存键覆盖输入网格、精度、设备、位置参数版本和插值设置。
    train、load_state_dict、to/_apply 会主动清空；普通参数更新也由版本号检测。
    直接修改 Parameter.data 会绕过 PyTorch 版本跟踪，调用方须主动 clear()。
    """

    def __init__(self, module):
        self.module = module
        self.original = module.interpolate_pos_encoding
        self.original_apply = module._apply
        self.original_train = module.train
        self.enabled = True
        self.key = None
        self.value = None
        module.interpolate_pos_encoding = self.interpolate
        module._apply = self.apply
        module.train = self.train
        module.register_load_state_dict_post_hook(self.loaded)

    def clear(self):
        self.key = None
        self.value = None

    def loaded(self, module, incompatible_keys):
        self.clear()

    def apply(self, *args, **kwargs):
        self.clear()
        try:
            return self.original_apply(*args, **kwargs)
        finally:
            self.clear()

    def train(self, mode=True):
        self.clear()
        return self.original_train(mode)

    def interpolate(self, x, width, height):
        module = self.module
        position = module.pos_embed
        if not self.enabled or module.training or position.requires_grad:
            self.clear()
            return self.original(x, width, height)
        try:
            version = position._version
        except RuntimeError:
            # inference tensor 没有版本计数，无法可靠判定更新，直接回退。
            self.clear()
            return self.original(x, width, height)

        patch_size = module.patch_size
        if isinstance(patch_size, (list, tuple)):
            patch_size = tuple(patch_size)
        key = (
            x.shape[1], x.shape[-1], width, height, x.device, x.dtype,
            id(position), position.data_ptr(), version, position.shape,
            position.device, position.dtype, patch_size,
            module.interpolate_offset, module.interpolate_antialias,
        )
        if key == self.key:
            return self.value
        self.clear()
        # 外部 inference_mode 下仍创建普通张量，供后续有梯度运算安全保存。
        with torch.inference_mode(False), torch.no_grad():
            value = self.original(x, width, height).clone()
        self.key = key
        self.value = value
        return value


def install_position_cache(module):
    """原地安装并返回控制器；重复安装保持幂等，state_dict 不增加字段。"""
    existing = getattr(module, "_frozen_position_cache", None)
    if existing is not None:
        return existing
    cache = FrozenPositionCache(module)
    module._frozen_position_cache = cache
    return cache
