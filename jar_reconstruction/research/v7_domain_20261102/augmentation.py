"""只用于 TRAIN 的真实源子集和同组灰度增强；不读取标签或改变标定。

输入是已经 collate 的 B×N 字典。保留参考槽 0；缺失槽不会被重新激活。
这些操作模拟灰度传感器/较少真实相机，不模拟新的基线或新几何。
"""

import torch


def augment_inputs(inputs, *, grayscale_probability=0.25, pair_probability=0.20,
                   triple_probability=0.10, generator=None):
    """返回新字典和每样本增强记录；原输入及 targets 均不修改。

    pair/triple 是无条件选择事件的概率：每样本一次抽签，分别保留最多
    一/两台真实源。相机不足时保留现有源，不复制图片，也不增加证据。
    generator 可为 CPU Generator，省略时使用可被 checkpoint 恢复的
    PyTorch CPU 随机流；训练入口应像原 runner 一样按 rank 设随机种子。
    """
    probabilities = (grayscale_probability, pair_probability, triple_probability)
    if any(not 0 <= value <= 1 for value in probabilities):
        raise ValueError("增强概率必须位于 [0, 1]")
    if pair_probability + triple_probability > 1:
        raise ValueError("双目与三目抽签概率之和不能大于 1")
    rgb = inputs["rgb"]
    batch, views = rgb.shape[:2]
    original_mask = inputs["view_mask"].bool()
    if rgb.ndim != 5 or original_mask.shape != (batch, views):
        raise ValueError("需要已 collate 的 rgb[B,N,3,H,W] 和 view_mask[B,N]")

    gray = torch.rand(batch, generator=generator) < grayscale_probability
    draw = torch.rand(batch, generator=generator)
    retained = original_mask.detach().cpu().clone()
    for sample in range(batch):
        sources = retained[sample, 1:].nonzero().flatten() + 1
        keep = 1 if draw[sample] < pair_probability else 2
        if draw[sample] >= pair_probability + triple_probability:
            continue
        if len(sources) > keep:
            shuffled = sources[torch.randperm(len(sources), generator=generator)]
            retained[sample, shuffled[keep:]] = False
    retained = retained.to(rgb.device)
    gray = gray.to(rgb.device)
    result = dict(inputs)
    result["view_mask"] = retained

    for name in ("rgb", "matching_rgb"):
        if name not in inputs:
            continue
        image = inputs[name]
        weights = image.new_tensor((0.299, 0.587, 0.114))[None, None, :, None, None]
        luminance = (image * weights).sum(2, keepdim=True)
        image = torch.where(gray[:, None, None, None, None], luminance, image)
        result[name] = image * retained[:, :, None, None, None]
    for name in ("image_mask", "matching_image_mask", "camera_valid"):
        if name in inputs:
            value = inputs[name]
            mask = retained.reshape(batch, views, *([1] * (value.ndim - 2)))
            result[name] = value & mask
    if "image_sizes" in inputs:
        sizes = inputs["image_sizes"]
        result["image_sizes"] = sizes * retained[:, :, None]
    return result, {"grayscale": gray, "retained_views": retained.sum(1)}
