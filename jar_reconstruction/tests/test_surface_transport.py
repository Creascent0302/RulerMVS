"""表面传递的几何、子像素选面和训练接口契约。

这些测试使用解析射线平面交点作为几何真值；不以随机独立图像证明重建精度。
"""

import pytest
import torch
from torch.nn import functional as F

from jar.pretraining.surface_decoder import SurfaceDecoder, SurfaceReadout, evaluate_planes


def _plane_depth(pixels, intrinsic, normal, distance):
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    rays = homogeneous @ torch.linalg.inv(intrinsic).T
    return distance / (rays * normal).sum(-1)


def test_inverse_depth_plane_transport_uses_actual_pixel_anchors():
    """九个不同锚点的同一倾斜平面，在同一子像素处应给出同一深度。"""
    dtype = torch.float64
    intrinsic = torch.tensor([[720., 0., 201.3], [0., 640., 151.7], [0., 0., 1.]], dtype=dtype)
    normal = torch.tensor([.7, -.4, 1.], dtype=dtype)
    distance = torch.tensor(7., dtype=dtype)
    owner = torch.tensor([8., 12.], dtype=dtype)
    dy, dx = torch.meshgrid(torch.arange(-1, 2, dtype=dtype),
                            torch.arange(-1, 2, dtype=dtype), indexing='ij')
    anchors = owner + 4 * torch.stack((dx, dy), dim=-1).reshape(9, 2)
    sy, sx = torch.meshgrid(torch.arange(4, dtype=dtype),
                            torch.arange(4, dtype=dtype), indexing='ij')
    pixels = owner + torch.stack((sx, sy), dim=-1).reshape(16, 2)
    anchor_z = _plane_depth(anchors, intrinsic, normal, distance)
    truth = _plane_depth(pixels, intrinsic, normal, distance)
    rho_slope = torch.stack((normal[0] / intrinsic[0, 0],
                             normal[1] / intrinsic[1, 1])) / distance
    relative_slope = 4 * rho_slope[None] * anchor_z[:, None]
    delta = (pixels[None] - anchors[:, None]) / 4
    transported = evaluate_planes(
        anchor_z[None, :, None, None, None, None],
        relative_slope.T[None, :, :, None, None, None, None],
        delta.permute(2, 0, 1).reshape(2, 9, 4, 4, 1, 1),
    )[0, :, :, :, 0, 0].reshape(9, 16)
    torch.testing.assert_close(transported, truth[None].expand(9, -1), rtol=1e-12, atol=1e-12)
    # 现有步幅4格点从0开始，子像素偏移为0/1/2/3；没有隐含半像素移动。
    assert torch.equal(pixels[0], owner)
    assert torch.equal(pixels[-1], owner + 3)
    assert float(delta.min()) == -1.
    assert float(delta.max()) == 1.75


def _inputs(views=4, batch=2, height=3, width=5, requires_grad=False):
    torch.manual_seed(2791)
    features = torch.randn(batch, 64, 12, height, width, requires_grad=requires_grad)
    guide = torch.randn(batch, 64, 2 * height, 2 * width, requires_grad=requires_grad)
    scores = torch.randn(batch, 12, height, width)
    centers = torch.tensor([3., 7., 13., 23.])[None, :, None, None, None]
    offsets = torch.tensor([.9, 1., 1.1])[None, None, :, None, None]
    depth = (centers * offsets).flatten(1, 2).expand(batch, -1, height, width).clone()
    valid = torch.ones_like(depth, dtype=torch.bool)
    source_valid = valid[:, None].expand(-1, views, -1, -1, -1).clone()
    source_valid[:, 0] = False
    support = source_valid.sum(1)
    index = scores.argmax(1, keepdim=True)
    fine = {
        'scores': scores, 'candidate_depth': depth, 'sampling_depth': depth,
        'valid': valid, 'source_valid': source_valid, 'support': support,
        'depth': depth.gather(1, index), 'selected_index': index,
        'ref_valid': valid.any(1, keepdim=True),
        'probability': scores.softmax(1), 'confidence': scores.softmax(1).gather(1, index),
    }
    mask = torch.ones(batch, 1, 4 * height, 4 * width, dtype=torch.bool)
    return fine, features, guide, mask


@pytest.mark.parametrize('views', [2, 4])
def test_shapes_and_hard_surface_selection(views):
    model = SurfaceReadout(channels=64, hidden=16, key_dim=8, slope_limit=.25)
    fine, features, guide, mask = _inputs(views)
    output = model(fine, features, guide, mask, .5, 80.)
    assert output['depth'].shape == (2, 1, 12, 20)
    render = output['render']
    for name in ('plane_depth', 'logits', 'valid'):
        assert render[name].shape == (2, 12, 12, 20)
    assert render['valid'].dtype == torch.bool
    assert torch.isfinite(output['depth']).all()
    assert torch.isfinite(render['plane_depth']).all()
    # 每个有效输出必须等于某一个合法面片，不能是前景与背景的平均距离。
    error = (render['plane_depth'] - output['depth']).abs()
    error = error.masked_fill(~render['valid'], torch.inf)
    assert torch.equal(error.amin(1, keepdim=True)[mask], torch.zeros_like(output['depth'][mask]))


def test_zero_geometry_does_not_average_foreground_and_background():
    model = SurfaceReadout(channels=64, hidden=16, key_dim=8, slope_limit=.25)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    fine, features, guide, mask = _inputs(batch=1)
    depths = torch.tensor([3., 7., 13., 23.]).repeat_interleave(3)
    fine['candidate_depth'] = depths[None, :, None, None].expand_as(fine['candidate_depth']).clone()
    fine['scores'].fill_(-2)
    for group, score in enumerate((4., 3., 2., 1.)):
        fine['scores'][:, 3 * group + 1] = score
    fine['selected_index'] = fine['scores'].argmax(1, keepdim=True)
    fine['probability'] = fine['scores'].softmax(1)
    fine['depth'] = fine['candidate_depth'].gather(1, fine['selected_index'])
    output = model(fine, features, guide, mask, .5, 80.)
    assert torch.isin(output['depth'][mask], depths).all()
    # 中央块九个邻居全合法，三个额外组应保留7/13/23三个可选择表面。
    center = output['render']['plane_depth'][0, :, 4, 8]
    for value in (7., 13., 23.):
        assert (center == value).any()


def test_full_resolution_plane_offsets_preserve_subpixel_phase():
    """受控非零斜率检验真实模块的面片供给、横纵顺序及4×4展开。"""
    model = SurfaceReadout(channels=64, hidden=16, key_dim=8, slope_limit=.25)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.patch[-1].bias[1] = torch.atanh(torch.tensor(.4))
        model.patch[-1].bias[2] = torch.atanh(torch.tensor(-.2))
    fine, features, guide, mask = _inputs(batch=1)
    fine['candidate_depth'].fill_(8.)
    fine['depth'].fill_(8.)
    output = model(fine, features, guide, mask, .5, 80.)
    y, x = torch.meshgrid(torch.arange(12), torch.arange(20), indexing='ij')
    subpixel = torch.stack((x % 4, y % 4)).float() / 4
    anchors = [(dx, dy) for dy in (-1., 0., 1.) for dx in (-1., 0., 1.)] + [(0., 0.)] * 3
    for index, (dx, dy) in enumerate(anchors):
        truth = 8. / (1 + .1 * (subpixel[0] - dx) - .05 * (subpixel[1] - dy))
        valid = output['render']['valid'][0, index]
        torch.testing.assert_close(output['render']['plane_depth'][0, index][valid],
                                   truth[valid], rtol=1e-6, atol=1e-6)


def test_no_source_and_no_valid_reference_are_finite():
    model = SurfaceReadout(channels=64, hidden=16, key_dim=8, slope_limit=.25)
    fine, features, guide, mask = _inputs(batch=1)
    fine['source_valid'].zero_()
    fine['support'].zero_()
    no_source = model(fine, features, guide, mask, .5, 80.)
    assert torch.isfinite(no_source['depth']).all()
    assert (no_source['depth'][mask] > 0).all()
    # 没有源视图不等于删掉参考图的深度输出。
    fine['valid'].zero_()
    fine['ref_valid'].zero_()
    no_candidate = model(fine, features, guide, mask, .5, 80.)
    assert torch.isfinite(no_candidate['depth']).all()
    assert not no_candidate['render']['valid'].any()
    mask.zero_()
    no_image = model(fine, features, guide, mask, .5, 80.)
    assert torch.equal(no_image['depth'], torch.zeros_like(no_image['depth']))


def test_depth_unit_covariance():
    model = SurfaceReadout(channels=64, hidden=16, key_dim=8, slope_limit=.25).eval()
    model.return_render = True
    fine, features, guide, mask = _inputs(batch=1)
    original = model(fine, features, guide, mask, .5, 80.)
    multiplier = 17.3
    scaled = {name: value.clone() for name, value in fine.items()}
    for name in ('depth', 'candidate_depth', 'sampling_depth'):
        scaled[name] *= multiplier
    result = model(scaled, features, guide, mask, .5 * multiplier, 80. * multiplier)
    torch.testing.assert_close(result['depth'], original['depth'] * multiplier, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(result['render']['logits'], original['render']['logits'], rtol=2e-5, atol=2e-5)
    assert torch.equal(result['render']['valid'], original['render']['valid'])


def test_decoder_support_is_anchor_metadata_not_surface_verification():
    """真实投影输入通过完整解码器后，支持数不能冒充输出表面验证。"""
    from tests.test_ruler_direct import fixture_inputs, tiny_settings

    inputs, _ = fixture_inputs(views=2)
    matching = inputs.pop('matching_features')[:, :, :8]
    inputs['pyramid'] = {4: matching}
    for stride in (8, 16):
        pooled = F.avg_pool2d(matching.flatten(0, 1), stride // 4)
        inputs['pyramid'][stride] = pooled.unflatten(0, (1, 2))
    inputs['matching_scale'] = torch.tensor([[1., 1.]])
    decoder = SurfaceDecoder(tiny_settings(use_scene=False)).eval()
    with torch.no_grad():
        output = decoder(inputs)
    assert output['depth'].shape == (1, 1, 32, 48)
    assert torch.isfinite(output['depth']).all()
    assert output['support'].max() <= 1
    assert output['unknown'].all() and not output['observed'].any()
    assert '面片锚点' in output['support_semantics']
    assert '未重新验证输出像素和修正深度' in output['support_semantics']


def test_training_and_fast_inference_are_exactly_equivalent():
    model = SurfaceReadout(channels=64, hidden=16, key_dim=8, slope_limit=.25)
    with torch.no_grad():
        torch.nn.init.normal_(model.patch[-1].weight[:3], std=.07)
        model.patch[-1].bias[:3].copy_(torch.tensor([.1, .2, -.3]))
    fine, features, guide, mask = _inputs(batch=2)
    fine['valid'][1, :, 1, 2] = False
    fine['source_valid'][1, :, :, 1, 2] = False
    fine['support'][1, :, 1, 2] = 0
    mask[1, :, 2:4, 3:6] = False
    trained = model.train()(fine, features, guide, mask, .5, 80.)
    fast = model.eval()(fine, features, guide, mask, .5, 80.)
    assert 'render' not in fast
    for name in ('depth', 'valid', 'confidence', 'support', 'unknown', 'observed', 'state'):
        assert torch.equal(trained[name], fast[name]), name
    # 使用全部面片重新gather，独立核对只评价winner的低成本路径。
    render = trained['render']
    expected = render['plane_depth'].gather(1, render['selected_index'])
    assert torch.equal(expected[mask], fast['depth'][mask])


def test_routing_and_geometry_receive_supervision():
    """硬选面之外显式加入路由CE；不能误称深度损失本身训练了argmax。"""
    model = SurfaceReadout(channels=64, hidden=16, key_dim=8, slope_limit=.25)
    fine, features, guide, mask = _inputs(batch=1, requires_grad=True)
    output = model(fine, features, guide, mask, .5, 80.)
    render = output['render']
    render['plane_depth'].retain_grad()
    render['logits'].retain_grad()
    truth = torch.linspace(3.4, 21., 20)[None, None, None].expand_as(output['depth'])
    distance = (render['plane_depth'].detach().log() - truth.log()).abs()
    target = distance.masked_fill(~render['valid'], torch.inf).argmin(1)
    loss = F.cross_entropy(render['logits'], target)
    loss = loss + (output['depth'].log() - truth.log()).abs().mean()
    closest_plane = render['plane_depth'].gather(1, target[:, None])
    loss = loss + .1 * (closest_plane.log() - truth.log()).abs().mean()
    loss.backward()
    assert render['logits'].grad is not None and torch.isfinite(render['logits'].grad).all()
    assert render['logits'].grad.abs().sum() > 0
    assert render['plane_depth'].grad is not None and torch.isfinite(render['plane_depth'].grad).all()
    assert render['plane_depth'].grad.abs().sum() > 0
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert model.patch[-1].weight.grad[:3].abs().sum() > 0
    assert model.query[0].weight.grad.abs().sum() > 0
    assert features.grad is not None and torch.isfinite(features.grad).all()
    # 零初始化几何头可暂时阻断上游梯度；一次受控更新后再查查询支路。
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1e-3)
    model.zero_grad(set_to_none=True)
    features.grad = None
    guide.grad = None
    output = model(fine, features, guide, mask, .5, 80.)
    F.cross_entropy(output['render']['logits'], target).backward()
    assert guide.grad is not None and torch.isfinite(guide.grad).all()
    assert guide.grad.abs().sum() > 0
