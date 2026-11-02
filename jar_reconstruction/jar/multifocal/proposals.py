"""细节评分加分区覆盖；离散选点不充当表面证据。"""

import torch
from torch import nn
from torch.nn import functional as F
from jar.geometry import pixel_grid, sample_maps
from jar.query_selection import select_query_candidates


class TeleBirthScorer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1), nn.SiLU(), nn.Conv2d(channels // 2, 1, 1)
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, pyramid):
        b, n, c, h, w = pyramid[4].shape
        return self.score(pyramid[4].reshape(b * n, c, h, w)).reshape(b, n, h, w)


def tele_births(inputs, b, count, logits, old_refs, old_uv, *, exploration=0.25, cell=16):
    rgb = inputs["rgb"][b]
    n, _, h, w = rgb.shape
    roles = inputs["camera_roles"][b]
    available = inputs["view_mask"][b] & (roles == 1)
    refs_out = []
    uv_out = []
    kind_out = []
    cameras = available.nonzero().flatten().tolist()
    for j, camera in enumerate(cameras):
        quota = count // max(len(cameras), 1) + (j < count % max(len(cameras), 1))
        if not quota:
            continue
        mask = inputs["image_mask"][b, camera].clone()
        mask[:2] = False
        mask[-2:] = False
        mask[:, :2] = False
        mask[:, -2:] = False
        if old_refs is not None:
            old = old_uv[old_refs == camera].round().long()
            if len(old):
                mask[old[:, 1].clamp(0, h - 1), old[:, 0].clamp(0, w - 1)] = False
        coords = pixel_grid(h, w, rgb.device).reshape(-1, 2)
        score = sample_maps(logits[b, camera][None, None], coords[None], 4)[0, :, 0].sigmoid()
        texture = inputs["_texture"][b, camera, 0].flatten()
        score = score * 0.25 + 0.75 * texture / (texture.max().clamp_min(1e-7))
        score = score.masked_fill(~mask.flatten(), -float("inf"))
        # 一格优先一个候选，以局部排名轮次优先、分数次优先抑制纹理大块霸占预算。
        padded = F.pad(score.reshape(h, w), (0, (-w) % cell, 0, (-h) % cell), value=-float("inf"))
        windows = padded.unfold(0, cell, cell).unfold(1, cell, cell).reshape(-1, cell * cell)
        values, indices = windows.sort(dim=-1, descending=True, stable=True)
        cells = (
            pixel_grid(padded.shape[0] // cell, padded.shape[1] // cell, rgb.device, stride=cell)
            .reshape(-1, 2)
            .long()
        )
        xs = cells[:, 0, None] + indices % cell
        ys = cells[:, 1, None] + indices // cell
        # 只遍历覆盖轮次，不逐节点递归。
        ranked = []
        for rank in range(min(cell * cell, quota)):
            order = values[:, rank].argsort(descending=True, stable=True)
            good = (
                torch.isfinite(values[order, rank]) & (xs[order, rank] < w) & (ys[order, rank] < h)
            )
            ranked.append((ys[order[good], rank] * w + xs[order[good], rank]))
            if sum(len(v) for v in ranked) >= quota * 2:
                break
        ranked = (
            torch.cat(ranked) if ranked else torch.empty(0, device=rgb.device, dtype=torch.long)
        )
        valid = mask.flatten().nonzero().flatten()
        e = min(len(valid), int(round(quota * exploration)))
        explore = (
            valid[torch.linspace(0, max(len(valid) - 1, 0), e, device=rgb.device).long()]
            if e
            else valid[:0]
        )
        chosen = torch.zeros(h * w, device=rgb.device, dtype=torch.bool)
        chosen[explore] = True
        detail = ranked[~chosen[ranked]][: max(0, quota - len(explore))]
        keys = torch.cat((explore, detail))
        uv = coords[keys]
        refs_out.append(torch.full((len(keys),), camera, device=rgb.device, dtype=torch.long))
        uv_out.append(uv)
        kind_out.append(torch.cat((torch.full_like(explore, 3), torch.full_like(detail, 2))))
    if not refs_out:
        return (
            torch.empty(0, device=rgb.device, dtype=torch.long),
            rgb.new_empty(0, 2),
            torch.empty(0, device=rgb.device, dtype=torch.long),
        )
    return torch.cat(refs_out), torch.cat(uv_out), torch.cat(kind_out)
