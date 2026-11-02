"""新同步序列逐序列抽样检查左右目深度重投影；不按误差删训练样本。"""
from pathlib import Path

import numpy as np

from expand import read_rows, write_json
from prepare_dynamicreplica import decode_depth


def audit(root):
    root = Path(root)
    reports = []
    for manifest in sorted((root / 'manifests/by_sequence').glob('*.jsonl')):
        rows = read_rows(manifest)
        for index in sorted({0, len(rows) // 2, len(rows) - 1}):
            row = rows[index]
            depths = [decode_depth(root / c['targets']['depth']) for c in row['cameras']]
            for ref in (0, 1):
                a, b = row['cameras'][ref], row['cameras'][1 - ref]
                z, other = depths[ref], depths[1 - ref]
                height, width = z.shape
                v, u = np.mgrid[12:height:24, 12:width:24]
                pixels = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], -1)
                depth = z[v, u].ravel()
                valid = np.isfinite(depth) & (depth > 0) & (depth < 1000)
                points = (pixels[valid] @ np.linalg.inv(a['K']).T) * depth[valid, None]
                transform = np.array(b['T_cam_from_rig']) @ np.linalg.inv(a['T_cam_from_rig'])
                points = points @ transform[:3, :3].T + transform[:3, 3]
                projected = points @ np.array(b['K']).T
                uv = projected[:, :2] / np.maximum(points[:, 2:3], 1e-8)
                nearest = np.rint(uv).astype(int)
                inside = ((points[:, 2] > 0) & (nearest[:, 0] >= 0) & (nearest[:, 0] < width)
                          & (nearest[:, 1] >= 0) & (nearest[:, 1] < height))
                measured = other[nearest[inside, 1], nearest[inside, 0]]
                predicted = points[inside, 2]
                valid = np.isfinite(measured) & (measured > 0) & (measured < 1000)
                relative = np.abs(measured[valid] - predicted[valid]) / predicted[valid]
                if not relative.size:
                    raise ValueError(f'新序列无可核对共同深度：{manifest.name}/{index}')
                reports.append(dict(sequence=row['source_sequence'], frame=row['frame_id'], reference=ref,
                    baseline=float(np.linalg.norm(transform[:3, 3])), comparisons=len(relative),
                    median_relative_error=float(np.median(relative)),
                    agreement_2percent=float((relative < .02).mean())))
    write_json(root / 'geometry_audit.json', dict(records=reports,
        selection='每序列首中末、双向；GT只作处理后核查，不选择/剔除训练组'))
    return len(reports)
