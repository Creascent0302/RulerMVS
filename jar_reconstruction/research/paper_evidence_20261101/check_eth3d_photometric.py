"""独立图像对应检查：GT视差正确方向与零视差/反方向配对，结果不用于调参。"""

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paper_evidence_20261101.prepare_eth3d import DATA, OUTPUT, pfm
from jar.pretraining.comparison import digest, write_json


def main():
    rows = []
    for scene in sorted((DATA / 'raw').iterdir()):
        left = cv2.imread(str(scene / 'im0.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        right = cv2.imread(str(scene / 'im1.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        disparity = pfm(scene / 'disp0GT.pfm')
        nonocc = cv2.imread(str(scene / 'mask0nocc.png'), cv2.IMREAD_GRAYSCALE) == 255
        y, x = np.indices(left.shape, dtype=np.float32)
        valid = nonocc & np.isfinite(disparity)
        valid &= (x - disparity >= 0) & (x + disparity < left.shape[1] - 1)
        safe_disparity = np.where(np.isfinite(disparity), disparity, 0)
        errors = {}
        for name, offset in (('correct', -safe_disparity), ('zero', np.zeros_like(x)), ('wrong_sign', safe_disparity)):
            warped = cv2.remap(right, x + offset, y, cv2.INTER_LINEAR)
            errors[name] = float(np.abs(left[valid] - warped[valid]).mean())
        rows.append({'pair': scene.name, 'pixels': int(valid.sum()), 'RGB_L1': errors})
    result = {'completed': True, 'source_sha256': digest(__file__), 'pairs': rows,
              'wins_against_zero': sum(x['RGB_L1']['correct'] < x['RGB_L1']['zero'] for x in rows),
              'wins_against_wrong_sign': sum(x['RGB_L1']['correct'] < x['RGB_L1']['wrong_sign'] for x in rows),
              'notes': ['仅验证数据转换方向/图像对应，GT不进入模型；不是模型精度实验。',
                        '三种偏移使用相同非遮挡且都在图内的像素，不依据误差删场景。']}
    write_json(OUTPUT / 'photometric_check.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}))


if __name__ == '__main__':
    main()
