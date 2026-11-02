#!/usr/bin/env bash
# 仅服务器网卡直连；不读取curl配置，不使用任何代理或SSH转发。
set -euo pipefail
cd /home/liuzonghao/3Dscene/jar_reconstruction
destination=/data/liuzonghao_data/dataset/ETH3D_two_view_paper
mkdir -p "$destination/archives" "$destination/raw"
tools/with_server_network.sh official_methods/.venv/bin/python - <<'PY'
import json
import shutil
from pathlib import Path
from jar.pretraining.comparison import digest

root = Path('/data/liuzonghao_data/dataset/ETH3D_two_view_paper')
if shutil.disk_usage(root).free < 205 * 2**30:
    raise RuntimeError('外域下载后必须保留200GiB空间')
plan = {
    'purpose': '外部真实双目深度确认；不训练，不按GT选场景或范围，不冒充RMVD或官方四目榜单',
    'source': 'https://www.eth3d.net/datasets',
    'archives': ['two_view_training.7z', 'two_view_training_gt.7z'],
    'selection': '官方全部16个有GT双目对，全部作为本项目外部测试',
    'image_size': [256, 448], 'near': 0.1, 'far': 100.0, 'depth_scale': 1.0,
    'range_policy': '下载标签前固定所有方法同0.1–100m；原生MVS细化允许自己的范围更新。',
    'weights_sha256': {
        name: digest(path) for name, path in {
            'v5': 'runs/active/rulermvs_v5_20261031/public/best.pt',
            'v6': 'runs/active/rulermvs_v6_20261101/full_training/public/best.pt',
            'mvsanywhere': 'runs/carla_research_20261011/cache/mvsanywhere_hero.ckpt',
        }.items()
    },
}
path = root / 'pre_download_plan.json'
if path.exists() and json.loads(path.read_text()) != plan:
    raise RuntimeError('已有预登记计划不同，拒绝覆盖')
path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
PY
for filename in two_view_training.7z two_view_training_gt.7z; do
    if [ ! -f "$destination/archives/$filename" ]; then
        tools/with_server_network.sh curl --disable --proxy '' --noproxy '*' \
            --fail --location --connect-timeout 20 --max-time 600 --retry 2 \
            --write-out '\nremote_ip=%{remote_ip} local_ip=%{local_ip} bytes=%{size_download}\n' \
            --output "$destination/archives/$filename.part" "https://www.eth3d.net/data/$filename"
        mv "$destination/archives/$filename.part" "$destination/archives/$filename"
    fi
    bsdtar -xf "$destination/archives/$filename" -C "$destination/raw"
done
if [ ! -e /home/liuzonghao/3Dscene/data/ETH3D_two_view_paper ]; then
    ln -s "$destination" /home/liuzonghao/3Dscene/data/ETH3D_two_view_paper
fi
sha256sum "$destination/archives/"*.7z > "$destination/archive_sha256.txt"
