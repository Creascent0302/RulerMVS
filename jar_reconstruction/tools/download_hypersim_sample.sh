#!/bin/bash
# 仅在 tmux 中，经服务器 eno5 直连下载一个完整 Hypersim 场景。
set -euo pipefail

unset http_proxy https_proxy all_proxy ftp_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY
export NO_PROXY='*' no_proxy='*'

test -n "${TMUX:-}" || { echo '请在 tmux 内启动本脚本。' >&2; exit 2; }
root=/data/liuzonghao_data/dataset/Hypersim_sample
mkdir -p "$root/archives" "$root/logs"
exec > >(tee -a "$root/logs/download.log") 2>&1
trap 'echo "下载脚本退出码：$?；时间：$(date -Is)"' EXIT

archive="$root/archives/ai_011_010.zip"
checksum=31d8a62a155bb2ea48923fbf1b3945394b764191dbc22df542d71a215f81ca75
echo "开始：$(date -Is)"
echo '代理禁用；强制 eno5；不允许回退到 SSH 代理。'
ip route get 39.99.133.195
if [ ! -f "$archive" ]; then
    /usr/bin/curl -q --proxy '' --noproxy '*' --interface eno5 \
        --proto '=https' --proto-redir '=https' \
        --cacert /etc/ssl/certs/ca-certificates.crt \
        --fail --location --continue-at - --connect-timeout 10 \
        --max-time 1800 --retry 3 --retry-delay 3 \
        --speed-limit 1024 --speed-time 60 --silent --show-error \
        --output "$archive.part" \
        --write-out 'HTTP=%{http_code} local=%{local_ip} remote=%{remote_ip} bytes=%{size_download} seconds=%{time_total} speed_Bps=%{speed_download}\n' \
        'https://modelscope.cn/datasets/OpenDataLab/Hypersim/resolve/master/raw/ai_011_010.zip'
    printf '%s  %s\n' "$checksum" "$archive.part" | sha256sum --check -
    test "$(stat -c %s "$archive.part")" = 112248512
    mv "$archive.part" "$archive"
fi
printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check -
echo "完成：$(date -Is)"
