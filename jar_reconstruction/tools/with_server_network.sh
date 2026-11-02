#!/bin/sh
# 仅清除本命令及其子进程的代理；不修改当前终端、SSH、tmux 或用户配置。
# 新交互 shell 请用 bash --noprofile --norc，防止 .bashrc 再次设置代理。
set -eu

if [ "$#" -eq 0 ]; then
    echo "用法：$0 命令 [参数...]" >&2
    exit 2
fi

unset http_proxy https_proxy all_proxy ftp_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY
export no_proxy='*'
export NO_PROXY='*'

exec "$@"
