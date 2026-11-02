"""服务器 eno5 直连；不读取代理、curlrc 或用户认证文件。"""

import collections
import io
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


ROOT = Path('/data/liuzonghao_data/dataset/public_expansion_20261102')
RESOLVED = {}


def reserve(upcoming=0):
    if shutil.disk_usage(ROOT).free < 200 * 2**30 + upcoming:
        raise RuntimeError('停止写入：必须保留至少200GiB，不自动删除旧数据')


def fetch(url, destination, byte_range=None, max_bytes=None, timeout=1800):
    """所有跳转均绑定 eno5；失败绝不回退代理。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reserve(max_bytes or 0)
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    env.update(NO_PROXY='*', no_proxy='*')
    headers = destination.with_suffix(destination.suffix + '.headers')
    args = ['/usr/bin/curl', '-q', '--proxy', '', '--noproxy', '*',
            '--interface', 'eno5', '--proto', '=https', '--proto-redir', '=https',
            '--cacert', '/etc/ssl/certs/ca-certificates.crt',
            '--fail', '--location', '--silent', '--show-error',
            '--connect-timeout', '15', '--max-time', str(timeout),
            '--retry', '2', '--retry-delay', '3',
            '--dump-header', str(headers), '--output', str(destination),
            '--write-out', '%{http_code} %{local_ip} %{remote_ip} %{size_download} %{time_total} %{url_effective}']
    if byte_range is not None:
        args += ['--range', f'{byte_range[0]}-{byte_range[1]}']
    if max_bytes:
        args += ['--max-filesize', str(max_bytes)]
    request_url = RESOLVED.get(url, url)
    process = subprocess.run(args + [request_url], env=env, capture_output=True, text=True)
    if process.returncode and request_url != url:
        RESOLVED.pop(url, None)
        process = subprocess.run(args + [url], env=env, capture_output=True, text=True)
    if process.returncode and 'hf-mirror.com/' in url and '429' in process.stderr:
        request_url = url.replace('hf-mirror.com/', 'huggingface.co/', 1)
        process = subprocess.run(args + [request_url], env=env, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f'服务器直连失败({process.returncode})，无代理回退：{url}: {process.stderr[-400:]}')
    code, local, remote, size, elapsed, effective = process.stdout.split()
    RESOLVED[url] = effective
    if local != '172.16.2.4' or ipaddress.ip_address(remote).is_loopback:
        raise RuntimeError(f'服务器连接地址不符：{local} -> {remote}')
    if byte_range is not None:
        ranges = re.findall(r'content-range:\s*bytes (\d+)-(\d+)/(\d+)', headers.read_text(), re.I)
        if code != '206' or not ranges or tuple(map(int, ranges[-1][:2])) != byte_range:
            raise RuntimeError('服务器没有严格返回所请求的ZIP范围，拒绝使用')
        if int(size) != byte_range[1] - byte_range[0] + 1:
            raise RuntimeError('范围响应长度错误')
    record = dict(url=url, http=int(code), local_ip=local, remote_ip=remote,
                  bytes=int(size), seconds=float(elapsed), range=byte_range,
                  proxy=False, interface='eno5', unix_time=time.time())
    with (ROOT / 'network.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    headers.unlink(missing_ok=True)
    return record


class RemoteZipReader(io.RawIOBase):
    """有界缓存的只读HTTP ZIP接口；zipfile负责ZIP结构和成员CRC。"""

    def __init__(self, url, size, block_size=8 * 2**20):
        self.url = url
        self.size = size
        self.position = 0
        self.block_size = block_size
        self.cache = collections.OrderedDict()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = (0, self.position, self.size)[whence]
        self.position = base + offset
        if self.position < 0:
            raise ValueError('negative seek')
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            index = self.position // self.block_size
            start = index * self.block_size
            stop = min(start + self.block_size, self.size)
            if index not in self.cache:
                with tempfile.TemporaryDirectory(dir=ROOT / 'temporary') as tmp:
                    path = Path(tmp) / 'range.bin'
                    fetch(self.url, path, (start, stop - 1), stop - start, timeout=180)
                    self.cache[index] = path.read_bytes()
                while len(self.cache) > 3:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(index)
            length = min(end, stop) - self.position
            chunks.append(self.cache[index][self.position - start:self.position - start + length])
            self.position += length
        return b''.join(chunks)


def extract_members(archive, destination, members):
    destination = Path(destination).resolve()
    reserve(sum(m.file_size for m in members if not (destination / m.filename).exists()))
    for info in members:
        target = (destination / info.filename).resolve()
        if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f'不安全ZIP路径：{info.filename}')
        if info.is_dir():
            continue
        if target.exists():
            if target.stat().st_size != info.file_size:
                raise RuntimeError(f'已有文件长度不符：{target}')
            continue
        reserve(info.file_size)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.part')
        with archive.open(info) as source, temporary.open('wb') as stream:
            shutil.copyfileobj(source, stream, 2**20)
        temporary.replace(target)
