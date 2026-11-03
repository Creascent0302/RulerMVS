"""只探测官方Tanks&Temples压缩包大小；直连服务器，不下载大包。"""

import base64
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from prepare import DATA, write, direct_http


def main():
    assert os.environ.get('TMUX')
    base = 'https://lmb.informatik.uni-freiburg.de/data/robustmvd/'
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    env.update(NO_PROXY='*', no_proxy='*')
    results = []
    for name in ['tanks_and_temples_images.zip', 'tanks_and_temples_depth.zip']:
        output = subprocess.run(['/usr/bin/curl', '-q', '--proxy', '', '--noproxy', '*',
            '--interface', 'eno5', '--proto', '=https', '--proto-redir', '=https',
            '--fail', '--location', '--silent', '--show-error', '--head', '--max-time', '45',
            '--write-out', '\n%{http_code} %{local_ip} %{remote_ip}', base + name],
            env=env, text=True, capture_output=True, check=True).stdout
        code, local, remote = output.strip().splitlines()[-1].split()
        assert local == '172.16.2.4' and not remote.startswith('127.')
        length = int(re.findall(r'content-length:\s*(\d+)', output, re.I)[-1])
        results.append(dict(url=base + name, bytes=length, local_ip=local, remote_ip=remote, proxy=False))
    write(DATA / 'tanks_probe.json', results)
    name = 'tanks_and_temples.robustmvd.mvd.pickle'
    response = DATA / 'protocol' / (name + '.api.json')
    direct_http.fetch('https://api.github.com/repos/lmb-freiburg/robustmvd/contents/rmvd/data/sample_lists/' + name,
                      response, max_bytes=3 * 2**20)
    obj = json.loads(response.read_text())
    content = base64.b64decode(obj['content'])
    assert hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == obj['sha']
    response.with_name(name).write_bytes(content)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
