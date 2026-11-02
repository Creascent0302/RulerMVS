"""在 tmux 中经服务器网卡直连下载、校验、解压已指定的数据子集。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import zipfile


def require_space(root, additional_bytes=0):
    """保留用户要求的300GiB，空间不足时停止，不删除已有数据。"""
    reserve = 300 * 1024 ** 3
    if shutil.disk_usage(root).free < reserve + additional_bytes:
        raise RuntimeError("继续写入将使数据盘剩余不足300GiB，已停止并保留已有文件")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry, root, interface):
    path = root / entry["output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get("sha256")
    record = {"url": entry["url"], "path": str(path), "expected_bytes": entry["bytes"]}
    if path.exists():
        if path.stat().st_size != entry["bytes"] or (expected and sha256(path) != expected):
            raise ValueError(f"已有文件不匹配，保留原文件并停止：{path}")
        record["already_downloaded"] = True
    else:
        temporary = path.with_name(path.name + ".part")
        written = temporary.stat().st_size if temporary.exists() else 0
        require_space(root, max(0, entry["bytes"] - written))
        # -q必须是第一个curl参数，以忽略用户配置；所有重定向也绑定相同网卡。
        command = [
            "/usr/bin/curl", "-q", "--proxy", "", "--noproxy", "*", "--interface", interface,
            "--proto", "=https", "--proto-redir", "=https",
            "--cacert", "/etc/ssl/certs/ca-certificates.crt", "--fail", "--location",
            "--connect-timeout", "15", "--max-time", "14400", "--retry", "3", "--retry-delay", "5",
            "--speed-limit", "8192", "--speed-time", "90", "--continue-at", "-",
            "--silent", "--show-error", "--output", str(temporary),
            "--write-out", "%{http_code} %{local_ip} %{remote_ip} %{size_download} %{time_total} %{speed_download}",
            entry["url"],
        ]
        print(f"下载：{entry['output']} ({entry['bytes']} bytes)", flush=True)
        process = subprocess.run(command, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError(f"服务器直连失败；不回退代理：{process.stdout} {process.stderr}")
        code, local, remote, count, elapsed, speed = process.stdout.strip().split()
        record.update(http_code=int(code), local_ip=local, remote_ip=remote,
                      transferred_bytes=int(count), seconds=float(elapsed), bytes_per_second=float(speed))
        if local.startswith("127.") or remote.startswith("127.") or remote == "::1":
            raise RuntimeError("发现回环连接，停止并保留部分文件")
        if temporary.stat().st_size != entry["bytes"]:
            raise ValueError(f"下载大小错误：{path}")
        if expected and sha256(temporary) != expected:
            raise ValueError(f"下载SHA256错误：{path}")
        temporary.replace(path)
    record["sha256"] = sha256(path)
    record["source_sha256_matched"] = bool(expected and record["sha256"] == expected)
    return record


def extract(entry, root):
    if not entry.get("extract_to"):
        return {}
    archive = root / entry["output"]
    destination = (root / entry["extract_to"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    marker = root / "logs" / (archive.name + ".extracted.json")
    if marker.exists():
        return json.loads(marker.read_text())

    def safe_path(name):
        path = (destination / name).resolve()
        if path != destination and destination not in path.parents:
            raise ValueError(f"归档路径超出解压目录：{name}")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as reader:
            entries = reader.infolist()
            for entry_info in entries:
                safe_path(entry_info.filename)
                if (entry_info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("不接受归档内软链接")
            require_space(root, sum(info.file_size for info in entries))
            # extractall会读取全部成员并验证CRC，不额外解压一遍。
            reader.extractall(destination)
            result = {"members": len(entries), "unpacked_bytes": sum(info.file_size for info in entries),
                      "zip_crc": "passed"}
    else:
        with tarfile.open(archive) as reader:
            entries = reader.getmembers()
            for entry_info in entries:
                safe_path(entry_info.name)
                if not (entry_info.isfile() or entry_info.isdir()):
                    raise ValueError(f"不接受归档内链接或设备：{entry_info.name}")
            require_space(root, sum(info.size for info in entries))
            reader.extractall(destination)
            result = {"members": len(entries), "unpacked_bytes": sum(info.size for info in entries)}
    marker.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(plan):
    if not os.environ.get("TMUX"):
        raise RuntimeError("数据下载必须在tmux中运行")
    for name in ("http_proxy", "https_proxy", "all_proxy", "ftp_proxy"):
        os.environ.pop(name, None)
        os.environ.pop(name.upper(), None)
    os.environ.update(NO_PROXY="*", no_proxy="*")
    root = Path(plan["root"]).resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    report_path = root / "download_evidence.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    previous_files = {item["url"]: item for item in previous.get("files", [])}
    report = {"plan": plan, "started_utc": previous.get("started_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
              "tmux": True, "proxy_enabled": False, "interface": "eno5", "files": []}
    for entry in plan["files"]:
        record = download(entry, root, "eno5")
        if record.get("already_downloaded"):
            record = {**previous_files.get(entry["url"], {}), **record}
        report["files"].append(record)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        record["extraction"] = extract(entry, root)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    report["completed"] = True
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("下载、校验和解压全部完成", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
