"""仅经服务器网卡下载官方 BlendedMVS 分卷，再合并并逐成员校验解压。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from download_public_dataset import extract, sha256


def check_capacity(root, additional_bytes, minimum_free_gib):
    minimum = int(minimum_free_gib * 1024 ** 3)
    free = shutil.disk_usage(root).free
    if free - max(0, additional_bytes) < minimum:
        raise RuntimeError(f"保留空间不足：当前{free}bytes，下一阶段最多增加{additional_bytes}bytes，须保留{minimum}bytes")


def download(entry, root):
    path = root / "archives" / entry["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = root / "logs" / f"{entry['name']}.download.json"
    if path.exists():
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"已有归档大小不符：{path}")
        record = json.loads(evidence.read_text()) if evidence.exists() else {}
        digest = sha256(path)
        if record.get("sha256") and digest != record["sha256"]:
            raise ValueError(f"已有归档与下载记录哈希不一致：{path}")
        return {**record, "name": entry["name"], "sha256": digest, "reused": True}
    temporary = path.with_name(path.name + ".part")
    command = ["/usr/bin/curl", "-q", "--proxy", "", "--noproxy", "*", "--interface", "eno5",
               "--proto", "=https", "--proto-redir", "=https", "--cacert", "/etc/ssl/certs/ca-certificates.crt",
               "--header", "Accept: application/octet-stream", "--fail", "--location",
               "--connect-timeout", "15", "--max-time", "14400", "--retry", "3", "--retry-delay", "5",
               "--speed-limit", "8192", "--speed-time", "90", "--continue-at", "-",
               "--silent", "--show-error", "--output", str(temporary),
               "--write-out", "%{http_code} %{local_ip} %{remote_ip} %{size_download} %{time_total}", entry["url"]]
    print(f"服务器直连下载：{entry['name']}，{entry['bytes']} bytes", flush=True)
    process = subprocess.run(command, text=True, capture_output=True)
    restarted = False
    if process.returncode == 33 and temporary.exists():
        # 个别公开转发节点忽略 Range；仅重建本工具的未完成文件，绝不拼接HTTP200正文。
        print(f"{entry['name']} 镜像未接受断点，安全重启该分卷完整下载", flush=True)
        temporary.unlink()
        restarted = True
        process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode:
        print(f"{entry['name']} 下载失败：{process.returncode} {process.stderr}", flush=True)
        raise RuntimeError(f"直连失败，不回退代理：{process.stdout} {process.stderr}")
    code, local, remote, count, seconds = process.stdout.strip().split()
    if local.startswith("127.") or remote.startswith("127.") or remote == "::1":
        raise RuntimeError("检测到回环代理连接")
    if temporary.stat().st_size != entry["bytes"]:
        raise ValueError(f"下载大小与官方资产清单不符：{temporary}")
    digest = sha256(temporary)
    temporary.replace(path)
    record = {"name": entry["name"], "official_asset_url": entry.get("official_asset_url", entry["url"]),
              "download_url": entry["url"], "transport": entry.get("transport", "official_github_api"),
              "range_restart_needed": restarted,
              "bytes": entry["bytes"],
              "sha256": digest, "source_sha256_available": False, "http_code": int(code),
              "local_ip": local, "remote_ip": remote, "transferred_bytes": int(count),
              "seconds": float(seconds), "proxy_enabled": False, "interface": "eno5"}
    evidence.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def run(config):
    if not os.environ.get("TMUX"):
        raise RuntimeError("下载必须运行于独立 tmux session")
    for key in ("http_proxy", "https_proxy", "all_proxy", "ftp_proxy"):
        os.environ.pop(key, None)
        os.environ.pop(key.upper(), None)
    os.environ.update(NO_PROXY="*", no_proxy="*")
    root = Path(config["root"]).resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    archive_bytes = sum(entry["bytes"] for entry in config["files"])
    last = next(entry for entry in config["files"] if entry["name"] == "BlendedMVS.zip")
    initial_bytes = 0 if (root / "archives" / last["name"]).exists() else last["bytes"]
    check_capacity(root, initial_bytes, config.get("minimum_free_gib", 300))
    records = [download(last, root)]
    # Python zipfile 拒绝跨盘 ZIP64；zipinfo 可只读末卷的完整中央目录。
    listing = subprocess.run(["zipinfo", "-l", str(root / "archives" / last["name"])],
                             check=True, text=True, capture_output=True).stdout
    members = [line.split(maxsplit=9) for line in listing.splitlines()
               if line.startswith(("-", "d")) and len(line.split(maxsplit=9)) == 10]
    unpacked = sum(int(member[3]) for member in members)
    depth_bytes = sum(int(member[3]) for member in members if member[9].endswith(".pfm"))
    (root / "logs" / "official_archive_inventory.txt").write_text(listing)
    # 合并后删除本工具产生的 joined.zip，转换深度只增加近似同尺寸的一份NPY。
    peak = archive_bytes + unpacked + max(archive_bytes, depth_bytes) + 1024 ** 3
    budget = int(config.get("maximum_footprint_gib", 140) * 1024 ** 3)
    storage = {"archive_bytes": archive_bytes, "unpacked_bytes": unpacked,
               "depth_pfm_bytes": depth_bytes, "estimated_peak_bytes": peak,
               "budget_bytes": budget, "members": len(members)}
    (root / "logs" / "storage_plan.json").write_text(json.dumps(storage, indent=2) + "\n")
    print(json.dumps(storage), flush=True)
    if peak > budget:
        raise RuntimeError("官方归档展开后的预计占用超过配置上限，停止其余下载")
    if shutil.disk_usage(root).free < peak - last["bytes"]:
        raise RuntimeError("数据盘剩余空间不足以容纳预计峰值")
    others = [entry for entry in config["files"] if entry != last]
    remaining = 0
    for entry in others:
        final = root / "archives" / entry["name"]
        partial = final.with_name(final.name + ".part")
        existing = final if final.exists() else partial
        remaining += max(0, entry["bytes"] - (existing.stat().st_size if existing.exists() else 0))
    check_capacity(root, remaining, config.get("minimum_free_gib", 300))
    with ThreadPoolExecutor(max_workers=int(config.get("download_workers", 3))) as pool:
        for record in pool.map(lambda entry: download(entry, root), others):
            records.append(record)
            print(f"下载及本地 SHA256 完成：{record['name']}", flush=True)
    report = {"source": "YoYo000/BlendedMVS official v1.0.0 release assets",
              "transport_note": "各文件记录实际下载地址；公开转发镜像不等于官方CDN，且始终禁用本地代理。",
              "tmux": True, "proxy_enabled": False, "interface": "eno5", "files": records,
              "storage": storage, "source_sha256_available": False,
              "integrity_protocol": "官方资产字节数、本地SHA256记录、合并ZIP逐成员CRC校验"}
    report_path = root / "download_evidence.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    marker = root / "logs" / "BlendedMVS.joined.zip.extracted.json"
    joined = root / "archives" / "BlendedMVS.joined.zip"
    if not marker.exists():
        if not joined.exists():
            check_capacity(root, archive_bytes, config.get("minimum_free_gib", 300))
            with (root / "logs" / "zip_merge.log").open("w") as stream:
                subprocess.run(["zip", "-s", "0", "BlendedMVS.zip", "--out", "BlendedMVS.joined.zip"],
                               cwd=root / "archives", stdout=stream, stderr=subprocess.STDOUT, check=True)
        check_capacity(root, unpacked, config.get("minimum_free_gib", 300))
        result = extract({"output": "archives/BlendedMVS.joined.zip", "extract_to": "raw"}, root)
    else:
        result = json.loads(marker.read_text())
    report["extraction"] = result
    report["completed"] = True
    report["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if joined.exists():
        joined.unlink()  # 可由原分卷无损重建，释放冗余副本，不删除原始归档。
    print("官方基础集下载与CRC解压完成，原始分卷保留", flush=True)
    if config.get("prepare_config"):
        check_capacity(root, depth_bytes + 512 * 1024 ** 2, config.get("minimum_free_gib", 300))
        with (root / "logs" / "prepare.log").open("a") as stream:
            subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_blendedmvs.py")),
                            "--config", config["prepare_config"]],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("全量转换完成：processed/preparation_summary.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
