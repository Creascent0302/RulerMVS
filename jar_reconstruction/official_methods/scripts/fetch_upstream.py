#!/usr/bin/env python3
"""固定提交的源码获取/审计；只补缺失文件，绝不覆盖已有文件。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import ssl
import subprocess
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "fast_mvsnet": (
        "svip-lab/FastMVSNet",
        "ccb686dda2717613c67d8a289dfe7b2aeb60e2fd",
        (),
        "fast_mvsnet_pytorch28.patch",
    ),
    "simplerecon": (
        "nianticlabs/simplerecon",
        "113b825084cc62166c6f5584d526650928127317",
        (
            "experiment_modules",
            "modules",
            "utils",
            "configs",
            "losses.py",
            "options.py",
            "README.md",
            "LICENSE",
            "simplerecon_env.yml",
            "train.py",
            "test.py",
        ),
        "simplerecon.patch",
    ),
    "mvsanywhere": (
        "nianticlabs/mvsanywhere",
        "5bd49bba0992aa56060e1aa3bb40b2150b04726f",
        (
            "src/mvsanywhere",
            "configs",
            "README.md",
            "LICENSE",
            "environment.yml",
            "hubconf.py",
            "setup.py",
            "pyproject.toml",
            "Makefile",
        ),
        "mvsanywhere.patch",
    ),
    "octreeocc": ("4DVLab/OctreeOcc", "54afdd64b210fa45a0b1e1f255e4f88739b4e5a8", (), None),
    "gaussianformer": (
        "huang-yh/GaussianFormer",
        "b7e22bfc04cd6360cdee74be5af7fdace102f0a3",
        (),
        "gaussianformer2_native.patch",
    ),
    "dinov2": (
        "facebookresearch/dinov2",
        "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        ("dinov2", "hubconf.py", "LICENSE", "README.md"),
        None,
    ),
    "pointcept": (
        "Pointcept/Pointcept",
        "1342eda30e96cbb5fadf5374ff8eb18f6de15c71",
        ("libs/pointops", "LICENSE"),
        None,
    ),
}


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def download(url):
    # 显式系统证书；不关闭 TLS 验证。
    context = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
    for attempt in range(4):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "official-methods-source-audit"}),
                timeout=45,
                context=context,
            ) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def tree_index(folder, repository, commit, read_only):
    index = folder / "UPSTREAM_TREE.json"
    if (folder / ".git").exists():
        actual = subprocess.check_output(
            ["git", "-C", str(folder), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != commit:
            raise RuntimeError(
                f"{folder}: HEAD={actual}，应为 {commit}；保留当前 checkout，不自动切换"
            )
    if index.exists():
        tree = json.loads(index.read_text())
    elif (folder / ".git").exists():
        raw = subprocess.check_output(["git", "-C", str(folder), "ls-tree", "-r", "-z", commit])
        entries = []
        for line in raw.split(b"\0"):
            if not line:
                continue
            metadata, path = line.decode().split("\t", 1)
            mode, kind, sha = metadata.split()
            entries.append({"path": path, "mode": mode, "type": kind, "sha": sha})
        tree = {"sha": commit, "tree": entries, "truncated": False}
    elif read_only:
        raise FileNotFoundError(f"{folder}: 缺少离线索引；审计模式不联网")
    else:
        url = f"https://api.github.com/repos/{repository}/git/trees/{commit}?recursive=1"
        tree = json.loads(download(url))
    if tree["sha"] != commit or tree.get("truncated", False):
        raise ValueError(f"{repository}: Git tree 不完整或提交不符")
    if not read_only and not index.exists():
        folder.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(tree, indent=2) + "\n")
    return tree["tree"]


def selected(entry, prefixes):
    path = entry["path"]
    # 不下载数据集、大权重与展示素材；必需网络源码、配置和许可证保留。
    excluded = (
        path.startswith(("assets/", "outputs/", "checkpoints/", "pretrained/"))
        or path.endswith(
            (".pth", ".pt", ".ckpt", ".npz", ".png", ".jpg", ".gif", ".mp4", ".DS_Store", ".pyc")
        )
        or entry.get("mode") == "120000"
    )  # 作者机器的数据路径软链接不是源码。
    return (
        entry["type"] == "blob"
        and not excluded
        and (not prefixes or any(path == p or path.startswith(p + "/") for p in prefixes))
    )


def apply_patch(folder, patch_file):
    # 排版后补丁上下文会改变；逐文件哈希仍能确认算法补丁已经存在。
    accepted = json.loads((ROOT / "patches/provenance.json").read_text())
    formatting = json.loads((ROOT / "patches/formatting.json").read_text())["files"]
    expected = accepted.get(folder.name, {}).get("files", {})
    if expected:
        matched = True
        for path, patched_sha in expected.items():
            file = folder / path
            sha = hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
            formatted = formatting.get(f"{folder.name}/{path}", {})
            matched &= sha == patched_sha or (
                formatted.get("before_sha256") == patched_sha
                and sha == formatted.get("after_sha256")
            )
        if matched:
            return "already_applied"
    command = ["git", "-C", str(folder), "apply"]
    if (
        subprocess.run(
            command + ["--reverse", "--check", str(patch_file)], capture_output=True
        ).returncode
        == 0
    ):
        return "already_applied"
    subprocess.run(command + ["--check", str(patch_file)], check=True)
    subprocess.run(command + [str(patch_file)], check=True)
    return "applied"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--root", type=Path, default=ROOT / "upstream")
    parser.add_argument("--audit", action="store_true", help="只读校验已有源码，不联网、不写文件")
    parser.add_argument("--dry-run", action="store_true", help="等价于 --audit，列出缺失/变更文件")
    parser.add_argument(
        "--apply-patches", action="store_true", help="获取完成后显式应用版本兼容补丁；已应用则跳过"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--report", type=Path, help="可选 JSON 审计报告路径")
    args = parser.parse_args()
    read_only = args.audit or args.dry_run
    if read_only and args.apply_patches:
        parser.error("只读模式不能应用补丁")
    patch_index = ROOT / "patches/provenance.json"
    accepted = json.loads(patch_index.read_text()) if patch_index.exists() else {}
    format_index = ROOT / "patches/formatting.json"
    formatting = json.loads(format_index.read_text())["files"] if format_index.exists() else {}
    reports = {}
    failed = False
    for name in args.methods:
        repository, commit, prefixes, patch = SOURCES[name]
        folder = args.root / name
        entries = [
            e for e in tree_index(folder, repository, commit, read_only) if selected(e, prefixes)
        ]

        def check_or_fetch(entry):
            path = folder / entry["path"]
            if path.exists():
                data = path.read_bytes()
                if blob_sha(data) == entry["sha"]:
                    return ("original", entry["path"])
                sha256 = hashlib.sha256(data).hexdigest()
                known = accepted.get(name, {}).get("files", {}).get(entry["path"])
                formatted = formatting.get(f"{name}/{entry['path']}", {})
                if sha256 == formatted.get("after_sha256") and (
                    formatted.get("before_blob_sha1") == entry["sha"]
                    or (known is not None and formatted.get("before_sha256") == known)
                ):
                    return ("recorded_format", entry["path"])
                return (
                    "recorded_patch" if known == sha256 else "modified_preserved",
                    entry["path"],
                )
            if read_only:
                return ("missing", entry["path"])
            data = download(
                f"https://raw.githubusercontent.com/{repository}/{commit}/{entry['path']}"
            )
            if blob_sha(data) != entry["sha"]:
                raise ValueError(f"Git blob SHA-1 校验失败：{path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            # 独占创建：即使并发任务刚刚添加此文件，也不覆盖它。
            with path.open("xb") as handle:
                handle.write(data)
            return ("downloaded_verified", entry["path"])

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            checked = list(pool.map(check_or_fetch, entries))
        result = {
            "repository": f"https://github.com/{repository}",
            "commit": commit,
            "selected_files": len(entries),
        }
        for status in sorted({status for status, _ in checked}):
            files = [path for state, path in checked if state == status]
            result[status] = (
                files
                if status in ("missing", "modified_preserved", "recorded_patch")
                else len(files)
            )
        result["patch_files"] = {
            entry["path"]: {
                "upstream_blob_sha1": entry["sha"],
                "local_sha256": accepted[name]["files"][entry["path"]],
            }
            for entry in entries
            if entry["path"] in result.get("recorded_patch", [])
        }
        if args.apply_patches and patch:
            result["patch"] = apply_patch(folder, ROOT / "patches" / patch)
        reports[name] = result
        failed |= bool(result.get("missing") or result.get("modified_preserved"))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
