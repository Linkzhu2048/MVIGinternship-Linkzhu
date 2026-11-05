"""下载和校验官方 π0.5 JAX 权重及分词器，支持分块断点续传，资源仅写入指定目录。"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.parse
import urllib.request
import google_crc32c
from .config import PROJECT_ROOT, write_json

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_base/params/"
TOKEN_BUCKET = "big_vision"
TOKEN_NAME = "paligemma_tokenizer.model"


def crc32c(path):
    """计算文件传输校验值；输入：文件路径；输出：与 GCS 元数据一致的 Base64 CRC32C。"""
    checksum = google_crc32c.Checksum()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()


def read_json(url):
    """读取公开资源元数据；输入：HTTPS 地址；输出：解析后的字典，失败最多重试五次。"""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def remote_files():
    """列出固定官方资源；输入：无；输出：包含对象版本、大小、校验码和项目内路径的清单。"""
    items, token = ([], None)
    while True:
        query = {"prefix": PREFIX, "maxResults": 1000}
        if token:
            query["pageToken"] = token
        listing = read_json(
            f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(query)
        )
        for item in listing.get("items", []):
            relative = item["name"].removeprefix(PREFIX)
            if not item["name"].startswith(PREFIX) or not relative or ".." in PurePosixPath(relative).parts:
                raise ValueError("权重清单含无效路径")
            items.append(
                {
                    **{k: item[k] for k in ("name", "size", "crc32c", "generation")},
                    "bucket": BUCKET,
                    "local": "params/" + relative,
                }
            )
        token = listing.get("nextPageToken")
        if not token:
            break
    if not items:
        raise ValueError("官方权重清单为空")
    tokenizer = read_json(f"https://storage.googleapis.com/storage/v1/b/{TOKEN_BUCKET}/o/{TOKEN_NAME}")
    items.append(
        {
            **{k: tokenizer[k] for k in ("name", "size", "crc32c", "generation")},
            "bucket": TOKEN_BUCKET,
            "local": "tokenizer.model",
        }
    )
    return items


def local_path(root, name):
    """限制资源写入范围；输入：资源根目录与相对文件名；输出：根目录内的绝对路径。"""
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"资源路径越界：{name}")
    return path


def fetch_file(item, root):
    """按对象版本续传一个资源；输入：远端元数据与目标根目录；输出：通过 CRC 校验的本地路径。"""
    target = local_path(root, item["local"])
    target.parent.mkdir(parents=True, exist_ok=True)
    size = int(item["size"])
    if target.is_file() and target.stat().st_size == size and (crc32c(target) == item["crc32c"]):
        return str(target)
    partial = target.with_name(target.name + ".download")
    if partial.exists() and partial.stat().st_size > size:
        partial.unlink()
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{item['bucket']}/o/"
        + urllib.parse.quote(item["name"], safe="")
        + "?alt=media&generation="
        + str(item["generation"])
    )
    for full_attempt in range(2):
        partial.touch(exist_ok=True)
        while partial.stat().st_size < size:
            start = partial.stat().st_size
            end = min(size, start + 8 * 1024 * 1024) - 1
            for attempt in range(8):
                try:
                    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=90) as response:
                        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                            f"bytes {start}-{end}/"
                        ):
                            raise IOError("服务器未按指定范围返回资源")
                        block = response.read(end - start + 2)
                    if len(block) != end - start + 1:
                        raise IOError("下载块长度不符")
                    with partial.open("ab") as stream:
                        stream.write(block)
                    break
                except OSError:
                    if attempt == 7:
                        raise
                    time.sleep(2 * (attempt + 1))
        if crc32c(partial) == item["crc32c"]:
            partial.replace(target)
            print(f"完成 {item['local']} ({size} 字节)", flush=True)
            return str(target)
        partial.unlink()
    raise ValueError(f"资源校验失败：{item['local']}")


def download(root, workers=4):
    """准备训练资源；输入：目录与并发数；输出：全部资源校验报告，重复运行复用已验证文件。"""
    import filelock

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(root) + ".lock", timeout=0):
        listing_path = root / "download_manifest.json"
        if listing_path.exists():
            items = json.loads(listing_path.read_text())["files"]
        else:
            items = remote_files()
            write_json(listing_path, {"source": "official_pi05_base", "files": items})
        needed = sum((int(item["size"]) for item in items if not local_path(root, item["local"]).exists()))
        if shutil.disk_usage(root).free < needed:
            raise OSError("磁盘剩余空间不足以下载基础权重")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda item: fetch_file(item, root), items))
        return check(root / "params", root / "tokenizer.model", listing_path)


def check(params, tokenizer, listing=None):
    """检查本地基础资源；输入：权重目录、分词器和可选下载清单；输出：结构与传输校验结果。"""
    import sentencepiece
    import orbax.checkpoint as ocp

    params, tokenizer = (Path(params).resolve(), Path(tokenizer).resolve())
    if not params.is_dir() or not tokenizer.is_file():
        raise FileNotFoundError("缺少 params/ 或 tokenizer.model，请先下载或完整复制资源")
    if listing is not None:
        listing = Path(listing)
        for item in json.loads(listing.read_text())["files"]:
            path = local_path(listing.parent, item["local"])
            if (
                not path.is_file()
                or path.stat().st_size != int(item["size"])
                or crc32c(path) != item["crc32c"]
            ):
                raise ValueError(f"资源传输校验失败：{item['local']}")
    with ocp.PyTreeCheckpointer() as reader:
        metadata = reader.metadata(params)
        if "params" not in metadata:
            raise ValueError("不是预期的 Orbax params 检查点")
    vocabulary = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer)).vocab_size()
    if vocabulary != 257152:
        raise ValueError("分词器词表大小与 PaliGemma 的 257152 个词元不匹配")
    return {
        "passed": True,
        "params": str(params),
        "tokenizer": str(tokenizer),
        "vocabulary_size": vocabulary,
        "crc_verified": listing is not None,
        "note": "资源结构检查不分配完整模型；参数形状及类型会在正式训练初始化时严格核对。",
    }


def main():
    """执行资源管理命令；输入：download/check 子命令和资源目录；输出：下载进度与校验报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "check"))
    parser.add_argument("--directory", type=Path, default=PROJECT_ROOT / "pretrained")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers 必须大于零")
    listing = args.directory / "download_manifest.json"
    result = (
        download(args.directory, args.workers)
        if args.command == "download"
        else check(
            args.directory / "params",
            args.directory / "tokenizer.model",
            listing if listing.is_file() else None,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
