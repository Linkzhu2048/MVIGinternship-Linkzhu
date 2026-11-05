"""将完成的训练检查点导出为可迁移 ZIP，包含 EMA 参数、统计量、分词器和所需源码。"""

import argparse
import json
from pathlib import Path
import tempfile
import zipfile
from .config import PROJECT_ROOT, digest
from .checkpoint import source_fingerprint


def export(run_dir, destination, step=None):
    """导出完整推理资产；输入：训练目录、ZIP 路径及可选步数；输出：ZIP 路径，拒绝覆盖已有文件。"""
    run_dir, destination = (Path(run_dir).resolve(), Path(destination).resolve())
    if destination.suffix.lower() != ".zip":
        raise ValueError("模型导出文件必须使用 .zip 扩展名")
    if destination.exists():
        raise FileExistsError(destination)
    info = json.loads((run_dir / "run.json").read_text())
    if info.get("test_only") or info["source_fingerprint"] != source_fingerprint():
        raise ValueError("当前代码与训练记录不一致，请使用训练时的源代码导出")
    checkpoints = run_dir / "checkpoints"
    completed = [
        int(p.name)
        for p in checkpoints.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / "_CHECKPOINT_METADATA").is_file()
    ]
    if not completed:
        raise ValueError("尚无完成写入的检查点")
    step = max(completed) if step is None else step
    if step not in completed:
        raise ValueError("指定步数尚未完成保存")
    checkpoint = checkpoints / str(step)
    files = {}
    for folder in ("params", "assets"):
        if not (checkpoint / folder).is_dir():
            raise FileNotFoundError(checkpoint / folder)
        for path in sorted((checkpoint / folder).rglob("*")):
            if path.is_file():
                files[path.relative_to(checkpoint).as_posix()] = path
    for name in ("run.json", "dataset_manifest.json", "norm_provenance.json", "tokenizer.model"):
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = path
    for folder in ("src", "vendor", "configs"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and (path.suffix != ".pyc"):
                files["code/" + path.relative_to(PROJECT_ROOT).as_posix()] = path
    for name in ("README.md", "requirements.txt", "pyproject.toml"):
        files["code/" + name] = PROJECT_ROOT / name
    settings = info["settings"]
    contract = {
        "schema_version": 1,
        "model": "pi05",
        "backend": "jax",
        "action_dim": 14,
        "model_action_dim": 32,
        "action_horizon": settings["action_horizon"],
        "fps": settings["fps"],
        "prompt": settings["prompt"],
        "cameras": ["left_wrist", "right_wrist"],
        "image_size": [224, 224],
        "image_format": "HWC uint8 RGB",
        "action_space": "absolute_joint_radians_and_normalized_grippers",
        "model_action_space": "relative_joint_targets_except_absolute_grippers",
        "checkpoint_step": step,
        "weights": "EMA",
        "ema_decay": settings["ema_decay"],
        "source_fingerprint": info["source_fingerprint"],
        "layout": ["left_joint_" + str(i) for i in range(1, 7)]
        + ["left_gripper"]
        + ["right_joint_" + str(i) for i in range(1, 7)]
        + ["right_gripper"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pi05_export_", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        contract_path = temporary / "contract.json"
        contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
        files["contract.json"] = contract_path
        checksums = {name: digest(path) for name, path in files.items()}
        archive = temporary / "model.zip"
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
        ) as bundle:
            for name, path in files.items():
                bundle.write(path, name)
            bundle.writestr("checksums.json", json.dumps(checksums, indent=2) + "\n")
        destination.hardlink_to(archive)
    return str(destination)


def verify(archive):
    """检查导出包完整性；输入：模型 ZIP；输出：模型契约，逐文件校验散列且不解压到磁盘。"""
    import hashlib
    from pathlib import PurePosixPath

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP 包含重复路径")
        expected = json.loads(bundle.read("checksums.json"))
        if set(names) != set(expected) | {"checksums.json"}:
            raise ValueError("ZIP 内容与校验清单不一致")
        for name, checksum in expected.items():
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError("ZIP 包含越界路径")
            actual = hashlib.sha256()
            with bundle.open(name) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    actual.update(block)
            if actual.hexdigest() != checksum:
                raise ValueError(f"模型包文件损坏：{name}")
        return json.loads(bundle.read("contract.json"))


def main():
    """执行模型导出或校验；输入：子命令与路径；输出：ZIP 路径或模型契约。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--run", type=Path, default=PROJECT_ROOT / "output/runs/pi05_v1")
    create.add_argument("--step", type=int)
    create.add_argument("--output", type=Path, default=PROJECT_ROOT / "output/models/pi05.zip")
    inspect = sub.add_parser("verify")
    inspect.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        print(export(args.run, args.output, args.step))
    else:
        print(json.dumps(verify(args.archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
