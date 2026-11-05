"""校验训练权重并导出独立 ZIP，包含模型源码、配置、统计及训练来源。"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import torch

from .config import PROJECT_ROOT
from .model import ACTPolicy


def export_run(run, destination, include_resume=False):
    """打包模型；输入训练目录、ZIP 路径及是否带续训状态，输出包信息，不覆盖已有文件。"""
    run, destination = Path(run).resolve(), Path(destination).resolve()
    temporary = destination.with_name(destination.name + '.partial')
    if destination.suffix.lower() != '.zip':
        raise ValueError('导出文件必须使用 .zip 后缀')
    if destination.exists() or temporary.exists():
        raise FileExistsError('目标 ZIP 或临时文件已存在')
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    policy = ACTPolicy(config['model'], pretrained=False)
    policy.load_state_dict(torch.load(run / 'policy_best.ckpt', map_location='cpu', weights_only=True), strict=True)
    required = ['policy_best.ckpt', 'policy_last.ckpt', 'config.json', 'dataset_stats.pkl',
                'provenance.json', 'metrics.jsonl', 'summary.json']
    if include_resume:
        required.append('resume.ckpt')
    files = []
    for name in required:
        path = run / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append((path, 'run/' + name))
    for path in sorted((run / 'dataset_assets').glob('*')):
        if path.is_file():
            files.append((path, 'run/dataset_assets/' + path.name))
    # 保持已有部署包的 code/ 与 run/ 布局，避免改变部署端的加载路径。
    for directory in ('src', 'vendor', 'configs'):
        for path in sorted((PROJECT_ROOT / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files.append((path, 'code/' + path.relative_to(PROJECT_ROOT).as_posix()))
    for name in ('pyproject.toml', 'requirements.txt'):
        files.append((PROJECT_ROOT / name, 'code/' + name))
    files.append((PROJECT_ROOT / 'README.md', 'README.md'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksums = {}
    with ZipFile(temporary, 'x', compression=ZIP_DEFLATED) as archive:
        for path, member in files:
            # 分块读入，避免大型检查点全部复制到内存。
            checksum = hashlib.sha256()
            with path.open('rb') as source, archive.open(member, 'w', force_zip64=True) as target:
                while block := source.read(4 * 1024 * 1024):
                    target.write(block)
                    checksum.update(block)
            checksums[member] = checksum.hexdigest()
        archive.writestr('manifest.json', json.dumps({'files': checksums}, indent=2))
    with ZipFile(temporary) as archive:
        failed = archive.testzip()
        if failed is not None:
            raise ValueError(f'ZIP 内容校验失败：{failed}')
    temporary.replace(destination)
    return dict(archive=str(destination), files=len(files), size_bytes=destination.stat().st_size,
                strict_weight_load_passed=True)


def main():
    """执行导出命令；输入训练目录及可选 ZIP 路径，输出打包结果，默认写入 output/models/。"""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--include-resume', action='store_true')
    args = parser.parse_args()
    destination = args.output or PROJECT_ROOT / 'output/models' / (args.run.resolve().name + '.zip')
    print(json.dumps(export_run(args.run, destination, args.include_resume), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
