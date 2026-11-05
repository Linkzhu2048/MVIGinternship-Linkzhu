"""记录代码、数据和环境身份，拒绝以不一致的设置继续训练。"""

from pathlib import Path
from .config import PROJECT_ROOT, digest, fingerprint


def source_fingerprint():
    """计算代码身份；输入：项目内 src、vendor 和依赖声明；输出：稳定的 SHA256 字符串。"""
    hashes = {}
    for directory in ("src", "vendor"):
        for path in sorted((PROJECT_ROOT / directory).rglob("*.py")):
            hashes[str(path.relative_to(PROJECT_ROOT))] = digest(path)
    for name in ("pyproject.toml", "requirements.txt"):
        hashes[name] = digest(PROJECT_ROOT / name)
    return fingerprint(hashes)


def run_identity(settings):
    """核对数据并记录代码、权重、分词器和依赖版本身份。

    输入：settings。
    输出：恢复训练时用于严格比较的身份字典。"""
    from .dataset import manifest, load_stats

    info = manifest(settings.dataset_root, verify_hashes=True)
    if info["repo_id"] != settings.repo_id or info["prompt"] != settings.prompt:
        raise ValueError("Dataset repo id/prompt differs from configuration")
    load_stats(settings)
    params = Path(settings.pretrained_params)
    if not params.is_dir() or not any(params.rglob("*")):
        raise FileNotFoundError("缺少基础权重 params 目录")
    weights_hash = fingerprint(
        {p.relative_to(params).as_posix(): digest(p) for p in sorted(params.rglob("*")) if p.is_file()}
    )
    value = dict(
        pretrained_sha256=weights_hash,
        settings=settings.as_dict(),
        source_fingerprint=source_fingerprint(),
        dataset_sha256=digest(Path(settings.dataset_root) / "i2rt_manifest.json"),
        norm_sha256=digest(Path(settings.assets_dir) / "norm_stats.json"),
        norm_provenance_sha256=digest(Path(settings.assets_dir) / "provenance.json"),
        tokenizer_sha256=digest(settings.tokenizer_path),
    )
    import importlib.metadata

    value["runtime_versions"] = {
        name: importlib.metadata.version(name)
        for name in (
            "jax",
            "jaxlib",
            "flax",
            "optax",
            "orbax-checkpoint",
            "lerobot",
            "datasets",
            "numpy",
            "pillow",
        )
    }
    return {"fingerprint": fingerprint(value), **value}
