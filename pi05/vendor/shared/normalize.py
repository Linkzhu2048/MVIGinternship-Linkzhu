"""读取与保存官方格式的均值、标准差及分位数统计量。"""

import json
import pathlib
import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None
    q99: numpydantic.NDArray | None = None


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """将官方统计量对象编码为 JSON。

    输入：norm_stats。
    输出：JSON 字符串。"""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """解析官方格式的归一化统计量。

    输入：data。
    输出：NormStats 字典。"""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """把归一化统计写入 norm_stats.json。

    输入：directory、norm_stats。
    输出：无返回值，生成统计文件。"""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """读取目录中的 norm_stats.json。

    输入：directory。
    输出：NormStats 字典。"""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
