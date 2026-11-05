"""从本地 Orbax 目录读取基础参数，并对齐模型所需的参数结构和数据类型。"""

import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable
import flax.traverse_util
import numpy as np
import openpi.models.model as _model
import openpi.shared.array_typing as at

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):

    def load(self, params: at.Params) -> at.Params:
        """约定权重加载与参数结构保持接口。

        输入：params。
        输出：与参考结构对应的参数树。"""


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        """从显式本地 Orbax 路径读取并对齐权重。

        输入：params。
        输出：与模型参考参数对应的数组树。"""
        loaded_params = _model.restore_params(self.params_path, restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """按参考树筛选和转换参数并填充允许缺失的低秩项。

    输入：loaded_params、params、missing_regex。
    输出：合并后的参数字典。"""
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
    flat_loaded.clear()
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]
    return flax.traverse_util.unflatten_dict(result, sep="/")
