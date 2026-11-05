"""保留官方归一化、反归一化、提示词编码和状态动作补维变换。"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable
import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats
T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):

    def __call__(self, data: DataDict) -> DataDict:
        """约定未分批数据的转换接口。

        输入：data。
        输出：转换后的数据字典。"""


@dataclasses.dataclass(frozen=True)
class Group:
    inputs: Sequence[DataTransformFn] = ()
    outputs: Sequence[DataTransformFn] = ()


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        """依次执行组成变换。

        输入：data。
        输出：完成全部转换的数据字典。"""
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """将一组变换封装成单个可调用对象。

    输入：transforms。
    输出：CompositeTransform 实例。"""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False
    strict: bool = False

    def __post_init__(self):
        """检查分位数归一化所需统计是否齐全。

        输入：无外部参数，读取当前对象配置。
        输出：无返回值，缺少统计时抛出异常。"""
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        """按照选择的统计字段进行归一化。

        输入：data。
        输出：归一化后的数据字典。"""
        if self.norm_stats is None:
            return data
        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        """使用均值和标准差作标准化。

        输入：x、stats。
        输出：标准化数组。"""
        mean, std = (stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]])
        return (x - mean) / (std + 1e-06)

    def _normalize_quantile(self, x, stats: NormStats):
        """使用 q01 与 q99 作官方线性归一化，不截断越界值。

        输入：x、stats。
        输出：归一化数组。"""
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = (stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]])
        return (x - q01) / (q99 - q01 + 1e-06) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False

    def __post_init__(self):
        """检查反归一化所需分位数统计。

        输入：无外部参数，读取当前对象配置。
        输出：无返回值，缺少统计时抛出异常。"""
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        """按统计量恢复原始数值尺度。

        输入：data。
        输出：反归一化后的数据字典。"""
        if self.norm_stats is None:
            return data
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        """逆转均值标准差标准化并兼容补维。

        输入：x、stats。
        输出：恢复尺度的数组。"""
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-06) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        """逆转分位数归一化并保留补维字段。

        输入：x、stats。
        输出：恢复尺度的数组。"""
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = (stats.q01, stats.q99)
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate(
                [(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-06) + q01, x[..., dim:]], axis=-1
            )
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-06) + q01


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        """弹出任务文本并可选连同状态进行编码。

        输入：data。
        输出：含令牌及有效掩码的数据字典。"""
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None
        if not isinstance(prompt, str):
            prompt = prompt.item()
        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        """将状态和动作末维补零至模型维度。

        输入：data。
        输出：补维后的数据字典。"""
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """用斜杠路径展开嵌套字典。

    输入：tree。
    输出：扁平字典。"""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """按斜杠路径恢复嵌套字典。

    输入：tree。
    输出：参数或数据树。"""
    return traverse_util.unflatten_dict(tree, sep="/")


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    """按选择器对匹配的数据叶子应用函数。

    输入：tree、selector、fn、strict。
    输出：变换后的数据树。"""
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        """转换被统计量选择器命中的叶子。

        输入：k、v。
        输出：转换值或保持原值。"""
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")
    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """沿指定轴将数组补到目标长度。

    输入：x、target_dim、axis、value。
    输出：补齐后的数组，长度足够时保持原数组。"""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    """检查各字段均有 q01 和 q99。

    输入：norm_stats。
    输出：无返回值，缺失统计时抛出异常。"""
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
