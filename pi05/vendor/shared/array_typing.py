"""提供官方数组类型标注和参数树一致性检查。"""

import contextlib
import functools as ft
import inspect
from typing import TypeAlias, TypeVar, cast
import beartype
import jax
import jax._src.tree_util as private_tree_util
import jax.core
from jaxtyping import ArrayLike
from jaxtyping import Bool
from jaxtyping import DTypeLike
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Key
from jaxtyping import Num
from jaxtyping import PyTree
from jaxtyping import Real
from jaxtyping import UInt8
from jaxtyping import config
from jaxtyping import jaxtyped
import jaxtyping._decorator
import torch

_original_check_dataclass_annotations = jaxtyping._decorator._check_dataclass_annotations
Array = jax.Array | torch.Tensor


def _check_dataclass_annotations(self, typechecker):
    """为数据类的字段标注执行运行期检查。

    输入：typechecker。
    输出：无返回值，类型不符时抛出异常。"""
    if not any(
        (
            frame.frame.f_globals.get("__name__") in {"jax._src.tree_util", "flax.nnx.transforms.compilation"}
            for frame in inspect.stack()
        )
    ):
        return _original_check_dataclass_annotations(self, typechecker)
    return None


jaxtyping._decorator._check_dataclass_annotations = _check_dataclass_annotations
KeyArrayLike: TypeAlias = jax.typing.ArrayLike
Params: TypeAlias = PyTree[Float[ArrayLike, "..."]]
T = TypeVar("T")


def typecheck(t: T) -> T:
    """为函数或数据类安装数组类型检查。

    输入：t。
    输出：带检查的原对象。"""
    return cast(T, ft.partial(jaxtyped, typechecker=beartype.beartype)(t))


@contextlib.contextmanager
def disable_typechecking():
    """在上下文内临时暂停数组类型检查。

    输入：无外部参数，读取当前对象配置。
    输出：上下文管理器。"""
    initial = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)
    yield
    config.update("jaxtyping_disable", initial)


def check_pytree_equality(
    *, expected: PyTree, got: PyTree, check_shapes: bool = False, check_dtypes: bool = False
):
    """检查参数树结构及可选形状和数据类型。

    输入：expected、got、check_shapes、check_dtypes。
    输出：无返回值，不匹配时抛出异常。"""
    if errors := list(private_tree_util.equality_errors(expected, got)):
        raise ValueError(
            "PyTrees have different structure:\n"
            + "\n".join(
                (
                    f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}.\n"
                    for path, thing1, thing2, explanation in errors
                )
            )
        )
    if check_shapes or check_dtypes:

        def check(kp, x, y):
            """逐叶核对参数数组的形状与类型。

            输入：kp 为参数路径，x 为参考数组，y 为待检查数组。
            输出：无返回值；不一致时抛出带参数路径的异常。"""
            if check_shapes and x.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}"
                )
            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(
                    f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}"
                )

        jax.tree_util.tree_map_with_path(check, expected, got)
