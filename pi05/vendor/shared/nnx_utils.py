"""提供 NNX 模块编译、参数路径筛选和状态映射工具。"""

from collections.abc import Callable
import dataclasses
import functools
import inspect
import re
from typing import Any, ParamSpec, TypeVar
import flax.nnx as nnx
import jax

P = ParamSpec("P")
R = TypeVar("R")


def module_jit(meth: Callable[P, R], *jit_args, **jit_kwargs) -> Callable[P, R]:
    """将 NNX 模块方法包装成带状态重建的 JIT 函数。

    输入：meth、*jit_args、**jit_kwargs。
    输出：可调用的编译包装器。"""
    if not (inspect.ismethod(meth) and isinstance(meth.__self__, nnx.Module)):
        raise ValueError("module_jit must only be used on bound methods of nnx.Modules.")
    graphdef, state = nnx.split(meth.__self__)

    def fun(state: nnx.State, *args: P.args, **kwargs: P.kwargs) -> R:
        """合并模块图结构与状态后调用原始方法。

        输入：state、*args、**kwargs。
        输出：原始模块方法的返回值。"""
        module = nnx.merge(graphdef, state)
        return meth.__func__(module, *args, **kwargs)

    jitted_fn = jax.jit(fun, *jit_args, **jit_kwargs)

    @functools.wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """提取模块当前状态并调用编译函数。

        输入：*args、**kwargs。
        输出：原始方法的返回值。"""
        return jitted_fn(state, *args, **kwargs)

    return wrapper


@dataclasses.dataclass(frozen=True)
class PathRegex:
    pattern: str | re.Pattern
    sep: str = "/"

    def __post_init__(self):
        """编译参数路径匹配表达式。

        输入：无外部参数，读取当前对象配置。
        输出：无返回值，保存正则对象。"""
        if not isinstance(self.pattern, re.Pattern):
            object.__setattr__(self, "pattern", re.compile(self.pattern))

    def __call__(self, path: nnx.filterlib.PathParts, x: Any) -> bool:
        """用完整参数路径匹配正则表达式。

        输入：path、x。
        输出：是否命中的布尔值。"""
        joined_path = self.sep.join((str(x) for x in path))
        assert isinstance(self.pattern, re.Pattern)
        return self.pattern.fullmatch(joined_path) is not None


def state_map(state: nnx.State, filter: nnx.filterlib.Filter, fn: Callable[[Any], Any]) -> nnx.State:
    """仅对过滤器命中的 NNX 参数应用函数。

    输入：state、filter、fn。
    输出：映射后的状态树。"""
    filtered_keys = set(state.filter(filter).flat_state())
    return state.map(lambda k, v: fn(v) if k in filtered_keys else v)
