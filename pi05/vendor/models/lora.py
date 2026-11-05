"""保留 Gemma 依赖的线性层与前馈层；当前配置进行全量微调，不启用低秩适配。"""

import math
import re
import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp
import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    rank: int
    alpha: float = 1.0
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    rslora: bool = False
    axes: tuple[int, int] = (-2, -1)
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        """计算低秩分支的尺度系数。

        输入：无外部参数，读取当前对象配置。
        输出：浮点缩放系数。"""
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    shape: tuple[int, ...]
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    lora_config: LoRAConfig | None = None

    def setup(self):
        """注册基础权重及可选低秩权重。

        输入：无外部参数，读取当前对象配置。
        输出：无返回值，初始化张量投影参数。"""
        self.w = self.param("w", self.init_fn, self.shape)
        if config := self.lora_config:
            shape_a, shape_b = (list(self.shape), list(self.shape))
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        """执行基础张量投影并按配置叠加低秩分支。

        输入：eqn、x。
        输出：投影后的张量。"""
        dtype = x.dtype
        result = jnp.einsum(eqn, x, self.w.astype(dtype))
        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value
        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        """把基础求和公式拆成两个低秩投影公式。

        输入：eqn。
        输出：两条 einsum 公式。"""
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()
        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label
        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"
        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"
        return (eqn_a, eqn_b)


class FeedForward(nn.Module):
    features: int
    hidden_dim: int
    lora_config: LoRAConfig | None = None

    def setup(self):
        """创建当前网络层需要的权重和子模块。

        输入：无外部参数，读取当前对象配置。
        输出：无返回值，注册参数与子层。"""
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear", nn.initializers.lecun_normal(in_axis=-2, out_axis=-1), (self.hidden_dim, self.features)
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            self.w_gating_lora = (
                self.param(
                    "gating_einsum_lora_a",
                    self.lora_config.init_fn,
                    (2, self.features, self.lora_config.rank),
                ),
                self.param(
                    "gating_einsum_lora_b",
                    self.lora_config.init_fn,
                    (2, self.lora_config.rank, self.hidden_dim),
                ),
            )
            self.w_linear_lora = (
                self.param(
                    "linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)
                ),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x):
        """执行门控前馈变换。

        输入：x。
        输出：变换后的隐藏特征。"""
        dtype = x.dtype
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.gelu(ff_gate)
        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        activations = gate_value * ff1
        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        """执行矩阵乘法并可选叠加低秩增量。

        输入：x、w、lora_weights。
        输出：乘积张量。"""
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
