"""定义官方 π 模型结构、输入形状和参数精度。"""

import dataclasses
from typing import TYPE_CHECKING
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override
from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None
    pi05: bool = False
    discrete_state_input: bool = None

    def __post_init__(self):
        """依据 π0.5 模式补齐令牌长度与离散状态开关。

        输入：无外部参数，读取当前对象配置。
        输出：无返回值，补齐配置默认项。"""
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        """取得模型类型标识。

        输入：无外部参数，读取当前对象配置。
        输出：对应的 ModelType 枚举值。"""
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        """按照配置创建对应组件。

        输入：rng。
        输出：初始化的模型、优化器或学习率函数。"""
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """构造模型观测与动作的形状规格。

        输入：batch_size。
        输出：观测结构和动作 ShapeDtypeStruct。"""
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return (observation_spec, action_spec)
