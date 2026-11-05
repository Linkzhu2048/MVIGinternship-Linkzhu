"""保留官方预热余弦学习率与带全局梯度裁剪的 AdamW。"""

import dataclasses
from typing import Protocol, runtime_checkable
import jax.numpy as jnp
import optax
import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):

    def create(self) -> optax.Schedule:
        """按照配置创建对应组件。

        输入：无外部参数，读取当前对象配置。
        输出：初始化的模型、优化器或学习率函数。"""
        ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    warmup_steps: int = 1000
    peak_lr: float = 2.5e-05
    decay_steps: int = 30000
    decay_lr: float = 2.5e-06

    def create(self) -> optax.Schedule:
        """构建预热后余弦衰减的学习率计划。

        输入：无外部参数，读取当前对象配置。
        输出：Optax 学习率函数。"""
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@runtime_checkable
class OptimizerConfig(Protocol):

    def create(
        self, lr: optax.ScalarOrSchedule, weight_decay_mask: at.PyTree | None = None
    ) -> optax.GradientTransformation:
        """按照配置创建对应组件。

        输入：lr、weight_decay_mask。
        输出：初始化的模型、优化器或学习率函数。"""
        ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-08
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self, lr: optax.ScalarOrSchedule, weight_decay_mask: at.PyTree | None = None
    ) -> optax.GradientTransformation:
        """构建全局梯度裁剪加 AdamW 的优化器。

        输入：lr、weight_decay_mask。
        输出：Optax GradientTransformation。"""
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )
        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    """将学习率计划与优化器配置组合。

    输入：optimizer、lr_schedule、weight_decay_mask。
    输出：Optax 优化器。"""
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
