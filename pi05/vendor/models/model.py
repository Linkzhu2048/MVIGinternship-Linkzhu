"""定义观测、模型接口和 JAX 权重恢复；仅保留本项目需要的接口。"""

import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar
import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    PI0 = "pi0"
    PI05 = "pi05"


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_RESOLUTION = (224, 224)


@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b s"]
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """将输入字典转为观测并把 uint8 图像归一化到负一至一。

        输入：data。
        输出：Observation 对象。"""
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = (
                    data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                )
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """将观测结构转换为输入字段字典。

        输入：无外部参数，读取当前对象配置。
        输出：含 image、image_mask 等键的字典。"""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """核对相机、调整图像尺寸并按训练标记施加官方增强。

    输入：rng、observation、train、image_keys、image_resolution。
    输出：预处理后的 Observation。"""
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")
    batch_shape = observation.state.shape[:-1]
    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)
        if train:
            image = image / 2.0 + 0.5
            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)
            image = image * 2.0 - 1.0
        out_images[key] = image
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])
    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    action_dim: int
    action_horizon: int
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """取得模型类型标识。

        输入：无外部参数，读取当前对象配置。
        输出：对应的 ModelType 枚举值。"""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """按照配置创建对应组件。

        输入：rng。
        输出：初始化的模型、优化器或学习率函数。"""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """按模型结构校验并装入完整参数。

        输入：params、remove_extra_params。
        输出：加载权重后的模型。"""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(
            expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False
        )
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """构造模型观测与动作的形状规格。

        输入：batch_size。
        输出：观测结构和动作 ShapeDtypeStruct。"""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        """为网络初始化构造形状占位观测，不作为训练数据。

        输入：batch_size。
        输出：全一观测张量，用于视觉编码器 lazy_init。"""
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: Observation, actions: Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """对加噪动作与真实速度目标计算流匹配损失。

        输入：rng、observation、actions、train。
        输出：各批次各动作时刻的均方误差。"""
        ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions:
        """复用前缀缓存从噪声迭代生成动作块。

        输入：rng、observation、**kwargs。
        输出：模型空间的动作张量，尚未反归一化。"""
        ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """恢复 Orbax 参数并去除 NNX value 后缀。

    输入：params_path、restore_type、dtype、sharding。
    输出：指定精度、数组类型与分片的参数树。"""
    params_path = (
        pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path
    )
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype),
                    item,
                ),
            ),
        )["params"]
    flat_params = traverse_util.flatten_dict(params)
    if all((kp[-1] == "value" for kp in flat_params)):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
