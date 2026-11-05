"""转换双腕 RGB 图像及 14 维机器人状态，完成相对关节动作、归一化和提示词编码衔接。"""

import dataclasses
import numpy as np
from openpi.shared.image_tools import resize_with_pad_numpy as resize_with_pad

CAMERAS = ("left_wrist", "right_wrist")
IMAGE_KEYS = ("left_wrist_0_rgb", "right_wrist_0_rgb")
JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
DEFAULT_PROMPT = "Put all five blocks into the cup."


def robot_array(value, name, *, chunk=False):
    """检查机器人数据的维度、有限性与夹爪范围。

    输入：value 为 [14] 或 [H,14] 数值，name 为报错字段名，chunk 指示是否是动作块。
    输出：独立的 float32 数组。"""
    array = np.array(value, dtype=np.float32, copy=True)
    if (
        array.ndim != (2 if chunk else 1)
        or array.shape[-1] != 14
        or (not array.size)
        or (not np.isfinite(array).all())
    ):
        raise ValueError(f"{name} must be finite with shape {('(H,14)' if chunk else '(14,)')}")
    if np.any(array[..., [6, 13]] < 0) or np.any(array[..., [6, 13]] > 1):
        raise ValueError(f"{name} grippers must be in [0,1]")
    return array


@dataclasses.dataclass(frozen=True)
class RobotInputs:
    default_prompt: str = DEFAULT_PROMPT

    def __call__(self, data):
        """生成两路模型图像并将关节目标减去当前状态。

        输入：data 含左右腕 HWC uint8 RGB 图像、14 维状态及可选 [H,14] 绝对动作。
        输出：包含图像、状态、任务及可选动作块的字典。"""
        state = robot_array(data["state"], "state")
        if set(data["images"]) != set(CAMERAS):
            raise ValueError(f"Expected exactly these cameras: {CAMERAS}")
        images = {}
        for camera, key in zip(CAMERAS, IMAGE_KEYS, strict=True):
            value = np.asarray(data["images"][camera])
            if (
                value.dtype != np.uint8
                or value.ndim != 3
                or value.shape[-1] != 3
                or (min(value.shape[:2]) < 1)
            ):
                raise ValueError(f"{camera} must be HWC uint8 RGB")
            images[key] = resize_with_pad(value, 224, 224)
        prompt = data.get("prompt", self.default_prompt)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty task prompt is required")
        result = dict(image=images, image_mask={k: np.True_ for k in IMAGE_KEYS}, state=state, prompt=prompt)
        if "actions" in data:
            actions = robot_array(data["actions"], "actions", chunk=True)
            actions[:, JOINTS] -= state[JOINTS]
            result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class RobotOutputs:

    def __call__(self, data):
        """反归一化后将相对关节目标还原到绝对坐标。

        输入：data 含反归一化后的 state 和 actions，允许存在模型补齐的维度。
        输出：仅含前 14 维绝对目标的 actions 字典，不裁剪预测。"""
        state = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        if state.ndim != 1 or state.shape[0] < 14 or (not np.isfinite(state).all()):
            raise ValueError("Expected unnormalized state with at least 14 dimensions")
        if (
            actions.ndim != 2
            or actions.shape[1] < 14
            or (not len(actions))
            or (not np.isfinite(actions).all())
        ):
            raise ValueError("Expected finite action chunk with at least 14 dimensions")
        result = actions[:, :14].copy()
        result[:, JOINTS] += state[JOINTS]
        return {"actions": result}


def data_transforms(default_prompt=DEFAULT_PROMPT):
    """组合机器人输入和输出变换。

    输入：default_prompt。
    输出：变换 Group。"""
    from openpi import transforms

    return transforms.Group(inputs=[RobotInputs(default_prompt)], outputs=[RobotOutputs()])


def model_transforms(config, *, tokenizer=None):
    """在归一化后编码任务状态并将状态动作补至 32 维。

    输入：config、tokenizer。
    输出：模型输入变换 Group。"""
    from openpi import transforms
    from openpi.models.tokenizer import PaligemmaTokenizer

    if not config.pi05 or config.action_dim != 32:
        raise ValueError("This adapter targets pi05 with action_dim=32")
    if tokenizer is None:
        raise ValueError("请显式提供使用本地 tokenizer.model 创建的分词器")
    return transforms.Group(
        inputs=[
            transforms.TokenizePrompt(tokenizer, discrete_state_input=config.discrete_state_input),
            transforms.PadStatesAndActions(config.action_dim),
        ]
    )


def transform_chains(config, norm_stats, *, tokenizer=None, prompt=DEFAULT_PROMPT):
    """核对统计量并按训练与推理一致的顺序组合变换。

    输入：config、norm_stats、tokenizer、prompt。
    输出：输入变换列表和输出变换列表。"""
    from openpi import transforms

    if norm_stats is None or set(norm_stats) != {"state", "actions"}:
        raise ValueError("Provide state and actions statistics computed after RobotInputs")
    for name, stats in norm_stats.items():
        for field in ("mean", "std", "q01", "q99"):
            value = np.asarray(getattr(stats, field))
            if value.shape != (14,) or not np.isfinite(value).all():
                raise ValueError(f"{name}.{field} must contain 14 finite numbers")
        if np.any(stats.q99 < stats.q01) or np.any(stats.std < 0):
            raise ValueError(f"Invalid statistics for {name}")
    robot = data_transforms(prompt)
    model = model_transforms(config, tokenizer=tokenizer)
    inputs = [*robot.inputs, transforms.Normalize(norm_stats, use_quantiles=True), *model.inputs]
    outputs = [*model.outputs, transforms.Unnormalize(norm_stats, use_quantiles=True), *robot.outputs]
    return (inputs, outputs)
