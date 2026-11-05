"""读取本地 LeRobot 数据、构造不跨片段的动作块，并按训练步数恢复确定性采样。"""

import json
from pathlib import Path
import numpy as np
from .adapter import CAMERAS, JOINTS
from .config import digest


def manifest(root, *, verify_hashes=False):
    """读取本地数据清单并核对路径及可选文件散列。

    输入：root、verify_hashes。
    输出：数据清单字典。"""
    root = Path(root)
    value = json.loads((root / "i2rt_manifest.json").read_text())
    if value["schema_version"] != 1 or value["split"] != "all_train" or value["fps"] != 30:
        raise ValueError("Unsupported dataset manifest")
    if value["cameras"] != list(CAMERAS) or value["image_size"] != [224, 224]:
        raise ValueError("Dataset must contain the two 224x224 wrist views")
    for name, expected in value["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe dataset file: {name}")
        if verify_hashes and digest(path) != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    return value


def open_dataset(root, horizon=None):
    """按固定 LeRobot 接口打开本地数据及未来动作窗口。

    输入：root、horizon。
    输出：LeRobotDataset 对象。"""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    info = manifest(root)
    dataset = LeRobotDataset(
        info["repo_id"],
        root=root,
        delta_timestamps=None if horizon is None else {"action": [i / info["fps"] for i in range(horizon)]},
    )
    if dataset.num_frames != info["total_frames"] or dataset.num_episodes != info["total_episodes"]:
        raise ValueError("LeRobot dataset size disagrees with manifest")
    return dataset


def numeric_episodes(root):
    """逐片段读取 Parquet 数值列并核对 14 维形状。

    输入：root。
    输出：依次产出状态数组与绝对目标数组。"""
    import pyarrow.parquet as pq

    info = manifest(root)
    paths = sorted((k for k in info["files"] if k.endswith(".parquet")))
    if len(paths) != info["total_episodes"]:
        raise ValueError("每个 episode 应有且仅有一个 Parquet 文件")
    for name in paths:
        table = pq.read_table(Path(root) / name, columns=["observation.state", "action", "episode_index"])
        state = np.asarray(table["observation.state"].to_pylist(), np.float32)
        actions = np.asarray(table["action"].to_pylist(), np.float32)
        if (
            len(set(table["episode_index"].to_pylist())) != 1
            or state.shape != actions.shape
            or state.shape[1:] != (14,)
        ):
            raise ValueError(f"Invalid episode table {name}")
        yield (state, actions)


def action_chunks(state, actions, horizon, block_size=256):
    """按观测时刻取未来动作，尾部重复末帧且只对关节作差。

    输入：state/actions 为 [T,14] 数组，horizon 为未来步数，block_size 为内存分块大小。
    输出：分块产出当前状态和相对关节动作块。"""
    for start in range(0, len(state), block_size):
        idx = np.arange(start, min(start + block_size, len(state)))
        indices = np.minimum(idx[:, None] + np.arange(horizon), len(actions) - 1)
        chunks = actions[indices].copy()
        chunks[..., JOINTS] -= state[idx][:, None, JOINTS]
        yield (state[idx], chunks)


def load_stats(settings):
    """核对统计来源、动作跨度和转换约定后读取统计量。

    输入：settings。
    输出：state 与 actions 的 NormStats 字典。"""
    from openpi.shared import normalize

    provenance = json.loads((Path(settings.assets_dir) / "provenance.json").read_text())
    if (
        provenance["dataset_manifest_sha256"] != digest(Path(settings.dataset_root) / "i2rt_manifest.json")
        or provenance["action_horizon"] != settings.action_horizon
        or provenance["fps"] != settings.fps
    ):
        raise ValueError("Statistics do not match this dataset/action horizon/fps")
    stats = normalize.load(settings.assets_dir)
    if set(stats) != {"state", "actions"}:
        raise ValueError("统计量必须同时包含 state 和 actions")
    for name, group in stats.items():
        for field in ("mean", "std", "q01", "q99"):
            value = np.asarray(getattr(group, field))
            if value.shape != (14,) or not np.isfinite(value).all():
                raise ValueError(f"{name}.{field} 必须是 14 维有限数值")
        if np.any(group.q99 < group.q01) or np.any(group.std < 0):
            raise ValueError(f"{name} 统计量顺序或标准差错误")
    if (
        provenance.get("split") != "all_train"
        or provenance.get("tail_policy") != "repeat_last_action"
        or provenance.get("actions") != "relative_to_current_state_except_absolute_grippers"
    ):
        raise ValueError("统计量与当前动作转换约定不一致")
    return stats


class TrainingDataset:

    def __init__(self, settings, model_config, *, tokenizer=None):
        """打开本地数据并绑定分词器及输入转换链。

        输入：settings、model_config、tokenizer。
        输出：无返回值，初始化训练数据读取器。"""
        from openpi import transforms
        from .adapter import transform_chains

        self.dataset = open_dataset(settings.dataset_root, settings.action_horizon)
        if tokenizer is None:
            from openpi.models.tokenizer import PaligemmaTokenizer

            tokenizer = PaligemmaTokenizer(model_config.max_token_len, model_path=settings.tokenizer_path)
        inputs, _ = transform_chains(
            model_config, load_stats(settings), tokenizer=tokenizer, prompt=settings.prompt
        )
        self.transform = transforms.compose(inputs)

    def __len__(self):
        """取得训练数据总观测帧数。

        输入：无外部参数，读取当前对象配置。
        输出：整数帧数。"""
        return len(self.dataset)

    def __getitem__(self, index):
        """读取指定真实帧及动作块并完成模型输入变换。

        输入：index 为当前观测在整个数据集中的帧索引。
        输出：归一化、编码和补维后的样本字典。"""
        item = self.dataset[int(index)]
        images = {}
        for camera in CAMERAS:
            image = item["observation.images." + camera].numpy()
            if (
                image.shape != (3, 224, 224)
                or not np.isfinite(image).all()
                or image.min() < 0
                or (image.max() > 1)
            ):
                raise ValueError("Unexpected LeRobot image representation")
            images[camera] = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
        return self.transform(
            dict(
                images=images,
                state=item["observation.state"].numpy(),
                actions=item["action"].numpy(),
                prompt=item["task"],
            )
        )


class StepBatchSampler:

    def __init__(self, length, batch_size, seed, start_step, stop_step):
        """设定数据长度、随机种子及起止训练步数。

        输入：length、batch_size、seed、start_step、stop_step。
        输出：无返回值，初始化采样器。"""
        if length < 1 or start_step < 0 or stop_step < start_step:
            raise ValueError("Invalid sampling bounds")
        self.length, self.batch_size, self.seed = (length, batch_size, seed)
        self.start_step, self.stop_step = (start_step, stop_step)

    def __len__(self):
        """取得待生成批次数。

        输入：无外部参数，读取当前对象配置。
        输出：结束步数减开始步数。"""
        return self.stop_step - self.start_step

    def __iter__(self):
        """按种子和遍历轮次恢复确定性乱序采样。

        输入：无外部参数，读取当前对象配置。
        输出：逐步产出全局批次的样本索引列表。"""
        epoch_cached, order = (None, None)
        for step in range(self.start_step, self.stop_step):
            batch = []
            for cursor in range(step * self.batch_size, (step + 1) * self.batch_size):
                epoch, index = divmod(cursor, self.length)
                if epoch != epoch_cached:
                    order = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(
                        self.length
                    )
                    epoch_cached = epoch
                batch.append(int(order[index]))
            yield batch


def collate_numpy(samples):
    """逐字段堆叠同一批次的样本。

    输入：samples。
    输出：NumPy 批次树。"""
    import jax

    return jax.tree.map(lambda *values: np.stack(values), *samples)


def batches(dataset, settings, start_step, stop_step, sharding):
    """通过数据工作进程读取批次并放置到 JAX 分片。

    输入：dataset、settings、start_step、stop_step、sharding。
    输出：逐步产出 Observation 和动作张量。"""
    import jax
    import torch
    from openpi.models.model import Observation

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=StepBatchSampler(
            len(dataset), settings.batch_size, settings.seed, start_step, stop_step
        ),
        num_workers=settings.num_workers,
        collate_fn=collate_numpy,
        multiprocessing_context="spawn" if settings.num_workers else None,
    )
    for data in loader:
        data = jax.tree.map(lambda x: jax.make_array_from_process_local_data(sharding, x), data)
        yield (Observation.from_dict(data), data["actions"])
