"""异步保存和恢复 Orbax 训练状态，分离 EMA 推理权重并保存归一化资源。"""

from __future__ import annotations
import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol
from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future
from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.utils as training_utils


class AssetsProvider(Protocol):

    def data_config(self):
        """返回统计量保存描述；输入：无；输出：含 norm_stats、asset_id 的对象。"""
        ...


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    """创建或恢复异步检查点管理器。

    输入：checkpoint_dir、keep_period、overwrite、resume。
    输出：CheckpointManager 及是否恢复的标记。"""
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume to indicate how to handle it."
            )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False
    return (mngr, resuming)


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: AssetsProvider,
    step: int,
):
    """分离 EMA 推理权重并异步保存训练状态和资源。

    输入：checkpoint_manager、state、data_loader、step。
    输出：无返回值，向管理器提交写入任务。"""

    def save_assets(directory: epath.Path):
        """将本次训练归一化统计写入资源目录。

        输入：directory。
        输出：无返回值，生成归一化文件。"""
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {"assets": save_assets, "train_state": train_state, "params": {"params": params}}
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: AssetsProvider,
    step: int | None = None,
) -> training_utils.TrainState:
    """恢复优化器、步数、模型参数及 EMA。

    输入：checkpoint_manager、state、data_loader、step。
    输出：可继续训练的 TrainState。"""
    del data_loader
    with at.disable_typechecking():
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step, items={"train_state": train_state, "params": {"params": params}}
        )
    return _merge_params(restored["train_state"], restored["params"])


class Callback(Protocol):

    def __call__(self, directory: epath.Path) -> None:
        """约定检查点资源保存回调。

        输入：directory。
        输出：无返回值，在目标目录写入资源。"""
        ...


class CallbackHandler(ocp.AsyncCheckpointHandler):

    def save(self, directory: epath.Path, args: CallbackSave):
        """在主进程执行资源保存回调。

        输入：directory、args。
        输出：无返回值。"""
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        """异步执行资源回调并关联提交信号。

        输入：directory、args。
        输出：供 Orbax 等待的 Future 列表。"""
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        """拒绝恢复仅用于保存的回调项。

        输入：*args、**kwargs。
        输出：始终抛出未实现异常。"""
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    """将推理 EMA 权重从训练状态分离。

    输入：state。
    输出：训练状态和推理参数。"""
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return (train_state, params)


def _merge_params(
    train_state: training_utils.TrainState, params: dict[str, at.Params]
) -> training_utils.TrainState:
    """将分离保存的参数重新装入训练状态。

    输入：train_state、params。
    输出：恢复 EMA 或主参数的 TrainState。"""
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
