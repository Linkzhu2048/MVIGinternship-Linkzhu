"""执行 JAX 全量微调，保留官方初始化、流匹配损失、AdamW 更新和 EMA，并保存可恢复检查点。"""

import dataclasses
import functools
import json
import logging
from pathlib import Path
import shutil
import time
from .config import Settings, TrainingConfig, PROJECT_ROOT, setup_caches, write_json
from .checkpoint import run_identity
from typing import Any
from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
import optax
from openpi.models import model as _model
from openpi.shared import array_typing as at, nnx_utils
from openpi.training import optimizer as _optimizer, weight_loaders as _weight_loaders, sharding
from openpi.training import utils as training_utils


def train_config(settings):
    """按用户设置构建官方优化器及双腕模型配置。

    输入：settings。
    输出：全量微调 TrainingConfig。"""
    from .config import TrainingConfig
    from openpi.training import optimizer, weight_loaders
    from .config import DualWristPi05Config

    model_config = DualWristPi05Config(action_horizon=settings.action_horizon)
    return TrainingConfig(
        model=model_config,
        weight_loader=weight_loaders.CheckpointWeightLoader(settings.pretrained_params),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=settings.warmup_steps,
            peak_lr=settings.learning_rate,
            decay_steps=settings.decay_steps,
            decay_lr=settings.end_learning_rate,
        ),
        optimizer=optimizer.AdamW(),
        ema_decay=settings.ema_decay,
    )


class AssetsProvider:

    def __init__(self, stats):
        """保存供检查点写入的归一化统计。

        输入：stats。
        输出：无返回值，初始化资源提供器。"""
        self.stats = stats

    def data_config(self):
        """取得检查点所需的统计量及资源标识。

        输入：无外部参数，读取当前对象配置。
        输出：包含 norm_stats 和 asset_id 的对象。"""
        from types import SimpleNamespace

        return SimpleNamespace(norm_stats=self.stats, asset_id="i2rt")


def run(settings, *, resume=False):
    """执行正式 GPU 训练或从完整检查点恢复。

    输入：settings、resume。
    输出：完成步数和性能报告，同时写日志与检查点。"""
    import filelock
    import jax
    import numpy as np
    from openpi.training import checkpoints, sharding
    from .dataset import TrainingDataset, batches, load_stats

    logging.basicConfig(level=logging.INFO)
    if jax.process_count() != 1:
        raise ValueError("Only single-host 1-2 GPU training is supported")
    if any((x.platform != "gpu" for x in jax.devices())):
        raise RuntimeError("正式训练需要 GPU；请检查 JAX CUDA 环境")
    if len(jax.devices()) != settings.expected_devices:
        raise ValueError("Visible device count differs from expected_devices; set CUDA_VISIBLE_DEVICES")
    config = train_config(settings)
    identity = run_identity(settings)
    directory = Path(settings.run_dir)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(directory) + ".lock", timeout=0):
        if resume:
            saved = json.loads((directory / "run.json").read_text())
            if saved != identity:
                raise ValueError("Resume refused: config, dataset, statistics or source code changed")
        else:
            directory.mkdir(exist_ok=False)
            write_json(directory / "run.json", identity)
            shutil.copy2(
                Path(settings.dataset_root) / "i2rt_manifest.json", directory / "dataset_manifest.json"
            )
            shutil.copy2(Path(settings.assets_dir) / "provenance.json", directory / "norm_provenance.json")
        mesh = sharding.make_mesh(settings.fsdp_devices)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        rng, init_rng = jax.random.split(jax.random.key(settings.seed))
        manager, _ = checkpoints.initialize_checkpoint_dir(
            directory / "checkpoints", keep_period=settings.keep_period, overwrite=False, resume=resume
        )
        try:
            has_checkpoint = manager.latest_step() is not None
            if resume and (not has_checkpoint):
                raise ValueError("No completed checkpoint to resume; start a new run directory")
            state, state_sharding = init_train_state(config, init_rng, mesh, resume=has_checkpoint)
            assets = AssetsProvider(load_stats(settings))
            if has_checkpoint:
                state = checkpoints.restore_state(manager, state, assets)
            jax.block_until_ready(state)
            start = int(state.step)
            stop = settings.num_steps
            if start >= stop:
                return {"completed_steps": start, "message": "Requested training already complete"}
            dataset = TrainingDataset(settings, config.model)
            shutil.copy2(settings.tokenizer_path, directory / "tokenizer.model")
            compiled_step = jax.jit(
                functools.partial(train_step, config),
                in_shardings=(replicated, state_sharding, data_sharding),
                out_shardings=(state_sharding, replicated),
                donate_argnums=(1,),
            )
            iterator = iter(batches(dataset, settings, start, stop, data_sharding))
            measured = []
            with (directory / "metrics.jsonl").open("a") as log:
                for step in range(start, stop):
                    began = time.monotonic()
                    batch = next(iterator)
                    loaded = time.monotonic()
                    with sharding.set_mesh(mesh):
                        state, metrics = compiled_step(rng, state, batch)
                    jax.block_until_ready(state)
                    finished = time.monotonic()
                    values = {key: float(value) for key, value in jax.device_get(metrics).items()}
                    if not all(np.isfinite(list(values.values()))):
                        raise FloatingPointError(f"Nonfinite training metrics at update {step + 1}")
                    record = dict(
                        step=step + 1,
                        **values,
                        data_seconds=loaded - began,
                        compute_seconds=finished - loaded,
                        total_seconds=finished - began,
                    )
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    if step - start >= settings.benchmark_warmup:
                        measured.append(record["total_seconds"])
                    if (step + 1) % settings.log_interval == 0 or step == start:
                        print(json.dumps(record), flush=True)
                    if (step + 1) % settings.save_interval == 0 or step + 1 == stop:
                        checkpoints.save_state(manager, state, assets, step + 1)
                manager.wait_until_finished()
            report = dict(
                completed_steps=int(state.step),
                measured_steps=len(measured),
                devices=[str(x) for x in jax.devices()],
                global_batch_size=settings.batch_size,
                fsdp_devices=settings.fsdp_devices,
                timing_scope="Steady step data+compute; excludes initialization, warmup and checkpoint writes",
            )
            if measured:
                report.update(
                    mean_step_seconds=float(np.mean(measured)),
                    median_step_seconds=float(np.median(measured)),
                    samples_per_second=settings.batch_size / float(np.mean(measured)),
                    estimated_remaining_hours=(settings.num_steps - int(state.step))
                    * float(np.mean(measured))
                    / 3600,
                )
            write_json(directory / "performance.json", report)
            return report
        finally:
            manager.close()


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """读取预训练权重并严格核对参数形状和类型。

    输入：loader、params_shape。
    输出：去除占位形状后的已加载参数树。"""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )


@at.typecheck
def init_train_state(
    config: TrainingConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """构建模型、优化器和 EMA 并加载基础权重。

    输入：config、init_rng、mesh、resume。
    输出：训练状态及对应分片，恢复模式只返回形状。"""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        """创建网络、混入基础参数并初始化优化器与 EMA。

        输入：rng、partial_params。
        输出：TrainState 对象。"""
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    if resume:
        return (train_state_shape, state_sharding)
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init, donate_argnums=(1,), in_shardings=replicated_sharding, out_shardings=state_sharding
    )(init_rng, partial_params)
    return (train_state, state_sharding)


@at.typecheck
def train_step(
    config: TrainingConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """执行官方流匹配损失、梯度更新和 EMA 更新。

    输入：config、rng、state、batch。
    输出：下一步训练状态与损失、梯度范数、参数范数。"""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        """对动作块逐步流匹配误差求均值。

        输入：model、rng、observation、actions。
        输出：标量训练损失。"""
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return (new_state, info)


def main():
    """启动或恢复正式训练；输入：命令行配置与恢复标记；输出：日志和 Orbax 检查点。"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/train.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    setup_caches(PROJECT_ROOT / "output/cache")
    print(json.dumps(run(Settings.load(args.config), resume=args.resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
