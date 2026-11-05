"""执行 ACT 训练和验证，保存最佳权重，支持断点续训与权重初始化。"""
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from .checkpoint import atomic_save, rng_state, restore_rng, resume_signature
from .config import PROJECT_ROOT, load_config, stable_hash
from .dataset import DatasetAssets, EpisodeDataset, EpochBatchSampler
from .model import ACTPolicy, make_optimizer

def seed_all(seed):
    """设置 Python、NumPy、PyTorch 和 CUDA 随机种子。

    输入：seed。
    输出：无；修改全局随机状态。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    """将批次中的张量移动到指定 CPU 或 GPU。

    输入：batch、device。
    输出：位于目标设备的批次字典。
    """
    return {k: v.to(device, non_blocking=device.type == 'cuda') for k, v in batch.items()}

@torch.no_grad()
def evaluate(policy, loader, device, max_batches=None):
    """禁用梯度并对验证批次计算平均损失。

    输入：policy、loader、device、max_batches。
    输出：loss、l1、kl 指标字典。
    """
    policy.eval()
    rows = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        b = to_device(batch, device)
        out = policy(b['qpos'], b['image'], b['actions'], b['is_pad'])
        rows.append({key: out[key].item() for key in ('loss', 'l1', 'kl')})
    if not rows:
        raise ValueError('Validation produced no samples')
    return {key: sum((row[key] for row in rows)) / len(rows) for key in rows[0]}

def build_loaders(config, assets):
    """按完整片段划分构建训练及验证加载器，并计算数据身份。

    输入：config、assets。
    输出：训练加载器、验证加载器及数据指纹。
    """
    m, t = (config['model'], config['training'])
    loaders = []
    identities = {}
    for split, tag, batch_size in [('train', 0, t['batch_size']), ('validation', 1, t['val_batch_size'])]:
        ds = EpisodeDataset(config['data_dir'], assets.splits[split], assets, m['camera_names'], m['chunk_size'], config['data']['image_size_hw'])
        sampler = EpochBatchSampler(ds, batch_size, t['seed'], tag)
        kwargs = dict(num_workers=t['num_workers'], pin_memory=t['device'].startswith('cuda'))
        if kwargs['num_workers'] > 0:
            kwargs.update(prefetch_factor=1, multiprocessing_context='spawn')
        loaders.append(DataLoader(ds, batch_sampler=sampler, generator=torch.Generator().manual_seed(t['seed'] + tag), **kwargs))
        identities[split] = ds.source_identities
    return (*loaders, stable_hash(dict(assets=assets.fingerprint, **identities)))

def run_training(config, output, resume=None, init_from=None):
    """执行训练、验证、保存及可选续训，不覆盖无关训练目录。

    输入：config、output、resume、init_from。
    输出：训练完成摘要并写入日志和检查点。
    """
    if resume and init_from:
        raise ValueError('resume and init-from are mutually exclusive')
    output = Path(output).resolve()
    t = config['training']
    if t.get('strategy') != 'official_epoch':
        raise ValueError('Expected official_epoch training configuration')
    if min(t['num_epochs'], t['batch_size'], t['val_batch_size'], t['save_interval_epochs']) < 1:
        raise ValueError('Epoch count, batch sizes and save interval must be positive')
    device = torch.device(t['device'])
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable; use --device cpu for local checks')
    torch.set_num_threads(t['torch_threads'])
    seed_all(t['seed'])
    assets = DatasetAssets(config['assets_dir'])
    saved = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if saved and saved.get('schema_version') != 2:
        raise ValueError('Legacy step-based checkpoint cannot resume epoch training; use --init-from with weights in a new run')
    start_epoch = saved['next_epoch'] if saved else 0
    if saved and saved['resume_signature'] != resume_signature(config):
        raise ValueError('Exact resume configuration changed; use --init-from for a new run')
    if start_epoch >= t['num_epochs']:
        raise ValueError('num_epochs must exceed completed epochs')
    if output.exists() and any(output.iterdir()) and (not resume):
        raise FileExistsError('Run directory is not empty; select a new output or use --resume')
    train_loader, val_loader, fingerprint = build_loaders(config, assets)
    if saved and saved['dataset_fingerprint'] != fingerprint:
        raise ValueError('Dataset/splits/normalization changed')
    policy = ACTPolicy(config['model'], pretrained=False if resume or init_from else None).to(device)
    optimizer = make_optimizer(policy, t)
    best = math.inf
    best_epoch = None
    best_model = None
    step = 0
    if saved:
        policy.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        best = saved['best_metric']
        best_epoch = saved['best_epoch']
        best_model = saved['best_model']
        step = saved['step']
        restore_rng(saved['rng'])
    elif init_from:
        policy.load_state_dict(torch.load(init_from, map_location='cpu', weights_only=True), strict=True)
    output.mkdir(parents=True, exist_ok=True)
    if saved and (output / 'metrics.jsonl').exists():
        rows = []
        for line in (output / 'metrics.jsonl').read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('epoch', start_epoch) < start_epoch:
                rows.append(row)
        temp = output / 'metrics.jsonl.tmp'
        temp.write_text(''.join((json.dumps(row) + '\n' for row in rows)))
        os.replace(temp, output / 'metrics.jsonl')
    if (output / 'summary.json').exists():
        (output / 'summary.json').unlink()
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    asset_output = output / 'dataset_assets'
    asset_output.mkdir(exist_ok=True)
    for p in Path(config['assets_dir']).iterdir():
        if p.is_file():
            shutil.copyfile(p, asset_output / p.name)
    shutil.copyfile(Path(config['assets_dir']) / 'dataset_stats.pkl', output / 'dataset_stats.pkl')
    provenance = dict(
        python=platform.python_version(), torch=torch.__version__,
        device=str(device), cuda_runtime=torch.version.cuda,
        packages={name: package_version(name)
                  for name in ('torch', 'torchvision', 'numpy', 'h5py', 'pillow')},
        dataset_fingerprint=fingerprint,
        source_sha256={
            str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (PROJECT_ROOT / 'src', PROJECT_ROOT / 'vendor')
            for p in directory.rglob('*.py')
        },
        training_strategy='official_epoch', gradient_clipping=False,
        initial_weights=str(init_from) if init_from else None,
        resumed_from=str(resume) if resume else None,
    )
    (output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    if best_model is not None:
        atomic_save(best_model, output / 'policy_best.ckpt')

    def save_state(epoch):
        """保存已完成轮次的权重、优化器、最佳模型和随机状态。

        输入：epoch。
        输出：无；写入最近权重和续训检查点。
        """
        atomic_save(policy.state_dict(), output / 'policy_last.ckpt')
        atomic_save(dict(schema_version=2, next_epoch=epoch + 1, step=step, model=policy.state_dict(), optimizer=optimizer.state_dict(), best_metric=best, best_epoch=best_epoch, best_model=best_model, rng=rng_state(), config=config, resume_signature=resume_signature(config), dataset_fingerprint=fingerprint), output / 'resume.ckpt')
    try:
        for epoch in range(start_epoch, t['num_epochs']):
            val_loader.batch_sampler.set_epoch(epoch)
            val = evaluate(policy, val_loader, device, t['max_val_batches'])
            if not all((math.isfinite(v) for v in val.values())):
                raise FloatingPointError('Nonfinite validation metric')
            if val['loss'] < best:
                best = val['loss']
                best_epoch = epoch
                best_model = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                atomic_save(best_model, output / 'policy_best.ckpt')
            train_loader.batch_sampler.set_epoch(epoch)
            policy.train()
            rows = []
            for batch in train_loader:
                b = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                values = policy(b['qpos'], b['image'], b['actions'], b['is_pad'])
                if not torch.isfinite(values['loss']):
                    raise FloatingPointError('Nonfinite training loss')
                values['loss'].backward()
                optimizer.step()
                step += 1
                rows.append({k: values[k].detach().item() for k in ('loss', 'l1', 'kl')})
            train = {k: sum((row[k] for row in rows)) / len(rows) for k in rows[0]}
            event = dict(epoch=epoch, step=step, train=train, validation=val)
            with (output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(event) + '\n')
            print(json.dumps(event), flush=True)
            if epoch % t['save_interval_epochs'] == 0 or epoch == t['num_epochs'] - 1:
                save_state(epoch)
    except KeyboardInterrupt:
        print('Interrupted; resume from the latest completed epoch checkpoint.', flush=True)
        raise
    summary = dict(completed_epochs=t['num_epochs'], step=step, best_epoch=best_epoch, best_metric=best, selection_metric='loss', dataset_fingerprint=fingerprint, output=str(output))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary

def main():
    """解析训练命令行参数并执行正式训练。

    输入：无。
    输出：无；终端输出训练进度和完成摘要。
    """
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/train.json')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--assets-dir', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--device')
    p.add_argument('--num-epochs', type=int)
    p.add_argument('--num-workers', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--max-val-batches', type=int)
    p.add_argument('--no-pretrained-backbone', action='store_true')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--resume', type=Path)
    modes.add_argument('--init-from', type=Path)
    a = p.parse_args()
    c = load_config(a.config)
    for key in ('data_dir', 'assets_dir'):
        if getattr(a, key) is not None:
            c[key] = str(getattr(a, key).resolve())
    for key in ('device', 'num_epochs', 'num_workers', 'batch_size', 'max_val_batches'):
        if getattr(a, key) is not None:
            c['training'][key] = getattr(a, key)
    if a.no_pretrained_backbone:
        c['model']['pretrained_backbone'] = False
    output = a.output or (a.resume.resolve().parent if a.resume else PROJECT_ROOT / 'output/runs' / (c['name'] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    print('COMPLETE', json.dumps(run_training(c, output, a.resume, a.init_from)), flush=True)
if __name__ == '__main__':
    main()
