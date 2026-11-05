"""原子保存模型及训练状态，恢复随机数并核对续训配置。"""
import os
from pathlib import Path
import random
import numpy as np
import torch

def atomic_save(value, path):
    """先写临时文件再替换目标检查点。

    输入：value、path。
    输出：无；生成目标文件。
    """
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)

def rng_state():
    """收集 Python、NumPy、PyTorch 及 CUDA 随机状态。

    输入：无。
    输出：随机状态字典。
    """
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    """恢复检查点保存的随机状态。

    输入：state。
    输出：无；修改当前进程随机数生成器。
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])

def resume_signature(config):
    """提取续训必须保持一致的模型、数据和优化参数。

    输入：config。
    输出：用于一致性比较的字典。
    """
    model = {k: v for k, v in config['model'].items() if k != 'pretrained_backbone'}
    keys = ('strategy', 'seed', 'batch_size', 'lr', 'lr_backbone', 'weight_decay', 'val_batch_size', 'max_val_batches')
    return dict(model=model, data=config['data'], training={k: config['training'][k] for k in keys})
