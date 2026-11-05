"""读取训练配置、定位子项目目录并计算数据与配置指纹。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(path):
    """读取 JSON 并检查模型和数据参数；将数据路径解析为相对子项目的路径。

    输入：path。
    输出：解析后的配置字典。
    """
    config = json.loads(Path(path).read_text())
    config = deepcopy(config)
    for key in ('data_dir', 'assets_dir'):
        p = Path(config[key]).expanduser()
        config[key] = str((PROJECT_ROOT / p if not p.is_absolute() else p).resolve())
    if config['model']['state_dim'] != 14 or config['model']['action_dim'] != 14:
        raise ValueError('This I2RT ACT network requires 14D state/action')
    if config['model']['chunk_size'] < 1:
        raise ValueError('chunk_size must be positive')
    if config['model']['hidden_dim'] % config['model']['nheads']:
        raise ValueError('hidden_dim must be divisible by nheads')
    return config

def stable_hash(value):
    """计算可序列化对象的稳定 SHA256。

    输入：value。
    输出：十六进制摘要字符串。
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
