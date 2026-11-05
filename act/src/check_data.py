"""检查真实 HDF5 数据、完整片段划分和训练集归一化统计，不生成合成数据。"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .config import PROJECT_ROOT, load_config
from .dataset import DatasetAssets, EpisodeDataset


def check(config, deep=False):
    """检查训练输入；输入配置和深度检查选项，输出汇总；深度模式读全部图像并复核训练集统计。"""
    assets = DatasetAssets(config['assets_dir'])
    model = config['model']
    summary = dict(train_episodes=len(assets.splits['train']),
                   validation_episodes=len(assets.splits['validation']),
                   asset_fingerprint=assets.fingerprint, data_dir=config['data_dir'], deep=deep, splits={})
    count = 0
    sums = {key: np.zeros(14, np.float64) for key in ('qpos', 'action')}
    squares = {key: np.zeros(14, np.float64) for key in sums}
    for split in ('train', 'validation'):
        dataset = EpisodeDataset(config['data_dir'], assets.splits[split], assets,
                                 model['camera_names'], model['chunk_size'], config['data']['image_size_hw'])
        summary['splits'][split] = dict(episodes=len(dataset.files), frames=len(dataset))
        if deep:
            for name in dataset.files:
                with h5py.File(Path(config['data_dir']) / name, 'r') as data:
                    timestamps = data['timestamp_ns'][:]
                    if len(timestamps) != len(data['action']) or not np.all(np.diff(timestamps) > 0):
                        raise ValueError(f'时间戳长度错误或不严格递增：{name}')
                    if not np.all(np.abs(np.diff(timestamps) / 1e9 - 1 / 30) < 1e-9):
                        raise ValueError(f'时间网格不是 30 Hz：{name}')
                    for key, path in [('qpos', 'observations/qpos'), ('action', 'action')]:
                        values = data[path][:].astype(np.float64)
                        if not np.isfinite(values).all():
                            raise ValueError(f'存在非有限数值：{name}/{key}')
                        if np.any(values[:, [6, 13]] < 0) or np.any(values[:, [6, 13]] > 1):
                            raise ValueError(f'夹爪不在 [0,1]：{name}/{key}')
                        if split == 'train':
                            sums[key] += values.sum(axis=0)
                            squares[key] += np.square(values).sum(axis=0)
                    if split == 'train':
                        count += len(timestamps)
                    for camera in model['camera_names']:
                        images = data['observations/images/' + camera]
                        for index in range(len(images)):
                            images[index]
        first, last = dataset[0], dataset[len(dataset) - 1]
        if not last['is_pad'][1:].all() or last['is_pad'][0]:
            raise ValueError('动作块末尾填充错误')
        summary['splits'][split]['sample_shapes'] = {key: list(value.shape) for key, value in first.items()}
    if deep:
        floor = assets.norm.get('std_floor')
        if floor is None:
            conversion = json.loads((assets.directory / 'conversion_config.json').read_text(encoding='utf-8'))
            floor = conversion['normalization_std_floor']
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError('归一化标准差下限无效')
        for key in sums:
            mean = sums[key] / count
            std = np.sqrt(np.maximum(0, squares[key] / count - mean ** 2))
            np.testing.assert_allclose(mean, assets.norm[key + '_mean'], atol=1e-6, rtol=0)
            np.testing.assert_allclose(np.maximum(std, floor), assets.norm[key + '_std'], atol=1e-6, rtol=0)
        summary['training_only_normalization_verified'] = True
    summary['passed'] = True
    return summary


def main():
    """执行数据检查命令；输入命令行参数，输出终端摘要及可选 JSON 报告。"""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/train.json')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--assets-dir', type=Path)
    parser.add_argument('--deep', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ('data_dir', 'assets_dir'):
        if getattr(args, key):
            config[key] = str(getattr(args, key).resolve())
    result = check(config, args.deep)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
