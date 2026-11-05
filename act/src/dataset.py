"""读取已对齐的 HDF5，执行归一化、动作块填充和按片段采样。"""
import bisect
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from .config import stable_hash

class DatasetAssets:
    """读取并校验数据清单、完整片段划分和训练集归一化统计。"""

    def __init__(self, directory):
        """从目录读取清单、划分和统计，拒绝训练/验证重叠与无效参数。

        输入：directory。
        输出：数据资产实例及其指纹。
        """
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'dataset_manifest.json').read_text())
        self.splits = json.loads((self.directory / 'splits.json').read_text())
        self.norm = json.loads((self.directory / 'normalization.json').read_text())
        train, val = (self.splits['train'], self.splits['validation'])
        if not train or not val or len(set(train)) != len(train) or (len(set(val)) != len(val)):
            raise ValueError('Splits must be nonempty and contain no duplicate episode files')
        if set(train) & set(val):
            raise ValueError('Episode leakage between train and validation')
        expected = {x['file'] for x in self.manifest['episodes']}
        if len(expected) != len(self.manifest['episodes']) or set(train + val) != expected:
            raise ValueError('Split coverage does not match dataset manifest')
        for name in expected:
            if Path(name).name != name or not name.endswith('.hdf5'):
                raise ValueError('Unsafe episode filename')
        for key in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std'):
            v = np.asarray(self.norm[key], np.float32)
            if v.shape != (14,) or not np.isfinite(v).all():
                raise ValueError('Invalid normalization statistics')
            if key.endswith('_std') and np.any(v <= 0):
                raise ValueError('Normalization std must be positive')
        if self.norm.get('computed_from') != 'training episodes only':
            raise ValueError('Normalization must be computed from training episodes only')
        self.fingerprint = stable_hash(dict(manifest=self.manifest, splits=self.splits, norm=self.norm))

class EpisodeDataset(Dataset):
    """把一个观察时刻及其后续动作块组织成训练样本。"""

    def __init__(self, data_dir, files, assets, camera_names, chunk_size, image_size_hw=None):
        """核对每个 HDF5 的身份、尺寸和颜色，建立全局帧索引。

        输入：data_dir、files、assets、camera_names、chunk_size、image_size_hw。
        输出：可按帧读取的训练数据集。
        """
        self.data_dir = Path(data_dir)
        self.files = list(files)
        self.assets = assets
        self.camera_names = list(camera_names)
        self.chunk_size = chunk_size
        self.image_size_hw = tuple(image_size_hw) if image_size_hw else None
        self.lengths = []
        manifest = {x['file']: x for x in assets.manifest['episodes']}
        self.source_identities = []
        for name in self.files:
            with h5py.File(self.data_dir / name, 'r') as f:
                n = len(f['action'])
                if n != manifest[name]['frames'] or f.attrs.get('episode_id') != manifest[name]['id']:
                    raise ValueError(f'Source identity/length mismatch: {name}')
                if f.attrs.get('human_review') != 'pass':
                    raise ValueError(f'Episode not approved: {name}')
                if f.attrs.get('image_color_space') != 'RGB':
                    raise ValueError(f'Expected RGB: {name}')
                if f.attrs.get('action_time_offset_ms', None) != 0:
                    raise ValueError('Unsupported action offset')
                if f['observations/qpos'].shape != (n, 14) or f['action'].shape != (n, 14):
                    raise ValueError(f'Invalid numeric dimensions: {name}')
                if abs(float(f.attrs['sample_hz']) - 30) > 1e-06:
                    raise ValueError('Expected aligned 30Hz dataset')
                for camera in self.camera_names:
                    d = f['observations/images/' + camera]
                    if d.shape[0] != n or len(d.shape) != 4 or d.shape[-1] != 3 or (d.dtype != np.uint8):
                        raise ValueError(f'Invalid image array: {name}/{camera}')
                self.source_identities.append(dict(file=name, episode_id=f.attrs['episode_id'], frames=n, conversion_config_sha256=f.attrs['conversion_config_sha256'], raw_files_sha256=stable_hash(f['provenance/raw_files_json'].asstr()[()])))
                self.lengths.append(n)
        self.ends = np.cumsum(self.lengths).tolist()
        self.starts = [0] + self.ends[:-1]
        self.norm = {key: np.asarray(assets.norm[key], np.float32) for key in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std')}

    def __len__(self):
        """统计全部入选片段的帧数。

        输入：无。
        输出：总帧数。
        """
        return self.ends[-1]

    def locate(self, index):
        """把全局帧索引定位到所属片段及片段内位置。

        输入：index。
        输出：片段索引和帧索引。
        """
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self.ends, index)
        return (episode, index - self.starts[episode])

    def __getitem__(self, index):
        """读取两腕 RGB、状态及动作块，执行归一化并标记尾部填充。

        输入：index。
        输出：image、qpos、actions、is_pad 张量字典。
        """
        episode, start = self.locate(int(index))
        with h5py.File(self.data_dir / self.files[episode], 'r') as f:
            qpos = f['observations/qpos'][start].astype(np.float32)
            actions = f['action'][start:start + self.chunk_size].astype(np.float32)
            images = []
            for camera in self.camera_names:
                image = f['observations/images/' + camera][start]
                if self.image_size_hw and image.shape[:2] != self.image_size_hw:
                    h, w = self.image_size_hw
                    image = np.asarray(Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR))
                images.append(image)
        if not np.isfinite(qpos).all() or not np.isfinite(actions).all():
            raise ValueError('Nonfinite training sample')
        qpos = (qpos - self.norm['qpos_mean']) / self.norm['qpos_std']
        count = len(actions)
        padded = np.zeros((self.chunk_size, 14), np.float32)
        padded[:count] = (actions - self.norm['action_mean']) / self.norm['action_std']
        is_pad = np.arange(self.chunk_size) >= count
        image = np.stack(images).transpose(0, 3, 1, 2).astype(np.float32) / 255
        return dict(image=torch.from_numpy(image), qpos=torch.from_numpy(qpos), actions=torch.from_numpy(padded), is_pad=torch.from_numpy(is_pad))

class EpochBatchSampler(Sampler):
    """每轮打乱片段顺序，每个片段随机取一个起点，保留末尾不足批次。"""

    def __init__(self, dataset, batch_size, seed, split=0):
        """保存数据集、批量大小、随机种子及划分标识。

        输入：dataset、batch_size、seed、split。
        输出：片段级采样器实例。
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.split = split
        self.epoch = 0

    def set_epoch(self, epoch):
        """设置当前轮次以得到可复现的采样序列。

        输入：epoch。
        输出：无；更新采样轮次。
        """
        self.epoch = epoch

    def __len__(self):
        """按片段数量计算本轮批次数，保留不足批次。

        输入：无。
        输出：批次数。
        """
        return (len(self.dataset.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """每个片段取一个随机帧并按批量大小分组。

        输入：无。
        输出：依次产生全局帧索引列表。
        """
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.split]))
        order = rng.permutation(len(self.dataset.lengths))
        indices = [self.dataset.starts[i] + int(rng.integers(self.dataset.lengths[i])) for i in order]
        for start in range(0, len(indices), self.batch_size):
            yield indices[start:start + self.batch_size]
