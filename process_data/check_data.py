"""检查原始数据及导出结果，区分结构错误和需人工判断的画面/动作警告。"""
from pathlib import Path

import h5py
import numpy as np

from align_data import align_episode
from read_data import (CAMERAS, STREAMS, decode_image, digest, find_episodes,
                       object_digest, outside_source, read_episode, read_json, source_files, write_json)


def image_warnings(episode):
    """检查所有原始双腕帧；输入原始片段，输出黑白帧、低变化和连续重复画面的计数及首个位置。"""
    warnings = []
    for key in ('camera/left', 'camera/right'):
        counts, first, previous, same_start, freeze_found = {}, {}, None, None, False
        for i, (timestamp, payload) in enumerate(episode['raw'][key]):
            rgb = decode_image(payload, episode['image_sizes'][key])
            mean, std = float(rgb.mean()), float(rgb.std())
            kind = 'black' if mean <= 1 else 'white' if mean >= 254 else 'constant' if std < 1 else None
            if kind:
                counts[kind] = counts.get(kind, 0) + 1
                first.setdefault(kind, i)
            if previous is None or not np.array_equal(rgb, previous):
                same_start, freeze_found = timestamp, False
            elif timestamp - same_start >= 1_000_000_000 and not freeze_found:
                counts['frozen_over_1s'] = counts.get('frozen_over_1s', 0) + 1
                first.setdefault('frozen_over_1s', i)
                freeze_found = True
            previous = rgb
        warnings.extend({'stream': key, 'kind': kind, 'count': count, 'first_source_index': first[kind]}
                        for kind, count in counts.items())
    return warnings


def episode_warnings(episode, config):
    """汇总质量提示；输入原始片段和配置，输出短时长、画面异常及夹爪突变提示，不判断任务成功。"""
    duration = (episode['stats']['ended_at_ns'] - episode['stats']['started_at_ns'])/1e9
    warnings = image_warnings(episode)
    if duration < config['short_episode_seconds']:
        warnings.append({'kind': 'short_episode', 'duration_s': duration})
    for key, records in episode['raw'].items():
        if key.startswith('gripper/'):
            jumps = np.flatnonzero(np.abs(np.diff([row[1][0] for row in records])) > 0.5)
            if len(jumps):
                warnings.append({'stream': key, 'kind': 'gripper_jump_over_0.5',
                                 'count': len(jumps), 'first_source_index': int(jumps[0]+1)})
    return warnings


def audit_episode(path, root, config):
    """检查一个原始片段；输入路径、根目录和配置，输出结构错误与质量提示，不产生审核状态。"""
    source = path.relative_to(root).as_posix()
    record = {'source': source, 'group': path.parent.relative_to(root).as_posix() if path != root else '.',
              'errors': [], 'warnings': []}
    try:
        before = source_files(path)
        episode = read_episode(path)
        record['raw_files'] = before
        if any(size != config['image_size_wh'] for size in episode['image_sizes'].values()):
            raise ValueError('原始图像尺寸与配置不一致')
        duration = (episode['stats']['ended_at_ns'] - episode['stats']['started_at_ns'])/1e9
        record['duration_s'] = duration
        record['warnings'] = episode_warnings(episode, config)
        aligned = align_episode(episode, config, source)
        record.update(frames=len(aligned['state']), alignment=aligned['summary'],
                      applied_exceptions=aligned['applied_exceptions'])
        if source_files(path) != before:
            raise ValueError('检查期间原始文件发生变化')
    except Exception as error:
        record['errors'].append(f'{type(error).__name__}: {error}')
    return record


def audit_data(root, output, config, limit=None):
    """执行全量初检；输入原始根目录、报告路径、配置及可选片段上限，输出并保存检查报告。"""
    root = Path(root).expanduser().resolve()
    output = outside_source(output, root)
    if output.exists():
        raise FileExistsError(f'报告已存在：{output}')
    entries = []
    for path in find_episodes(root)[:limit]:
        entry = audit_episode(path, root, config)
        entries.append(entry)
        print(f"检查 {entry['source']}：错误 {len(entry['errors'])}，待复核 {len(entry['warnings'])}", flush=True)
    result = {'schema_version': 1, 'raw_root': str(root), 'config_sha256': object_digest(config),
              'scope': 'limited' if limit else 'all_discovered', 'episodes': entries}
    write_json(output, result)
    return result


def validate_hdf5(path, expected_frames):
    """完整读取 HDF5；输入文件和预期帧数，输出无，检查维度、RGB、时钟及原始选帧证据。"""
    with h5py.File(path, 'r') as data:
        if data.attrs.get('image_color_space') != 'RGB' or data.attrs.get('selection_method') != 'manual_directory_filter':
            raise ValueError('HDF5 颜色或数据选择方式不符')
        for key in ('action', 'observations/qpos'):
            values = data[key][:]
            if values.shape != (expected_frames, 14) or not np.isfinite(values).all():
                raise ValueError(f'HDF5 数值无效：{key}')
            if np.any(values[:, [6, 13]] < 0) or np.any(values[:, [6, 13]] > 1):
                raise ValueError('夹爪数值越界')
        grid = data['timestamp_ns'][:]
        if len(grid) != expected_frames or np.any(np.diff(grid) <= 0):
            raise ValueError('HDF5 时间网格无效')
        np.testing.assert_allclose(data['timestamp'][:], (grid-grid[0])/1e9, atol=1e-8)
        for key in STREAMS:
            source = data['alignment/source_timestamp_ns/' + key][:]
            age = data['alignment/age_ns/' + key][:]
            index = data['alignment/source_index/' + key][:]
            if (source.shape != grid.shape or age.shape != grid.shape or index.shape != grid.shape
                    or np.any(source > grid) or np.any(age < 0) or np.any(np.diff(index) < 0)):
                raise ValueError('HDF5 选帧溯源无效')
            np.testing.assert_array_equal(grid - source, age)
        for camera in CAMERAS:
            images = data['observations/images/' + camera]
            if images.ndim != 4 or images.shape[0] != expected_frames or images.shape[-1] != 3 or images.dtype != np.uint8:
                raise ValueError('HDF5 图像形状无效')
            for i in range(expected_frames):
                images[i]


def validate_output(root):
    """复核导出目录；输入输出目录，输出摘要，校验全部文件摘要、帧数及两种格式的内容结构。"""
    root = Path(root).expanduser().resolve()
    manifest = read_json(root / 'processing_manifest.json')
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if root not in path.parents or digest(path) != expected:
            raise ValueError(f'导出文件缺失或摘要变化：{name}')
    frames = 0
    for entry in manifest['episodes']:
        path = root / entry['file']
        if manifest['format'] == 'hdf5':
            validate_hdf5(path, entry['frames'])
        else:
            from export_data import validate_parquet
            validate_parquet(path, entry['frames'])
        frames += entry['frames']
    if frames != manifest['total_frames']:
        raise ValueError('导出总帧数不符')
    splits = manifest['splits']
    if set(splits['train']) & set(splits['validation']) or sorted(splits['train']+splits['validation']) != list(range(len(manifest['episodes']))):
        raise ValueError('训练/验证片段划分不完整或存在重叠')
    return {'passed': True, 'format': manifest['format'], 'episodes': len(manifest['episodes']),
            'frames': frames, 'checked_files': len(manifest['files'])}
