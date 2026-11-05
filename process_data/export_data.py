"""导出 ACT HDF5 或 LeRobot v2.1，生成整片段划分、模型专用统计和溯源清单。"""
from collections import defaultdict
import io
import json
from pathlib import Path
import pickle

import h5py
import numpy as np
from PIL import Image

from align_data import align_episode
from read_data import (CAMERAS, LAYOUT, STREAMS, decode_image, digest, object_digest,
                       outside_source, find_episodes, read_episode, source_files, write_json)


def aligned_image(aligned, side, index, padded=False):
    """取一帧训练图像；输入对齐结果、左右侧、帧号及缩放选项，输出 RGB uint8。"""
    key = 'camera/' + side
    source_index = aligned['indices'][key][index]
    episode = aligned['episode']
    rgb = decode_image(episode['raw'][key][source_index][1], episode['image_sizes'][key])
    if not padded:
        return rgb
    # 与 openpi 的 PIL 双线性等比例缩放、居中黑边规则一致；不裁剪工作区。
    height, width = rgb.shape[:2]
    ratio = max(width/224, height/224)
    size = (int(width/ratio), int(height/ratio))
    result = Image.new('RGB', (224, 224))
    result.paste(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR),
                 ((224-size[0])//2, (224-size[1])//2))
    return np.asarray(result)


def write_hdf5(path, aligned, entry, config):
    """写单个 ACT 片段并逐帧核对；输入路径、对齐数据、源片段记录及配置，输出无。"""
    n = len(aligned['state'])
    grid = aligned['provenance']['timestamp_ns']
    with h5py.File(path, 'w') as data:
        # human_review 是旧 ACT 读取器的兼容字段；表示调用方已通过目录筛选，不代表程序判定任务成功。
        data.attrs.update(sim=False, compress=False, sample_hz=config['fps'],
                          action_time_offset_ms=0, episode_id=entry['source'], human_review='pass',
                          image_color_space='RGB', selection_method='manual_directory_filter', state_layout=','.join(LAYOUT),
                          action_kind='reconstructed_leader_absolute_target',
                          conversion_config_sha256=object_digest(config))
        data.create_dataset('observations/qpos', data=aligned['state'])
        data.create_dataset('action', data=aligned['actions'])
        data.create_dataset('timestamp_ns', data=grid)
        data.create_dataset('timestamp', data=(grid-grid[0])/1e9)
        data.create_dataset('provenance/raw_files_json', data=json.dumps(entry['raw_files'], ensure_ascii=False),
                            dtype=h5py.string_dtype())
        for key in STREAMS:
            prefix = key.replace('/', '__')
            for field in ('source_timestamp_ns', 'source_index', 'age_ns'):
                data.create_dataset(f'alignment/{field}/{key}', data=aligned['provenance'][prefix+'__'+field])
        for side, camera in zip(('left', 'right'), CAMERAS):
            first = aligned_image(aligned, side, 0)
            images = data.create_dataset('observations/images/'+camera, (n, *first.shape), dtype='uint8',
                                         chunks=(1, *first.shape), compression='lzf')
            for i in range(n):
                images[i] = aligned_image(aligned, side, i)
    with h5py.File(path, 'r') as data:
        np.testing.assert_array_equal(data['action'][:], aligned['actions'])
        np.testing.assert_array_equal(data['observations/qpos'][:], aligned['state'])
        for side, camera in zip(('left', 'right'), CAMERAS):
            for i in range(n):
                np.testing.assert_array_equal(data['observations/images/'+camera][i], aligned_image(aligned, side, i))


def array_stats(array):
    """统计一组数值；输入二维样本数组，输出 LeRobot 所需最值、均值、标准差及计数。"""
    array = np.asarray(array, np.float64)
    return {'min': array.min(axis=0).tolist(), 'max': array.max(axis=0).tolist(),
            'mean': array.mean(axis=0).tolist(), 'std': array.std(axis=0).tolist(), 'count': [len(array)]}


def write_parquet(path, aligned, episode_index, offset, config):
    """分批写 LeRobot 片段；输入路径、对齐数据、片段号及全局偏移，输出该片段的特征统计。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Features, Image as ImageFeature, Sequence, Value

    features = Features({**{key: Sequence(Value('float32'), length=14) for key in ('observation.state', 'action')},
                         **{'observation.images.'+camera: ImageFeature() for camera in CAMERAS},
                         'timestamp': Value('float32'),
                         **{key: Value('int64') for key in ('frame_index', 'episode_index', 'index', 'task_index')}})
    n, stats = len(aligned['state']), {}
    images_sum = {camera: np.zeros(3) for camera in CAMERAS}
    images_sq = {camera: np.zeros(3) for camera in CAMERAS}
    images_min = {camera: np.ones(3) for camera in CAMERAS}
    images_max = {camera: np.zeros(3) for camera in CAMERAS}
    with pq.ParquetWriter(path, features.arrow_schema, compression='snappy') as writer:
        for start in range(0, n, 32):
            stop = min(start+32, n)
            rows = {'observation.state': aligned['state'][start:stop].tolist(),
                    'action': aligned['actions'][start:stop].tolist(),
                    'timestamp': (np.arange(start, stop)/config['fps']).astype(np.float32).tolist(),
                    'frame_index': list(range(start, stop)), 'episode_index': [episode_index]*(stop-start),
                    'index': list(range(offset+start, offset+stop)), 'task_index': [0]*(stop-start)}
            for side, camera in zip(('left', 'right'), CAMERAS):
                encoded = []
                for i in range(start, stop):
                    image = aligned_image(aligned, side, i, padded=True)
                    pixels = image.reshape(-1, 3).astype(np.float64)/255
                    images_sum[camera] += pixels.sum(axis=0)
                    images_sq[camera] += np.square(pixels).sum(axis=0)
                    images_min[camera] = np.minimum(images_min[camera], pixels.min(axis=0))
                    images_max[camera] = np.maximum(images_max[camera], pixels.max(axis=0))
                    buffer = io.BytesIO()
                    Image.fromarray(image).save(buffer, format='PNG')
                    encoded.append({'bytes': buffer.getvalue(), 'path': None})
                rows['observation.images.'+camera] = encoded
            writer.write_table(pa.Table.from_pydict(rows, schema=features.arrow_schema))
    numeric = {'observation.state': aligned['state'], 'action': aligned['actions'],
               'timestamp': (np.arange(n)/config['fps']).astype(np.float32)[:,None],
               'frame_index': np.arange(n)[:,None], 'episode_index': np.full((n,1), episode_index),
               'index': np.arange(offset,offset+n)[:,None], 'task_index': np.zeros((n,1))}
    stats.update({key: array_stats(value) for key, value in numeric.items()})
    for camera in CAMERAS:
        mean = images_sum[camera]/(n*224*224)
        std = np.sqrt(np.maximum(0, images_sq[camera]/(n*224*224)-mean**2))
        stats['observation.images.'+camera] = {key: value[:,None,None].tolist() for key, value in
                                             [('mean', mean), ('std', std), ('min', images_min[camera]), ('max', images_max[camera])]}
        stats['observation.images.'+camera]['count'] = [n]
    validate_parquet(path, n, aligned, episode_index, offset, config['fps'])
    return stats


def validate_parquet(path, frames, aligned=None, episode_index=None, offset=None, fps=30):
    """读回 Parquet；输入文件、帧数及可选源对照，输出无，逐帧核对数值和无损 RGB。"""
    import pyarrow.parquet as pq
    position = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
        rows = batch.to_pydict()
        end = position+batch.num_rows
        for name, target in [('observation.state', 'state'), ('action', 'actions')]:
            values = np.asarray(rows[name], np.float32)
            if values.shape != (batch.num_rows, 14) or not np.isfinite(values).all():
                raise ValueError('Parquet 数值维度或内容无效')
            if np.any(values[:, [6,13]] < 0) or np.any(values[:, [6,13]] > 1):
                raise ValueError('Parquet 夹爪越界')
            if aligned is not None:
                np.testing.assert_array_equal(values, aligned[target][position:end])
        np.testing.assert_array_equal(rows['frame_index'], np.arange(position, end))
        np.testing.assert_array_equal(np.asarray(rows['timestamp'], np.float32), (np.arange(position,end)/fps).astype(np.float32))
        np.testing.assert_array_equal(rows['task_index'], np.zeros(batch.num_rows, np.int64))
        if episode_index is not None:
            np.testing.assert_array_equal(rows['episode_index'], np.full(batch.num_rows, episode_index))
        if offset is not None:
            np.testing.assert_array_equal(rows['index'], np.arange(offset+position, offset+end))
        for side, camera in zip(('left', 'right'), CAMERAS):
            for i, item in enumerate(rows['observation.images.'+camera]):
                if not item['bytes'] or not item['bytes'].startswith(b'\x89PNG\r\n\x1a\n'):
                    raise ValueError('图像不是嵌入的无损 PNG')
                with Image.open(io.BytesIO(item['bytes'])) as image:
                    rgb = np.asarray(image)
                    if rgb.shape != (224,224,3) or image.mode != 'RGB':
                        raise ValueError('Parquet 图像形状或模式错误')
                    if aligned is not None:
                        np.testing.assert_array_equal(rgb, aligned_image(aligned, side, position+i, padded=True))
        position = end
    if position != frames:
        raise ValueError('Parquet 帧数不符')


def split_episodes(entries, config, format_name):
    """划分整个片段；输入源片段记录、配置和格式，输出训练/验证索引，禁止帧级随机划分。"""
    if format_name == 'lerobot':
        return {'train': list(range(len(entries))), 'validation': []}
    if len(entries) < 2:
        raise ValueError('ACT 训练/验证划分至少需要两个保留片段')
    groups = defaultdict(list)
    for i, entry in enumerate(entries):
        groups[entry['group']].append(i)
    rng = np.random.default_rng(config['split']['seed'])
    validation = []
    for group in sorted(groups):
        values = rng.permutation(groups[group]).tolist()
        count = min(len(values)-1, max(1, round(len(values)*config['split']['validation_fraction']))) if len(values)>1 else 0
        validation.extend(values[:count])
    if not validation:
        validation = [int(rng.integers(len(entries)))]
    return {'train': [i for i in range(len(entries)) if i not in validation], 'validation': sorted(validation)}


def numeric_episodes(root, entries, indices, format_name):
    """迭代数值数据；输入目录、清单、片段索引和格式，输出逐片段状态/动作数组，不解码图像。"""
    for i in indices:
        path = root / entries[i]['file']
        if format_name == 'hdf5':
            with h5py.File(path, 'r') as data:
                yield data['observations/qpos'][:], data['action'][:]
        else:
            import pyarrow.parquet as pq
            rows = pq.read_table(path, columns=['observation.state', 'action']).to_pydict()
            yield np.asarray(rows['observation.state'], np.float32), np.asarray(rows['action'], np.float32)


def write_act_assets(root, entries, splits, config):
    """生成 ACT 训练资产；输入输出目录、清单、划分和配置，输出无，仅用训练片段计算统计。"""
    total, sums, squares = 0, np.zeros((2,14)), np.zeros((2,14))
    for state, actions in numeric_episodes(root, entries, splits['train'], 'hdf5'):
        values = np.stack([state, actions]).astype(np.float64)
        total += len(state)
        sums += values.sum(axis=1)
        squares += np.square(values).sum(axis=1)
    mean = sums/total
    std = np.sqrt(np.maximum(0, squares/total-mean**2))
    norm = {'computed_from': 'training episodes only', 'frames': total,
            'std_floor': config['normalization_std_floor']}
    for i, key in enumerate(('qpos', 'action')):
        norm[key+'_mean'] = mean[i].tolist()
        norm[key+'_std_before_floor'] = std[i].tolist()
        norm[key+'_std'] = np.maximum(std[i], config['normalization_std_floor']).tolist()
    assets = root / 'training_assets'
    write_json(assets / 'normalization.json', norm)
    write_json(assets / 'splits.json', {**{key: [entries[i]['file'] for i in value] for key, value in splits.items()},
                                      'seed': config['split']['seed'], 'unit': 'whole_episode'})
    write_json(assets / 'dataset_manifest.json', {'schema_version': 1, 'dataset_id': config['dataset_id'],
                                                'episodes': entries, 'cameras': list(CAMERAS), 'layout': LAYOUT})
    write_json(assets / 'conversion_config.json', config)
    # 只生成本程序自己的 pickle，用户不需要载入任何来源不明的 pickle。
    with (assets / 'dataset_stats.pkl').open('wb') as stream:
        pickle.dump({key: np.asarray(norm[key], np.float32) for key in ('qpos_mean','qpos_std','action_mean','action_std')}, stream)


def action_chunks(state, actions, horizon, block_size=256):
    """生成 π0.5 动作块；输入状态、绝对动作及块长度，输出分批相对关节/绝对夹爪动作。"""
    joints = [i for i in range(14) if i not in (6,13)]
    for start in range(0, len(state), block_size):
        steps = np.arange(start, min(start+block_size, len(state)))
        index = np.minimum(steps[:,None]+np.arange(horizon), len(actions)-1)
        chunks = actions[index].copy()
        chunks[..., joints] -= state[steps][:,None,joints]
        yield state[steps], chunks


class RunningStats:
    """沿用 openpi 的 5000 桶流式统计；改编来源和 Apache-2.0 许可见 LICENSE.openpi。"""

    def __init__(self):
        """建立统计器；输入无，输出空统计实例，内存不随帧数增长。"""
        self.count = 0
        self.mean = self.mean_square = self.minimum = self.maximum = None
        self.histograms = self.edges = None
        self.bins = 5000

    def update(self, batch):
        """更新均值、平方均值和直方图；输入末维为特征的数组，输出无，原地累计。"""
        batch = batch.reshape(-1, batch.shape[-1])
        count, dimensions = batch.shape
        if not count:
            raise ValueError('统计批次不能为空')
        if not self.count:
            self.mean = np.mean(batch, axis=0)
            self.mean_square = np.mean(batch**2, axis=0)
            self.minimum = np.min(batch, axis=0)
            self.maximum = np.max(batch, axis=0)
            self.histograms = [np.zeros(self.bins) for _ in range(dimensions)]
            self.edges = [np.linspace(self.minimum[i]-1e-10, self.maximum[i]+1e-10, self.bins+1)
                          for i in range(dimensions)]
        else:
            if dimensions != self.mean.size:
                raise ValueError('统计维度发生变化')
            minimum, maximum = np.min(batch,axis=0), np.max(batch,axis=0)
            changed = np.any(minimum < self.minimum) or np.any(maximum > self.maximum)
            self.minimum = np.minimum(self.minimum, minimum)
            self.maximum = np.maximum(self.maximum, maximum)
            if changed:
                for i in range(dimensions):
                    edges = np.linspace(self.minimum[i], self.maximum[i], self.bins+1)
                    histogram, _ = np.histogram(self.edges[i][:-1], bins=edges, weights=self.histograms[i])
                    self.histograms[i], self.edges[i] = histogram, edges
        self.count += count
        self.mean += (np.mean(batch,axis=0)-self.mean)*(count/self.count)
        self.mean_square += (np.mean(batch**2,axis=0)-self.mean_square)*(count/self.count)
        for i in range(dimensions):
            histogram, _ = np.histogram(batch[:,i], bins=self.edges[i])
            self.histograms[i] += histogram

    def statistics(self):
        """提取统计值；输入无，输出均值、标准差、q01/q99 字典，不伪造退化维度或截断数值。"""
        if self.count < 2:
            raise ValueError('统计至少需要两个样本')
        result = {'mean':self.mean.tolist(),
                  'std':np.sqrt(np.maximum(0,self.mean_square-self.mean**2)).tolist()}
        for name, quantile in [('q01',.01),('q99',.99)]:
            result[name] = [float(edges[np.searchsorted(np.cumsum(histogram),quantile*self.count)])
                            for histogram,edges in zip(self.histograms,self.edges)]
        return result


def write_pi_stats(root, entries, splits, config):
    """生成 π0.5 专用统计；输入目录、清单、划分和配置，输出来源说明，沿用 openpi 流式算法。"""
    horizon = config['pi05']['action_horizon']
    accumulators = {key:RunningStats() for key in ('state','actions')}
    frames = 0
    for state, actions in numeric_episodes(root, entries, splits['train'], 'lerobot'):
        for states, chunks in action_chunks(state, actions, horizon):
            accumulators['state'].update(states)
            accumulators['actions'].update(chunks.reshape(-1,14))
            frames += len(states)
    norm = {key:accumulator.statistics() for key,accumulator in accumulators.items()}
    write_json(root / 'assets/norm_stats.json', {'norm_stats':norm})
    write_json(root / 'assets/normalization_diagnostics.json', {
        name: {'zero_std_dimensions': np.flatnonzero(np.asarray(value['std']) < 1e-8).tolist(),
               'collapsed_quantile_dimensions': np.flatnonzero(np.asarray(value['q99'])-value['q01'] < 1e-8).tolist()}
        for name,value in norm.items()})
    return {'action_horizon':horizon, 'fps':config['fps'], 'state_samples':frames,
            'action_samples':frames*horizon, 'split':'all_train', 'tail_policy':'repeat_last_action',
            'actions':'relative_to_current_state_except_absolute_grippers',
            'quantile_method':'openpi_running_histogram_5000_bins', 'statistics_clipping':False}


def write_lerobot_meta(root, entries, statistics, config):
    """写 LeRobot v2.1 元数据；输入输出目录、片段清单、逐片段统计及配置，输出无。"""
    features = {key: {'dtype': 'float32', 'shape': [14], 'names': LAYOUT} for key in ('observation.state', 'action')}
    features.update({'observation.images.'+c: {'dtype':'image', 'shape':[3,224,224], 'names':['channels','height','width']} for c in CAMERAS})
    features.update({key: {'dtype': 'float32' if key=='timestamp' else 'int64', 'shape':[1], 'names':None}
                     for key in ('timestamp','frame_index','episode_index','index','task_index')})
    write_json(root / 'meta/info.json', {'codebase_version':'v2.1', 'robot_type':'i2rt_bimanual',
               'total_episodes':len(entries), 'total_frames':sum(e['frames'] for e in entries),
               'total_tasks':1, 'total_videos':0, 'total_chunks':(len(entries)+999)//1000,
               'chunks_size':1000, 'fps':config['fps'], 'splits':{'train':f'0:{len(entries)}'},
               'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
               'video_path':None, 'features':features})
    documents = {'episodes.jsonl': [{'episode_index':i, 'tasks':[config['prompt']], 'length':e['frames']} for i,e in enumerate(entries)],
                 'tasks.jsonl':[{'task_index':0, 'task':config['prompt']}],
                 'episodes_stats.jsonl':[{'episode_index':i, 'stats':s} for i,s in enumerate(statistics)]}
    for name, rows in documents.items():
        (root / 'meta' / name).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows), encoding='utf-8')


def convert_data(root, output, config, format_name):
    """转换目录内全部片段；输入筛选后的原始根目录、目标目录、配置和格式，输出清单，失败保留 partial。"""
    from check_data import episode_warnings, validate_output

    if format_name not in ('hdf5', 'lerobot'):
        raise ValueError('仅支持 hdf5 或 lerobot')
    root = Path(root).expanduser().resolve()
    output = outside_source(output, root)
    stage = output.with_name(output.name+'.partial')
    if output.exists() or stage.exists():
        raise FileExistsError('目标或 .partial 已存在；请换新目录，不覆盖已有数据')
    entries = [{'source': path.relative_to(root).as_posix(),
                'group': path.parent.relative_to(root).as_posix() if path != root else '.'}
               for path in find_episodes(root)]
    sources = {entry['source'] for entry in entries}
    if any(item['source'] not in sources for item in config['alignment_exceptions']):
        raise ValueError('配置含当前目录中不存在的片段年龄例外，请核对配置')
    splits = split_episodes(entries, config, format_name)
    stage.mkdir(parents=True)
    receipts, statistics, offset = [], [], 0
    for i, entry in enumerate(entries):
        path = (root / entry['source']).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f'源路径越界：{entry["source"]}')
        entry['raw_files'] = source_files(path)
        episode = read_episode(path)
        if any(list(size) != config['image_size_wh'] for size in episode['image_sizes'].values()):
            raise ValueError('原始图像尺寸与配置不一致，不静默拉伸')
        entry['warnings'] = episode_warnings(episode, config)
        if entry['warnings']:
            print(f'质量提示 {entry["source"]}：{entry["warnings"]}', flush=True)
        aligned = align_episode(episode, config, entry['source'])
        filename = f'episode_{i}.hdf5' if format_name=='hdf5' else f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet'
        target = stage / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if format_name == 'hdf5':
            write_hdf5(target, aligned, entry, config)
        else:
            statistics.append(write_parquet(target, aligned, i, offset, config))
        receipt = {**entry, 'id':entry['source'], 'file':filename, 'frames':len(aligned['state']),
                   'training_episode_index':i, 'selection_method':'manual_directory_filter',
                   'trimmed_start_s':aligned['trimmed_start_s'],
                   'trimmed_end_s':aligned['trimmed_end_s'], 'alignment':aligned['summary'],
                   'applied_exceptions':aligned['applied_exceptions'], 'all_pixels_verified':True}
        provenance_dir = stage / 'provenance'
        provenance_dir.mkdir(exist_ok=True)
        np.savez_compressed(provenance_dir / f'episode_{i:06d}.npz', **aligned['provenance'])
        write_json(provenance_dir / f'episode_{i:06d}.json', receipt)
        if source_files(path) != entry['raw_files']:
            raise ValueError('转换期间原始文件发生变化')
        receipts.append(receipt)
        offset += receipt['frames']
        print(f'转换 {i+1}/{len(entries)}：{entry["source"]} → {filename}，{receipt["frames"]} 帧', flush=True)
    write_json(stage / 'conversion_config.json', config)
    write_json(stage / 'source_manifest.json', {'raw_root':str(root),
               'selection_method':'manual_directory_filter', 'episodes':receipts})
    if format_name == 'hdf5':
        write_act_assets(stage, receipts, splits, config)
    else:
        write_lerobot_meta(stage, receipts, statistics, config)
        pi_provenance = write_pi_stats(stage, receipts, splits, config)
        pi_manifest = {'schema_version':1, 'repo_id':config['repo_id'], 'fps':config['fps'], 'prompt':config['prompt'],
                       'cameras':list(CAMERAS), 'layout':LAYOUT, 'image_size':[224,224], 'split':'all_train',
                       'total_episodes':len(entries), 'total_frames':offset, 'sources':receipts,
                       'action_space':'absolute_joint_radians_and_normalized_grippers',
                       'files':{p.relative_to(stage).as_posix():digest(p) for p in sorted(stage.rglob('*'))
                                if p.is_file() and 'assets' not in p.relative_to(stage).parts}}
        write_json(stage / 'i2rt_manifest.json', pi_manifest)
        write_json(stage / 'assets/provenance.json', {**pi_provenance, 'dataset_manifest_sha256':digest(stage / 'i2rt_manifest.json')})
    manifest = {'schema_version':1, 'format':format_name, 'episodes':receipts, 'total_frames':offset,
                'splits':splits, 'config_sha256':object_digest(config),
                'source_manifest_sha256':digest(stage / 'source_manifest.json'),
                'action_kind':'reconstructed_leader_absolute_target_not_final_command_log',
                'files':{p.relative_to(stage).as_posix():digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}}
    write_json(stage / 'processing_manifest.json', manifest)
    validate_output(stage)
    stage.rename(output)
    return {'output':str(output), 'episodes':len(entries), 'frames':offset, 'format':format_name}
