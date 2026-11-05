"""将双腕图像、follower 状态和 leader 动作按过去样本对齐，保存选帧证据。"""
import numpy as np

from read_data import STREAMS


def time_grid(start, end, origin, fps):
    """建立整数纳秒网格；输入覆盖起止、录制起点和帧率，输出不越界的时间数组。"""
    # 先包含边界附近候选点，再按整数纳秒筛选，避免四舍五入后恰在边界的帧被丢弃。
    first = ((int(start) - int(origin)) * fps) // 1_000_000_000
    last = ((int(end) - int(origin)) * fps + 999_999_999) // 1_000_000_000
    steps = np.arange(first, last + 1, dtype=np.int64)
    grid = int(origin) + (steps * 1_000_000_000 + fps // 2) // fps
    return grid[(grid >= start) & (grid <= end)]


def select_samples(times, grid, limit_ms, exceptions=(), origin=0):
    """选最近过去样本；输入时间数组、网格、年龄上限和精确例外，输出索引、年龄、例外证据。"""
    times = np.asarray(times, np.int64)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('源时间戳必须非空且严格递增')
    indices = np.searchsorted(times, grid, side='right') - 1
    if np.any(indices < 0):
        raise ValueError('目标时刻没有过去样本')
    ages = grid - times[indices]
    limits = np.full(len(grid), round(limit_ms * 1e6), np.int64)
    used, positions = [], set()
    for exception in exceptions:
        hits = np.flatnonzero(grid - origin == exception['time_from_recording_start_ns'])
        if len(hits) != 1 or not exception.get('reason', '').strip():
            raise ValueError('年龄例外必须对应唯一时刻并写明人工批准原因')
        i = int(hits[0])
        if i in positions or ages[i] <= limits[i]:
            raise ValueError('重复或不必要的年龄例外')
        if abs(int(ages[i]) - round(exception['observed_age_ms'] * 1e6)) > 1:
            raise ValueError('例外处的实际年龄与批准值不符')
        limits[i] = round(exception['limit_ms'] * 1e6)
        positions.add(i)
        used.append({**exception, 'frame_index': i, 'actual_age_ns': int(ages[i])})
    bad = np.flatnonzero((ages < 0) | (ages > limits))
    if len(bad):
        i = int(bad[0])
        raise ValueError(f'第 {i} 帧样本年龄 {ages[i]/1e6:.6f} ms 超过 {limits[i]/1e6:.6f} ms')
    return indices, ages, used


def align_episode(episode, config, source):
    """对齐单个片段；输入原始数据、配置和相对源路径，输出网格、14维状态/动作及溯源数组。"""
    raw, stats = episode['raw'], episode['stats']
    start = max(stats['started_at_ns'], *(records[0][0] for records in raw.values()))
    end = min(stats['ended_at_ns'], *(records[-1][0] for records in raw.values()))
    grid = time_grid(start, end, stats['started_at_ns'], config['fps'])
    if len(grid) < 2:
        raise ValueError('共同覆盖时长不足两帧')
    indices, provenance, summary, applied, values = {}, {'timestamp_ns': grid}, {}, [], {}
    for key in STREAMS:
        times = np.asarray([record[0] for record in raw[key]], np.int64)
        exceptions = [item for item in config['alignment_exceptions']
                      if item['source'] == source and item['stream'] == key]
        try:
            index, age, used = select_samples(times, grid, config['max_age_ms'][key.split('/')[0]],
                                             exceptions, stats['started_at_ns'])
        except ValueError as error:
            raise ValueError(f'{source}/{key}：{error}') from error
        indices[key] = index
        prefix = key.replace('/', '__')
        provenance[prefix + '__source_index'] = index
        provenance[prefix + '__source_timestamp_ns'] = times[index]
        provenance[prefix + '__age_ns'] = age
        summary[key] = {'max_age_ms': float(age.max()/1e6), 'source_samples': len(times),
                        'selected_samples': int(len(np.unique(index)))}
        applied.extend(used)
        if not key.startswith('camera/'):
            values[key] = np.asarray([record[1] for record in raw[key]], np.float64)[index]
    arrays = {role: np.concatenate([values[f'{kind}/{role}/{side}']
                                    for side in ('left', 'right') for kind in ('joints', 'gripper')], axis=1)
              for role in ('follower', 'leader')}
    state = arrays['follower'].astype(np.float32)
    actions = (arrays['leader'] * config['action']['scale'] + config['action']['offset']).astype(np.float32)
    if not np.isfinite([state, actions]).all() or np.any(actions[:, [6, 13]] < 0) or np.any(actions[:, [6, 13]] > 1):
        raise ValueError('映射后动作无效；请核对映射，不自动裁剪')
    return {'episode': episode, 'indices': indices, 'provenance': provenance,
            'state': state, 'actions': actions, 'summary': summary, 'applied_exceptions': applied,
            'trimmed_start_s': float((grid[0]-stats['started_at_ns'])/1e9),
            'trimmed_end_s': float((stats['ended_at_ns']-grid[-1])/1e9)}
