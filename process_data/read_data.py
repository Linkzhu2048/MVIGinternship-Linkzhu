"""读取 NEDF3 原始目录、MCAP 内嵌协议及十路数据；不依赖采集程序或私有 SDK。"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.reader import make_reader

LAYOUT = [name for side in ('left', 'right')
          for name in [*(f'{side}_joint{i}' for i in range(6)), f'{side}_gripper']]
CAMERAS = ('left_wrist', 'right_wrist')
STREAMS = ['camera/left', 'camera/right'] + [
    f'{kind}/{role}/{side}' for role in ('follower', 'leader')
    for side in ('left', 'right') for kind in ('joints', 'gripper')]
TOPICS = {'nmx/hal/camera/camera_1/rgbd': 'camera/left',
          'nmx/hal/camera/camera_0/rgbd': 'camera/right'}
TOPICS.update({f'nmx/hal/i2rt/{role}/{side}/{topic}': f'{kind}/{role}/{side}'
               for role in ('follower', 'leader') for side in ('left', 'right')
               for topic, kind in (('joint', 'joints'), ('hand_state', 'gripper'))})


def read_json(path):
    """读取 JSON；输入文件路径，输出解码对象，格式错误直接报错。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """写入可阅读的 JSON；输入路径及对象，输出无，拒绝非有限数值。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    """分块计算文件 SHA256；输入文件路径，输出十六进制摘要，不一次加载整文件。"""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def object_digest(value):
    """计算对象的稳定摘要；输入可序列化对象，输出 SHA256 字符串。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def source_files(episode):
    """记录原始文件身份；输入 episode 目录，输出相对路径、字节数和摘要列表。"""
    episode = Path(episode).resolve()
    files = [episode / 'metadata.json', episode / 'episode_stats.json',
             episode / 'data/mcap_index.json', *shard_paths(episode)]
    return [{'path': p.relative_to(episode).as_posix(), 'size_bytes': p.stat().st_size,
             'sha256': digest(p)} for p in sorted(files)]


def find_episodes(root):
    """发现录制片段；输入原始数据根目录，输出排序后的 episode 路径列表。"""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    episodes = [p.parent for p in root.rglob('metadata.json')
                if (p.parent / 'data').is_dir()]
    if not episodes:
        raise ValueError(f'没有找到原始 episode：{root}')
    return sorted(episodes)


def outside_source(output, root):
    """保护原始数据；输入输出路径和原始根目录，输出解析路径，禁止写入原始目录内部。"""
    output, root = Path(output).expanduser().resolve(), Path(root).expanduser().resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('输出目录与原始目录不能相同或互相包含')
    return output


def shard_paths(episode):
    """校验分片索引；输入 episode 路径，输出按索引顺序排列且完整的 MCAP 路径。"""
    directory = (Path(episode) / 'data').resolve()
    entries = read_json(directory / 'mcap_index.json')
    if not isinstance(entries, list) or not entries:
        raise ValueError('MCAP 索引为空或格式错误')
    paths = []
    for entry in entries:
        filename = entry.get('filename') if isinstance(entry, dict) else None
        if not isinstance(filename, str) or not filename:
            raise ValueError('MCAP 索引缺少文件名')
        path = (directory / filename).resolve()
        if path.parent != directory or path.suffix != '.mcap' or not path.is_file():
            raise ValueError(f'MCAP 路径无效：{filename}')
        paths.append(path)
    if len(set(paths)) != len(paths) or set(paths) != set(directory.glob('*.mcap')):
        raise ValueError('MCAP 索引与实际分片不一致')
    return paths


class Decoder:
    """缓存 MCAP 内嵌 protobuf 消息类型，避免依赖私有生成代码。"""

    def __init__(self):
        """初始化类型缓存；输入无，输出解码器实例。"""
        self.classes = {}

    def decode(self, schema, payload):
        """解析一条消息；输入 MCAP schema 和字节，输出 protobuf 对象。"""
        if schema is None or schema.encoding != 'protobuf':
            raise ValueError('只支持带内嵌描述的 protobuf 消息')
        key = (schema.name, schema.data)
        if key not in self.classes:
            pool = descriptor_pool.DescriptorPool()
            pending = list(descriptor_pb2.FileDescriptorSet.FromString(schema.data).file)
            while pending:
                remaining = []
                for descriptor in pending:
                    try:
                        pool.Add(descriptor)
                    except Exception:
                        remaining.append(descriptor)
                if len(remaining) == len(pending):
                    raise ValueError('无法解析 protobuf 描述依赖')
                pending = remaining
            self.classes[key] = message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name))
        return self.classes[key].FromString(payload)


def decode_image(data, size=None):
    """解码 v2 JPEG；输入压缩字节及可选宽高，输出 RGB uint8 图像，失败报错。"""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or (size and list(bgr.shape[1::-1]) != list(size)):
        raise ValueError('JPEG 损坏或图像尺寸不符')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_episode(path, cameras_only=False):
    """读取片段；输入路径及仅相机选项，输出元数据和数据流；可视化时不要求机器人数据完整。"""
    path = Path(path).resolve()
    metadata, stats = read_json(path / 'metadata.json'), read_json(path / 'episode_stats.json')
    extra = metadata.get('extra', {})
    if (extra.get('contract_version', metadata.get('contract_version')) != 'i2rt-nedf3-v2'
            or (not cameras_only and extra.get('state_layout') != 'left_joint0..5,left_gripper,right_joint0..5,right_gripper')
            or metadata.get('camera_info') != {'camera_0': 'right_wrist', 'camera_1': 'left_wrist', 'camera_2': 'head'}):
        raise ValueError('原始协议、颜色版本或相机/关节语义不匹配；禁止猜测映射')
    if not cameras_only and (stats.get('writer_error') or stats.get('dropped_batches', 0)):
        raise ValueError('原始录制存在写入错误或丢批')
    if stats['ended_at_ns'] <= stats['started_at_ns']:
        raise ValueError('录制起止时间无效')
    raw, counts, decoder = {key: [] for key in STREAMS}, Counter(), Decoder()
    image_sizes = {}
    for shard in shard_paths(path):
        with shard.open('rb') as stream:
            for schema, channel, message in make_reader(stream, validate_crcs=True).iter_messages(log_time_order=False):
                counts[channel.topic] += 1
                if channel.topic not in TOPICS:
                    continue
                key = TOPICS[channel.topic]
                if cameras_only and not key.startswith('camera/'):
                    continue
                decoded = decoder.decode(schema, message.data)
                if key.startswith('camera/'):
                    value = bytes(decoded.rgb.data)
                    size = [decoded.rgb.cols, decoded.rgb.rows]
                    if not value or min(size) <= 0 or image_sizes.get(key, size) != size:
                        raise ValueError(f'{key} 图像为空或尺寸发生变化')
                    image_sizes[key] = size
                else:
                    names = ['gripper'] if key.startswith('gripper/') else [f'joint{i}' for i in range(6)]
                    joints = {str(j.name): float(j.pos) for j in decoded.joints}
                    if len(joints) != len(decoded.joints) or not set(names).issubset(joints):
                        raise ValueError(f'{key} 关节字段缺失或重复')
                    value = [joints[name] for name in names]
                    if not np.isfinite(value).all():
                        raise ValueError(f'{key} 存在非有限值')
                    if names == ['gripper'] and not 0 <= value[0] <= 1:
                        raise ValueError(f'{key} 夹爪超出 [0,1]，不自动截断')
                timestamp = int(message.log_time)
                if raw[key] and timestamp <= raw[key][-1][0]:
                    raise ValueError(f'{key} 原始时间戳重复或倒退')
                raw[key].append((timestamp, value))
    if cameras_only:
        if not raw['camera/left'] and not raw['camera/right']:
            raise ValueError('没有可显示的腕部图像')
    elif dict(counts) != stats['topic_counts'] or any(not values for values in raw.values()):
        raise ValueError('消息计数不匹配或必要数据流缺失')
    return {'path': path, 'metadata': metadata, 'stats': stats, 'raw': raw, 'image_sizes': image_sizes}
