"""提供独立的检查、视频可视化、转换和验证命令行入口，不启动硬件或训练。"""
import argparse
from pathlib import Path
import sys

import numpy as np

from read_data import STREAMS, read_json

PROJECT_DIR = Path(__file__).resolve().parent


def load_config(path):
    """读取并校验处理配置；输入 JSON 路径，输出配置字典，不接受隐式动作偏移或未知键。"""
    config = read_json(path)
    expected = {'schema_version','dataset_id','repo_id','prompt','fps','image_size_wh','max_age_ms',
                'action','split','normalization_std_floor','short_episode_seconds','alignment_exceptions','pi05'}
    if set(config) != expected or config['schema_version'] != 1:
        raise ValueError('配置字段缺失、未知或版本不支持')
    if config['fps'] != 30:
        raise ValueError('当前训练接口要求 30 Hz')
    for key in ('dataset_id','repo_id','prompt'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'配置 {key} 不能为空')
    if len(config['image_size_wh']) != 2 or any(type(x) is not int or x <= 0 for x in config['image_size_wh']):
        raise ValueError('图像宽高无效')
    action = config['action']
    if set(action) != {'scale','offset','time_offset_ms','mapping_note'} or action['time_offset_ms'] != 0:
        raise ValueError('动作必须明确为零偏移，不叠加 ACT 原版的 t-1 取样')
    if not action['mapping_note'].strip():
        raise ValueError('需要说明 leader → follower 映射依据')
    for key in ('scale','offset'):
        array = np.asarray(action[key], np.float64)
        if array.shape != (14,) or not np.isfinite(array).all():
            raise ValueError('动作映射必须是 14 维有限向量')
    if np.any(np.asarray(action['scale']) == 0):
        raise ValueError('动作缩放不能为零')
    if set(config['max_age_ms']) != {'camera','joints','gripper'} or any(not np.isfinite(v) or v <= 0 for v in config['max_age_ms'].values()):
        raise ValueError('年龄限制无效')
    if not 0 < config['split']['validation_fraction'] < 1 or type(config['split']['seed']) is not int:
        raise ValueError('ACT 验证比例和随机种子无效')
    if not np.isfinite(config['normalization_std_floor']) or config['normalization_std_floor'] <= 0:
        raise ValueError('ACT 标准差下限无效')
    if not np.isfinite(config['short_episode_seconds']) or config['short_episode_seconds'] <= 0:
        raise ValueError('短片段警告阈值无效')
    pi = config['pi05']
    if set(pi) != {'action_horizon','split'} or pi['split'] != 'all_train' or type(pi['action_horizon']) is not int or pi['action_horizon'] < 1:
        raise ValueError('当前 π0.5 分支沿用 all_train 和正整数动作块长度')
    for item in config['alignment_exceptions']:
        if (set(item) != {'source','stream','time_from_recording_start_ns','observed_age_ms','limit_ms','reason'}
                or item['stream'] not in STREAMS or not item['reason'].strip()
                or type(item['time_from_recording_start_ns']) is not int
                or not np.isfinite([item['observed_age_ms'],item['limit_ms']]).all()
                or not 0 < item['observed_age_ms'] <= item['limit_ms']):
            raise ValueError('年龄例外必须指定片段、流、精确时刻、观测值、上限和原因')
    return config


def main(argv=None):
    """执行命令行；输入可选参数列表，输出退出码，成功为 0、检查或参数错误为 2。"""
    parser = argparse.ArgumentParser(description='NEDF3 检查、视频可视化、ACT/π0.5 数据转换')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [('check','读取全部原始帧并检查'), ('visualize','独立导出按 episode 命名的视频'),
                            ('convert','转换原始目录中保留的全部片段'), ('validate','验证已导出的数据')]:
        command = commands.add_parser(name, help=help_text)
        if name == 'validate':
            command.add_argument('--data', required=True, help='已导出的 HDF5 或 LeRobot 根目录')
            continue
        command.add_argument('--raw', default=str(PROJECT_DIR / 'dataset'),
                             help='原始 NEDF3 根目录，默认本子项目的 dataset/')
        if name != 'visualize':
            command.add_argument('--config', default=str(Path(__file__).with_name('config.example.json')))
        command.add_argument('--output', help='输出路径，默认写入本子项目的 output/ 下')
        if name == 'check':
            command.add_argument('--limit', type=int, help='仅检查前 N 个片段，供试跑')
        if name == 'visualize':
            command.add_argument('--fps', type=int, default=15, help='回放视频帧率，默认 15')
        if name == 'convert':
            command.add_argument('--format', choices=['hdf5','lerobot'], required=True)
    args = parser.parse_args(argv)
    if args.command != 'validate' and args.output is None:
        destination = {'check': 'check.json', 'visualize': 'videos',
                       'convert': getattr(args, 'format', None)}[args.command]
        args.output = str(PROJECT_DIR / 'output' / destination)
    try:
        if args.command == 'validate':
            from check_data import validate_output
            result = validate_output(args.data)
        elif args.command == 'visualize':
            from visualize_data import visualize_data
            report = visualize_data(args.raw, args.output, args.fps)
            result = {'videos':report['successful'], 'failed':len(report['episodes'])-report['successful'],
                      'index':str(Path(args.output)/'video_index.json')}
            if result['failed']:
                print(result)
                return 2
        else:
            config = load_config(args.config)
            if args.command == 'check':
                from check_data import audit_data
                if args.limit is not None and args.limit < 1:
                    raise ValueError('--limit 必须为正整数')
                result = audit_data(args.raw, args.output, config, args.limit)
                errors = sum(bool(e['errors']) for e in result['episodes'])
                print(f'初检完成：{len(result["episodes"])} 个片段，{errors} 个存在阻断项；报告：{args.output}')
                return 2 if errors else 0
            else:
                from export_data import convert_data
                result = convert_data(args.raw, args.output, config, args.format)
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'失败：{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
