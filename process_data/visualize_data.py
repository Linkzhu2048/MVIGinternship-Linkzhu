"""独立导出双腕回放视频，保留原始 episode 名称和目录关系，便于人工定位并剔除片段。"""
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from align_data import time_grid
from read_data import decode_image, find_episodes, outside_source, read_episode, source_files, write_json


def make_video(episode, output, fps=15):
    """生成双腕 MP4；输入原始片段、输出路径和回放帧率，输出无，画面标注时间及图像年龄。"""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    stats, raw = episode['stats'], episode['raw']
    grid = time_grid(stats['started_at_ns'], stats['ended_at_ns'], stats['started_at_ns'], fps)
    keys = ('camera/left', 'camera/right')
    indices = {key: np.searchsorted([row[0] for row in raw[key]], grid, side='right')-1 for key in keys}
    # 标注使用英文以避免 OpenCV 内置字体无法绘制中文；左右相机顺序保持固定。
    command = [ffmpeg, '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1280x520', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    log_path = Path(output).with_suffix('.ffmpeg.log')
    cache = {}
    with log_path.open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for i, timestamp in enumerate(grid):
                canvas = np.zeros((520, 1280, 3), np.uint8)
                for side, key in enumerate(keys):
                    index = int(indices[key][i])
                    label = f'{key}  t={(timestamp-stats["started_at_ns"])/1e9:.2f}s'
                    if index >= 0:
                        source_time, payload = raw[key][index]
                        if cache.get(key, (-1, None))[0] != index:
                            cache[key] = (index, cv2.resize(decode_image(payload), (640, 480)))
                        canvas[40:, side*640:(side+1)*640] = cache[key][1]
                        label += f'  age={(timestamp-source_time)/1e6:.1f}ms'
                    else:
                        label += '  NO PAST FRAME'
                    cv2.putText(canvas, label, (side*640+10, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f'视频编码失败，查看 {log_path}')
        except BaseException:
            process.kill()
            process.wait()
            raise
    log_path.unlink()


def visualize_data(root, output, fps=15):
    """批量导出视频；输入原始根目录、视频目录和帧率，输出源片段与视频对应清单，不依赖检查报告。"""
    root = Path(root).expanduser().resolve()
    output = outside_source(output, root)
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError('回放帧率必须是 1 到 60 的整数')
    paths = find_episodes(root)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    entries = []
    for i, path in enumerate(paths):
        relative = path.relative_to(root)
        # 保留分组路径，避免不同采集目录中的同名 episode 覆盖彼此。
        video_path = relative.parent / (path.name + '.mp4')
        target = output / video_path
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = {'source': relative.as_posix(), 'source_directory': str(path),
                 'video': video_path.as_posix(), 'error': None}
        print(f'视频 {i+1}/{len(paths)}：{entry["source"]}', flush=True)
        try:
            before = source_files(path)
            make_video(read_episode(path, cameras_only=True), target, fps)
            if source_files(path) != before:
                raise ValueError('生成视频期间原始文件发生变化')
        except (OSError, ValueError, RuntimeError) as error:
            entry['error'] = f'{type(error).__name__}: {error}'
            target.unlink(missing_ok=True)
            print(f'视频失败 {entry["source"]}：{error}', flush=True)
        entries.append(entry)
    result = {'raw_root': str(root), 'fps': fps, 'episodes': entries,
              'successful': sum(e['error'] is None for e in entries)}
    write_json(output / 'video_index.json', result)
    return result
