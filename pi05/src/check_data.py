"""检查真实 LeRobot 数据、动作块和统计量，可逐帧解码图像；不会生成训练数据。"""

import argparse
import json
from pathlib import Path
import numpy as np
from .adapter import CAMERAS, robot_array
from .config import PROJECT_ROOT, Settings, setup_caches, write_json
from .dataset import action_chunks, load_stats, manifest, numeric_episodes, open_dataset


def check(settings, *, deep=False):
    """核对可训练数据；输入：训练设置和是否逐帧解码；输出：帧数、范围、图像提示的报告，结构错误抛异常。"""
    from openpi.transforms import Normalize

    info = manifest(settings.dataset_root, verify_hashes=True)
    if (info["repo_id"], info["prompt"]) != (settings.repo_id, settings.prompt):
        raise ValueError("配置中的 repo_id、prompt 必须与数据清单一致")
    stats = load_stats(settings)
    normalize = Normalize(stats, use_quantiles=True, strict=True)
    counts = 0
    episodes = 0
    bounds = {
        name: {"min": np.full(14, np.inf), "max": np.full(14, -np.inf)} for name in ("state", "actions")
    }
    for state, actions in numeric_episodes(settings.dataset_root):
        robot_array(state, "state", chunk=True)
        robot_array(actions, "actions", chunk=True)
        episodes += 1
        counts += len(state)
        for current, chunks in action_chunks(state, actions, settings.action_horizon):
            values = normalize({"state": current, "actions": chunks})
            for name, value in values.items():
                value = value.reshape(-1, 14)
                if not np.isfinite(value).all():
                    raise ValueError(f"{name} 归一化后存在非有限值")
                bounds[name]["min"] = np.minimum(bounds[name]["min"], value.min(0))
                bounds[name]["max"] = np.maximum(bounds[name]["max"], value.max(0))
    if counts != info["total_frames"] or episodes != info["total_episodes"] or (not counts):
        raise ValueError("实际帧数或片段数量与清单不符")
    dataset = open_dataset(settings.dataset_root, settings.action_horizon)
    indices = range(len(dataset)) if deep else sorted({0, len(dataset) // 2, len(dataset) - 1})
    image_warnings = []
    warning_count = 0
    for index in indices:
        sample = dataset[index]
        if sample["task"] != settings.prompt or tuple(sample["action"].shape) != (
            settings.action_horizon,
            14,
        ):
            raise ValueError(f"第 {index} 帧的任务或动作块形状错误")
        for camera in CAMERAS:
            value = sample["observation.images." + camera].numpy()
            if (
                value.shape != (3, 224, 224)
                or not np.isfinite(value).all()
                or value.min() < 0
                or (value.max() > 1)
            ):
                raise ValueError(f"第 {index} 帧的 {camera} 图像格式错误")
            if value.max() <= 2 / 255 or value.min() >= 253 / 255:
                warning_count += 1
                if len(image_warnings) < 100:
                    image_warnings.append({"index": index, "camera": camera, "reason": "接近全黑或全白"})
    review = []
    for name, value in bounds.items():
        span = stats[name].q99 - stats[name].q01
        for dimension in range(14):
            peak = max(abs(value["min"][dimension]), abs(value["max"][dimension]))
            if span[dimension] <= 1e-06 or peak > 100:
                review.append(
                    {
                        "group": name,
                        "dimension": dimension,
                        "quantile_width": float(span[dimension]),
                        "normalized_peak": float(peak),
                    }
                )
    return {
        "passed": True,
        "episodes": episodes,
        "frames": counts,
        "split": "all_train",
        "action_horizon": settings.action_horizon,
        "decoded_frames": len(indices),
        "deep": deep,
        "normalized_ranges": {k: {n: v.tolist() for n, v in b.items()} for k, b in bounds.items()},
        "normalization_review": review,
        "image_warning_count": warning_count,
        "image_warnings_first_100": image_warnings,
        "requires_review": bool(review or warning_count),
        "note": "passed 仅表示格式、完整性和数值检查通过；提示需人工查看，不会自动裁剪或剔除数据。",
    }


def main():
    """运行数据检查命令；输入：配置、深度选项和报告路径；输出：终端概要及 JSON 报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/train.json")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output/data_check.json")
    args = parser.parse_args()
    setup_caches(PROJECT_ROOT / "output/cache")
    result = check(Settings.load(args.config), deep=args.deep)
    write_json(args.output, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("normalized_ranges", "image_warnings_first_100")},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"报告：{args.output.resolve()}")


if __name__ == "__main__":
    main()
