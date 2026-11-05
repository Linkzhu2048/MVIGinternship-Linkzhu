# 数据处理

独立完成 NEDF3 数据的可视化、检查及 ACT HDF5 / π0.5 LeRobot 转换。安装 `requirements.txt` 中的依赖即可使用，不需要采集项目或训练项目的源码。

流程：**放入原始 episode → 导出视频 → 人工删除不需要的 episode → 转换 → 验证**。各工具可独立运行。

## 目录与文件

请将原始 episode 文件夹放入本子项目的 `dataset/`。生成内容统一存入 `output/`；这两个目录初始为空，下面展示放入数据并运行后的结构。

```text
process_data/
├── dataset/                    # 用户放入的原始数据
│   ├── episode_0000/
│   │   ├── metadata.json
│   │   ├── episode_stats.json
│   │   └── data/
│   │       ├── mcap_index.json
│   │       └── *.mcap           # 保留索引中的全部分片
│   └── episode_0001/
├── output/                     # 程序生成的内容
│   ├── videos/                 # 逐 episode 的双腕 MP4 和路径索引
│   ├── check.json              # 原始数据检查报告
│   ├── hdf5/                   # ACT 数据及训练资产
│   └── lerobot/                # π0.5 数据及统计
└── ...                         # 下表中的源码和配置
```

| 文件 | 作用 |
| --- | --- |
| `main.py` | 命令行入口，默认读取 `dataset/`、写入 `output/` |
| `visualize_data.py` | 导出双腕视频，保留 episode 名称 |
| `read_data.py` | 读取 NEDF3 元数据、MCAP 和内嵌 protobuf |
| `check_data.py` | 检查数据质量，验证转换结果 |
| `align_data.py` | 对齐图像、状态和动作，记录每帧来源 |
| `export_data.py` | 导出 HDF5 / LeRobot 和归一化统计 |
| `config.example.json` | 任务、相机尺寸、动作映射及对齐配置 |
| `requirements.txt` | Python 依赖 |
| `LICENSE.openpi` | 所整合 openpi 统计代码的许可证 |

## 命令行使用

在 `process_data/` 目录执行，使用 Python 3.10 或更新版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**1. 可视化与人工剔除。** 将数据放入 `dataset/` 后执行：

```bash
python main.py visualize
```

用视频播放器打开 `output/videos/episode_0000.mp4` 等文件，检查任务完成情况、画面异常和过短片段。左侧为左腕、右侧为右腕；`age` 表示图像距回放时刻的时间差。

需要剔除时，手动删除对应的 **`dataset/episode_0000/` 整个文件夹**，或将它移到 `dataset/` 之外。只删除视频不会剔除训练数据。程序不自动删除原始数据，也不需要网页审核或审核决定文件。

不同批次允许使用 `dataset/批次名/episode_0000/`，视频会保留同样的分组路径；`output/videos/video_index.json` 记录完整对应关系及导出失败原因。

**2. 检查数据（可独立运行）。** 按任务修改 `config.example.json`，核对相机尺寸、leader → follower 的方向和偏置。

```bash
python main.py check
```

查看 `output/check.json` 中的短片段、黑白帧、卡帧、夹爪突变等提示。文件损坏、缺流、时间戳异常或无法按配置对齐会报错。转换会重新检查，不依赖此报告。

**3. 转换并验证，按需要选择格式。** 转换处理 `dataset/` 中仍保留的全部 episode。

```bash
# ACT：输出到 output/hdf5/
python main.py convert --format hdf5
python main.py validate --data output/hdf5

# π0.5：输出到 output/lerobot/，无需先转 HDF5
python main.py convert --format lerobot
python main.py validate --data output/lerobot
```

已有输出不覆盖；重新处理可指定新的子目录，例如 `--output output/hdf5_v2`。失败时保留的 `.partial` 目录仅供诊断，不能用于训练。

## 训练数据说明

- HDF5 的 `episode_*.hdf5` 保存图像、状态和动作，`training_assets/` 保存整片段划分及训练集统计；至少需要两个 episode。
- LeRobot v2.1 的 `data/` 保存内嵌无损图像的 Parquet，`meta/` 保存数据集信息，`assets/` 保存 π0.5 统计。图像缩放补边为 224×224，当前配置使用全部片段训练。
- 两种输出均包含源片段清单、逐帧来源和配置。训练时保留整个 `output/hdf5/` 或 `output/lerobot/`，不要只取单个数据文件。

状态来自 follower，动作由 leader 按配置映射重建，均为 14 维：左 6 关节、左夹爪、右 6 关节、右夹爪。关节单位为弧度，夹爪范围为 `[0,1]`；动作不是最终下发命令日志。按 30 Hz 选最近的过去样本，视频不参与训练，ACT 与 π0.5 的归一化统计不能混用。

`dataset/`、`output/` 和 `.venv/` 用于使用者本地运行，不属于需要发布的源码。
