# MVIG 真机实践

本仓库为 2026 MVIG 真机实践的个人仓库，不包含其余组员项目成果。

本项目围绕 I2RT 双臂机器人的随机位置多积木入杯任务，主要包含数据处理与模型训练的相关代码，数据处理与部署管线的代码在部署机器，不存放在本仓库中。

## 项目目录

```text
MVIGinternship-Linkzhu-main/
├── README.md               # 项目总览、使用顺序和开发约定
├── process_data/           # NEDF3 可视化、质量检查、时间对齐与格式转换
│   ├── dataset/            # 使用者放入原始 episode，初始为空
│   └── output/             # 生成视频、报告、HDF5 或 LeRobot 数据
├── act/                    # ACT 训练、验证、续训与模型导出
│   ├── configs/            # 训练配置
│   ├── src/                # 本项目训练适配代码
│   ├── vendor/             # 实际使用的官方 ACT 网络代码及许可证
│   ├── dataset/            # 完整的已处理 HDF5 数据集，初始为空
│   └── output/             # 训练日志、检查点和模型 ZIP
├── pi05/                   # 双腕 π0.5 全量微调、续训与模型导出
│   ├── configs/            # 训练配置
│   ├── src/                # 本项目数据与训练适配代码
│   ├── vendor/             # 实际使用的 OpenPI 组件及许可证
│   ├── dataset/            # 完整的已处理 LeRobot 数据集，初始为空
│   ├── pretrained/         # 使用者准备基础权重和分词器，初始为空
│   └── output/             # 训练日志、检查点和模型 ZIP
└── demo/                   # 任务演示视频与关键帧
    ├── README.md           # 演示内容与文件说明
    └── keyframes/          # ACT 与 π0.5 关键帧及展示拼图
```

各子项目有独立的依赖与说明，通过数据文件或模型包衔接，不需要通过绝对路径引用其他子项目源码。

| 子项目 | 输入 | 主要输出 | 使用说明 |
| --- | --- | --- | --- |
| data_process | 完整的 NEDF3 episode 文件夹 | 双腕视频、检查报告、对齐后的训练数据及统计量 | [process_data/README.md](process_data/README.md) |
| ACT | HDF5 数据与配套 `training_assets/` | PyTorch `.ckpt` 权重及 ZIP 模型包 | [act/README.md](act/README.md) |
| π0.5 | LeRobot v2.1 数据、配套统计量与基础权重 | JAX Orbax `params/` 参数目录及 ZIP 模型包 | [pi05/README.md](pi05/README.md) |

## 使用顺序

### 1. 准备并人工筛选示范

将每个原始 episode 的完整文件夹放入 `process_data/dataset/`，保留元数据、MCAP 索引及其全部分片。按[数据处理说明](process_data/README.md)安装依赖，在该子项目内执行：

```bash
python main.py visualize
python main.py check
```

使用视频播放器查看 `output/videos/`，结合 `output/check.json` 检查任务是否成功、片段是否过短，以及黑帧、卡帧和数据异常。

剔除数据时，将对应 episode 文件夹移出 `dataset/`，或删除该文件夹；**只删除视频不会排除训练数据**。审核与转换是独立步骤。

### 2. 选择转换格式

在 `process_data/` 内执行需要的分支：

```bash
# ACT
python main.py convert --format hdf5
python main.py validate --data output/hdf5

# π0.5，无需先转换为 HDF5
python main.py convert --format lerobot
python main.py validate --data output/lerobot
```

核对 `config.example.json` 中的任务、相机和动作映射配置后再转换。将转换结果目录的**全部内容**放入对应训练项目：

```text
process_data/output/hdf5/   → act/dataset/
process_data/output/lerobot/ → pi05/dataset/
```

配套清单、划分、归一化统计和逐帧来源需与数据一起保留，不能只传单个 HDF5 或 Parquet 文件。

### 3. 训练并导出模型

分别按照 [ACT 说明](act/README.md)或 [π0.5 说明](pi05/README.md)创建独立环境、检查数据并启动训练。训练输出写入各自的 `output/`。

| 项目 | ACT 当前配置 | π0.5 当前配置 |
| --- | --- | --- |
| 训练框架 | PyTorch | JAX / Flax |
| 图像输入 | 左右腕 RGB，480×640 | 左右腕 RGB，224×224 |
| 动作块长度 | 100 步 | 50 步 |
| 数据划分 | 按完整 episode 划分训练集与验证集 | `all_train`，使用全部已筛选片段 |
| 归一化 | 训练集均值、标准差 | 配套 q01 / q99 分位数 |
| 模型选取 | 验证损失最低的权重 | 指定保存步数，默认导出最新完整检查点的 EMA 权重 |

两条训练管线的数据处理和模型选取方式不同，训练损失不能直接作为两种模型任务表现的比较依据。

导出的模型统一使用 ZIP。ACT 的 `.ckpt` 是 PyTorch 权重文件，无需改名为 `.pt`；π0.5 的权重为 Orbax 参数目录及分片。后续使用模型时需同时保留配置、归一化统计，以及 π0.5 的分词器，不能只取一个权重分片。

## 展示与验证状态

`demo/` 已包含 ACT 与 π0.5 的演示视频、关键帧和展示拼图，具体内容见 [演示说明](demo/README.md)。这些演示用于展示任务现象，仓库暂不填写未经统计的任务成功率或受控模型比较结论。
