# ACT 训练

独立完成 ACT 的数据读取、训练、验证、断点续训和 ZIP 模型导出。输入为已处理的 HDF5；不依赖数据处理项目或完整官方仓库，不连接真机。

## 目录与文件

```text
act/
├── configs/train.json       # 正式训练配置
├── src/                     # 数据读取与训练代码，职责见下表
├── vendor/                  # 实际使用的官方 ACT 网络及许可证
├── dataset/                # 用户放入的完整 HDF5 数据集，初始为空
├── output/                 # 训练结果与模型包，初始为空
├── requirements.txt        # 正式运行依赖
├── pyproject.toml          # 包安装及命令行入口
└── README.md
```

| 文件 | 作用 |
| --- | --- |
| `config.py` | 读取配置；相对数据路径以 `act/` 为基准 |
| `dataset.py` | 双腕图像、14 维状态/动作读取，归一化、动作块填充及片段级采样 |
| `model.py` | ACT 封装、图像归一化、L1＋KL 损失和 AdamW 优化器 |
| `train.py` | 训练、验证、日志、最佳模型选择和续训 |
| `checkpoint.py` | 原子保存检查点，恢复优化器和随机状态 |
| `check_data.py` | 检查实际训练数据、划分和归一化统计 |
| `export.py` | 严格加载权重验证后，导出 ZIP 模型包 |
| `vendor/detr/models/` | 图像骨干、正弦位置编码、Transformer 和 ACT CVAE |

保留官方来源和许可证，删除未使用的 CNNMLP 与可学习图像位置编码分支。项目不附带测试目录、合成数据生成代码、冒烟配置或仿真资产。

## 准备数据与环境

将数据处理工具输出的 **HDF5 目录内全部内容** 放入 `dataset/`，包括：

```text
dataset/
├── episode_0.hdf5
├── episode_1.hdf5
├── ...
└── training_assets/
    ├── dataset_manifest.json
    ├── splits.json
    ├── normalization.json
    ├── dataset_stats.pkl
    └── conversion_config.json
```

`training_assets/` 必须对应这批 HDF5，不沿用其他数据集的统计。状态和动作均为 14 维；图像为 `left_wrist`、`right_wrist` 两路 RGB。数据已经完成 30 Hz 对齐，训练时不再移动动作一帧。

在本子项目目录执行，使用 Python 3.10–3.12：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e .
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

`src/` 直接存放训练代码，安装时映射为 Python 包 `i2rt_act`，因此命令使用 `python -m i2rt_act...`。保持整个项目目录，以可编辑方式安装；复制到训练服务器后在服务器重新创建环境。不要复制其他机器的虚拟环境。默认配置使用单张 GPU，开始训练前 CUDA 检查应输出 `True`。

## 检查和训练

修改 `configs/train.json` 中的任务名称及需要调整的参数，然后检查输入：

```bash
python -m i2rt_act.check_data --deep
```

深度检查读取全部图像和数值，复核时间网格及仅由训练集计算的均值/标准差。普通检查省略 `--deep`。这不会生成数据或启动试验训练。

```bash
python -u -m i2rt_act.train --output output/runs/act_v1
```

默认保留既有训练设置：ResNet18 预训练、动作块 100、隐藏维度 512、前馈维度 3200、编码器 4 层、解码器 7 层、KL 权重 10、batch 8、学习率 `1e-5`、5000 epoch、无梯度裁剪。首次使用预训练骨干时，torchvision 需要下载权重或读取已有缓存。

每轮先验证再训练，每个片段随机取一个起点；验证损失用于选择最佳模型。`output/runs/act_v1/metrics.jsonl` 记录各轮指标。完整续训状态默认每 100 轮及最后一轮保存，`policy_last.ckpt` 对应最近保存的检查点，不保证是终端显示的最新一轮。

远程训练如需在断开 SSH 后继续，可在服务器的 `tmux` 会话中运行：

```bash
tmux new -s act
# 在会话内激活 .venv 并执行训练；Ctrl+B 后按 D 脱离会话。
# 再次查看：tmux attach -t act
```

**中断后续训：** 恢复优化器和随机状态，从最近保存的完整轮次继续。

```bash
python -m i2rt_act.train --resume output/runs/act_v1/resume.ckpt
```

**使用已有权重开始新的训练：** 使用新优化器，可更换数据集；网络尺寸必须兼容。

```bash
python -m i2rt_act.train \
  --init-from output/runs/act_v1/policy_best.ckpt \
  --output output/runs/act_v2
```

## 模型输出

训练目录中的主要文件：

| 文件 | 用途 |
| --- | --- |
| `policy_best.ckpt` | 验证损失最低的权重 |
| `policy_last.ckpt` | 最近保存的训练权重 |
| `resume.ckpt` | 权重、优化器、随机状态等完整续训信息 |
| `config.json`、`dataset_stats.pkl` | 模型配置和归一化参数 |
| `dataset_assets/` | 本次使用的数据清单、划分和统计副本 |
| `metrics.jsonl`、`summary.json` | 指标记录及完成摘要 |
| `provenance.json` | 运行环境、代码和数据指纹 |

训练完成后导出：

```bash
python -m i2rt_act.export --run output/runs/act_v1
```

默认生成 `output/models/act_v1.zip`，包含 `run/` 下的权重、配置、统计，以及 `code/` 下的必要源码。需要携带续训状态时增加 `--include-resume`。`.ckpt` 是 PyTorch 保存的文件，不必改名为 `.pt`；部署需要权重、模型配置和统计共同使用。

`dataset/`、`output/`、`.venv/` 是运行时目录，不属于需要发布的数据或源码内容。
