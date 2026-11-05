# π0.5 双腕训练

独立的 I2RT 双臂 π0.5 全量微调项目。沿用 OpenPI 的 JAX/Flax 网络、流匹配训练和 EMA，只输入左右腕两路图像；不需要另外放置官方仓库，也不连接机器人。

## 文件作用

| 路径 | 作用 |
|---|---|
| `configs/train.json` | 数据、基础权重、训练输出路径及训练参数 |
| `src/config.py` | 配置校验、项目内路径、双腕模型规格 |
| `src/dataset.py` | LeRobot 读取、片段内动作块、断点采样 |
| `src/adapter.py` | 图像转换、相对关节动作、归一化及提示词转换链 |
| `src/model.py` | 保持官方参数布局的双腕 π0.5 模型 |
| `src/train.py` | GPU 训练、日志和检查点保存 |
| `src/checkpoint.py` | 代码、数据、资源和环境身份校验 |
| `src/check_data.py` | 真实数据完整性、数值与图像检查 |
| `src/pretrained.py` | 官方基础权重及分词器下载、续传、校验 |
| `src/export.py` | 导出和验证包含 EMA 权重的 ZIP |
| `vendor/` | 实际使用的官方网络和训练工具；来源、原始散列见 `UPSTREAM_FILES.json` |
| `dataset/` | 放置已经转换并人工筛选的完整 LeRobot 数据 |
| `pretrained/` | 基础权重 `params/` 与 `tokenizer.model` |
| `output/` | 训练记录、检查报告、缓存和模型包 |

`src/` 安装为 `i2rt_pi05`，`vendor/` 安装为 `openpi`。后者是裁剪并适配后的官方代码，保留授权文件，并非完整原仓库。不要在同一环境安装另一份 `openpi`。项目没有测试目录、合成训练数据或仿真入口；模型初始化所需的形状占位张量不用于训练。

## 数据放置与含义

将 `process_data` 的 **LeRobot 转换结果目录内部全部内容**复制到 `dataset/`；本项目不调用 `process_data`，复制完成即可独立使用。不要放原始 MCAP 或 ACT 的 HDF5。

```text
dataset/
├── data/chunk-000/episode_000000.parquet
├── meta/                     # info.json、episodes/tasks/stats 等
├── assets/
│   ├── norm_stats.json
│   └── provenance.json
├── i2rt_manifest.json
└── ...                       # 同时保留其他清单和对齐来源文件
```

每帧为两路 `224×224` RGB 腕部图像、14 维 follower 实测状态与14维绝对目标。顺序均为“左臂六关节、左夹爪、右臂六关节、右夹爪”；关节用 rad，夹爪范围 `[0,1]`。目标由采集时的 leader 记录按遥操作映射重建，不能当作最终下发命令日志或末端笛卡尔位姿。

读取时使用 30 Hz、默认 50 步的未来动作块，不跨 episode，尾部重复最后动作。关节目标减去该观测时刻的 follower 状态，夹爪保持绝对目标，再使用配套 q01/q99 统计量归一化。任务文本与归一化状态一起编码，状态和动作补到模型的 32 维。输出经反归一化和关节差值还原后得到14维绝对目标。

当前沿用 `all_train`：全部已筛选片段用于训练，没有验证集，也不自动选“验证最优模型”。若更改任务、动作映射或动作跨度，应重新转换并生成匹配的统计量；本项目不会自动修改统计量或剔除片段。

## 安装与资源准备

以下在 **pi05 项目根目录**执行。使用 Linux、Python 3.11、独立环境；JAX 使用 GPU，Torch 只承担数据读取。不要改动 ACT 的环境。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -c 'import jax; print(jax.devices())'
```

最后必须显示 GPU 设备。当前配置以单张 H20 为目标，未在这次目录整理中重新完成 H20 全量训练验证。LeRobot 固定到指定提交以保持 v2.1 数据接口；不要直接升级为新版。

下载官方基础资源（约 12.4 GB 权重，另有分词器；中断后重跑同一命令续传）：

```bash
python -m i2rt_pi05.pretrained download
python -m i2rt_pi05.pretrained check
```

也可把已有的官方 `pi05_base/params/` 完整目录和 `paligemma_tokenizer.model` 分别复制为 `pretrained/params/` 和 `pretrained/tokenizer.model`。不得只复制某个参数分片。手动复制且没有下载清单时，`check` 只检查资源结构和分词器；正式初始化仍会核对完整参数结构。

## 检查和训练

先确认 `configs/train.json` 中 `repo_id`、`prompt` 与 `dataset/i2rt_manifest.json` 一致。配置中的相对路径按项目根目录解析。

```bash
python -m i2rt_pi05.check_data --deep
```

报告位于 `output/data_check.json`。`passed` 表示格式、文件散列及数值检查通过；若 `requires_review` 为真，按报告定位黑白帧或退化统计维度，人工检查原始片段，再重新转换。该检查不能判断任务是否成功。

默认保留：batch 64、20,000 更新步、动作跨度50、学习率 `2.5e-5`、1,000步预热、30,000步余弦衰减计划、EMA 0.99。优化器沿用官方 AdamW，包括全局梯度范数裁剪1.0；并未照搬 ACT 的优化器设置。

```bash
mkdir -p output/runs
tmux new -s pi05
```

在 tmux 中执行：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m i2rt_pi05.train \
  --config configs/train.json \
  2>&1 | tee output/runs/pi05_v1_console.log
```

按 `Ctrl+B` 后按 `D` 离开 tmux，随后可以断开 SSH；训练实例仍需保持运行。查看进度：

```bash
tmux attach -t pi05
tail -n 20 output/runs/pi05_v1/metrics.jsonl
```

每1,000步保存检查点，保留最近检查点和5,000步的整数倍检查点。日志包括 loss、梯度范数、计算时间；首次编译和保存权重可能较慢。模型中断后，从最后一个完成保存的检查点恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m i2rt_pi05.train \
  --config configs/train.json --resume
```

恢复时配置、代码、数据、统计量、基础权重、分词器与关键依赖版本必须匹配。旧项目的运行身份与新目录不同，不能把旧训练目录直接交给 `--resume`。重新训练请在配置中指定新的 `run_dir`，不要覆盖旧运行。

## 取出模型

```bash
python -m i2rt_pi05.export create \
  --run output/runs/pi05_v1 \
  --output output/models/pi05.zip
python -m i2rt_pi05.export verify output/models/pi05.zip
```

默认选择最新的完整检查点，也可加 `--step 20000`。拿回整个 `output/models/pi05.zip`，其中有 `params/`、`assets/`、`tokenizer.model`、输入输出契约、校验清单和本项目必要源码。此模型是 **JAX Orbax 参数目录及分片，不是 `.pt` 文件**。导出包不包含优化器；若需继续训练，应另外保留完整的训练运行目录和原训练资源。

模型包供后续部署适配读取；本项目不包含硬件控制或部署 UI。真实任务效果仍需通过实际模型评估确认。
