"""读取项目内配置、管理缓存路径，并定义双腕模型与全量微调设置。"""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from .adapter import DEFAULT_PROMPT
from flax import nnx
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from .adapter import IMAGE_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """分块计算文件 SHA256。

    输入：path。
    输出：十六进制散列字符串。"""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fingerprint(value):
    """为可序列化配置生成稳定身份。

    输入：value。
    输出：SHA256 字符串。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    """经临时文件原子写入 JSON。

    输入：path、value。
    输出：无返回值，生成目标文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


@dataclasses.dataclass(frozen=True)
class Settings:
    dataset_root: str
    assets_dir: str
    run_dir: str
    repo_id: str = "local/five_blocks_into_cup"
    pretrained_params: str = "pretrained/params"
    tokenizer_path: str = "pretrained/tokenizer.model"
    prompt: str = DEFAULT_PROMPT
    fps: int = 30
    action_horizon: int = 50
    batch_size: int = 64
    num_steps: int = 20000
    decay_steps: int = 30000
    num_workers: int = 2
    seed: int = 42
    fsdp_devices: int = 1
    expected_devices: int = 1
    learning_rate: float = 2.5e-05
    end_learning_rate: float = 2.5e-06
    warmup_steps: int = 1000
    ema_decay: float = 0.99
    save_interval: int = 1000
    keep_period: int = 5000
    log_interval: int = 20
    benchmark_warmup: int = 10

    def __post_init__(self):
        """检查频率、学习率、步数与设备整除关系。

        输入：无外部参数，读取当前对象配置。
        输出：无返回值，非法配置抛出异常。"""
        for key in (
            "fps",
            "action_horizon",
            "batch_size",
            "num_steps",
            "decay_steps",
            "fsdp_devices",
            "expected_devices",
            "save_interval",
            "keep_period",
            "log_interval",
        ):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.fps != 30:
            raise ValueError("This data contract currently requires 30Hz")
        if (
            not isinstance(self.num_workers, int)
            or self.num_workers < 0
            or not 0 <= self.warmup_steps < self.num_steps
            or self.benchmark_warmup < 0
        ):
            raise ValueError("Invalid worker, warmup or step configuration")
        if self.decay_steps <= self.warmup_steps:
            raise ValueError("decay_steps must exceed warmup_steps")
        if not 0 <= self.ema_decay < 1 or not 0 < self.end_learning_rate <= self.learning_rate:
            raise ValueError("Invalid EMA or learning rate")
        if self.expected_devices % self.fsdp_devices or self.batch_size % self.expected_devices:
            raise ValueError("Device count must divide batch; FSDP count must divide device count")
        if not self.prompt.strip() or len(self.repo_id.split("/")) != 2 or ".." in self.repo_id.split("/"):
            raise ValueError("Provide a task prompt and namespace/dataset repo id")

    @classmethod
    def load(cls, path):
        """读取 JSON 并将相对资源路径按项目根目录解析。

        输入：path。
        输出：Settings 配置对象。"""
        path = Path(path).resolve()
        value = json.loads(path.read_text())
        for key in ("dataset_root", "assets_dir", "run_dir", "pretrained_params", "tokenizer_path"):
            if key not in value:
                field = next((f for f in dataclasses.fields(cls) if f.name == key))
                if field.default is dataclasses.MISSING:
                    raise ValueError(f"配置缺少 {key}")
                value[key] = field.default
            if "://" in value[key]:
                raise ValueError(f"{key} 必须是本地路径")
            p = Path(os.path.expandvars(value[key])).expanduser()
            value[key] = str((PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve())
        return cls(**value)

    def as_dict(self):
        """将训练配置转换为普通字典。

        输入：无外部参数，读取当前对象配置。
        输出：可序列化配置字典。"""
        return dataclasses.asdict(self)


def setup_caches(root):
    """为下载和编译缓存设置项目内默认目录。

    输入：root。
    输出：无返回值，设置尚未指定的环境变量。"""
    root = Path(root).expanduser().resolve()
    for name, folder in {
        "OPENPI_DATA_HOME": "openpi",
        "HF_HOME": "huggingface",
        "XDG_CACHE_HOME": "xdg",
        "JAX_COMPILATION_CACHE_DIR": "jax",
    }.items():
        os.environ.setdefault(name, str(root / folder))
    import jax

    jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])


@dataclasses.dataclass(frozen=True)
class DualWristPi05Config(Pi0Config):
    pi05: bool = True

    def __post_init__(self):
        """核对双腕 π0.5 的结构约束。

        输入：无外部参数，读取当前对象配置。
        输出：无返回值，要求 pi05 模式及 32 维动作。"""
        super().__post_init__()
        if not self.pi05 or self.action_dim != 32:
            raise ValueError("DualWristPi05Config requires pi05=True and action_dim=32")

    def inputs_spec(self, *, batch_size=1):
        """构造模型观测与动作的形状规格。

        输入：batch_size。
        输出：观测结构和动作 ShapeDtypeStruct。"""
        observation, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            return (
                dataclasses.replace(
                    observation,
                    images={k: observation.images[k] for k in IMAGE_KEYS},
                    image_masks={k: observation.image_masks[k] for k in IMAGE_KEYS},
                ),
                actions,
            )

    def create(self, rng):
        """初始化保持官方参数布局的双腕模型。

        输入：rng。
        输出：DualWristPi05 模型。"""
        from .model import DualWristPi05

        return DualWristPi05(self, rngs=nnx.Rngs(rng))


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """仅保留全量微调初始化和训练步骤使用的设置。"""

    model: DualWristPi05Config
    weight_loader: object
    lr_schedule: object
    optimizer: object
    ema_decay: float
    freeze_filter: object = dataclasses.field(default_factory=nnx.Nothing)

    @property
    def trainable_filter(self):
        """取得可训练参数过滤器；输入：自身冻结设置；输出：全部未冻结 NNX 参数。"""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))
