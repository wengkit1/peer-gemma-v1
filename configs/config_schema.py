from typing import List, Optional, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING
# PEER Surgery Configuration
@dataclass
class ModelConfig:
    """Configuration for PEER surgery on pre-trained models"""

    # Model loading
    model_name_or_path: str = "google/gemma-2-9b"

    replace_layers: List[int] = List[6, 9]

    peer_config: dict = None

    torch_dtype: str = "bfloat16"
    low_cpu_mem_usage: bool = True
    device_map: Optional[str] = "auto"

@dataclass
class DataConfig:
    """Configuration for data loading"""
    sequence_length: int = 2048
    vocab_size: int = 256000
    batch_size: int = 2
    num_samples: int = 100000

    dataset_name: Optional[Dict[str, float]] = field(default_factory=dict)
    dataset_config: Optional[List[str]] = field(default_factory=list)
    tokenizer: Optional[str] = None

    streaming: bool = False
    cache_dir: Optional[str] = None # Optional for dev test

    seed: int = 42

@dataclass
class TrainingConfig:
    """Configuration for training"""
    max_epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    val_check_interval: float = 0.25,
    save_top_k: int = 3,
    monitor="val_loss",
    mode="min"

@dataclass
class SystemConfig:
    """Configuration for System"""

@dataclass
class Config:
    """Main configuration combining all sub-configs"""
    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    training: TrainingConfig = MISSING
    system: SystemConfig = MISSING