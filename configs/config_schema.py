from typing import List, Optional, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class ModelConfig:
    """Configuration for PEER surgery on pre-trained models"""

    # Model loading
    model_name_or_path: str = "google/gemma-2-9b"

    replace_layers: List[int] = field(default_factory=lambda: [6, 9])

    peer_config: Optional[dict] = None

    torch_dtype: str = "bfloat16"
    low_cpu_mem_usage: bool = True
    device_map: Optional[str] = "auto"

@dataclass
class DataConfig:
    """Configuration for data loading"""
    sequence_length: int = 2048
    batch_size: int = 2
    num_samples: int = 100000

    dataset_name: Optional[Dict[str, float]] = field(default_factory=dict)
    dataset_config: Optional[List[str]] = field(default_factory=list)
    tokenizer: Optional[str] = None

    streaming: bool = False
    cache_dir: Optional[str] = None

    seed: int = 42

@dataclass
class TrainingConfig:
    """Configuration for training"""
    max_steps: int = 30000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: float = 0.25
    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: str = "min"

@dataclass
class SystemConfig:
    """Configuration for System"""
    accelerator: str = "gpu"
    devices: int = 8
    num_nodes: int = 1
    precision: str = "bf16-mixed"
    benchmark: bool = True
    deterministic: bool = False

# Main config schema
@dataclass
class Config:
    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    training: TrainingConfig = MISSING
    system: SystemConfig = MISSING

    # Training arguments
    output_dir: str = "${oc.env:SCRATCH_DIR,/tmp}/peer_gemma_experiments/checkpoints"
    logging_dir: str = "${oc.env:SCRATCH_DIR,/tmp}/peer_gemma_experiments/logs"
    deepspeed_config: Optional[str] = "configs/deepspeed_z2.json"

    # Project names (did not put in config)
    wandb_project: str = "peer-gemma-9b-nscc"
    wandb_entity: str = "naqibl-nus"

    # Debug
    debug: bool = False