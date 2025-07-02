from typing import List, Optional, Dict
from dataclasses import dataclass
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

class DataConfig:
    """Configuration for data loading"""
    sequence_length: int = 2048
    vocab_size: int = 256000
    batch_size: int = 2
    num_samples: int = 100000

    dataset_name: Dict[str, float] = {"c4", 1.0}
    dataset_config: List[Optional[str]] = ["en"]
    tokenizer: Optional[str] = None

    streaming: bool = False
    cache_dir: Optional[str] = None # Optional for dev test

class TrainingConfig:
    """Configuration for training"""

class SystemConfig:
    """Configuration for System"""

@dataclass
class Config:
    """Main configuration combining all sub-configs"""
    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    training: TrainingConfig = MISSING
    system: SystemConfig = MISSING