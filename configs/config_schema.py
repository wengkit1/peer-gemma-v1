from typing import List, Optional, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra_zen import store, builds

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
    vocab_size: int = 256000
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
    max_epochs: int = 3
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
    strategy: str = "ddp"
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
    deepspeed_config: Optional[str] = "configs/deepspeed_z2.yaml"

    # Project names (did not put in config)
    wandb_project: str = "peer-gemma-9b-nscc"
    wandb_entity: str = "naqibl-nus"

    # Debug
    debug: bool = False

from configs.model_configs import gemma_2b_model, gemma_7b_model, gemma_9b_model
from configs.data_configs import c4_data, c4_large_data
from configs.training_configs import full_training, quick_training
from configs.system_configs import nscc_system, local_system

def build_main_store():
    cs = store(group="model")
    cs(gemma_7b_model, name="gemma_7b")
    cs(gemma_2b_model, name="gemma_2b")
    cs(gemma_9b_model, name="gemma_9b")

    cs = store(group="data")
    cs(c4_data, name="c4")
    cs(c4_large_data, name="c4_large")

    cs = store(group="training")
    cs(full_training, name="full")
    cs(quick_training, name="quick")

    cs = store(group="system")
    cs(nscc_system, name="nscc")
    cs(local_system, name="local")

    main_config = builds(
        Config,
        model=MISSING,
        data=MISSING,
        training=MISSING,
        system=MISSING,
        deepspeed_config="configs/deepspeed_z2.yaml",
        output_dir="${oc.env:SCRATCH_DIR,/tmp}/peer_gemma_experiments/checkpoints",
        logging_dir="${oc.env:SCRATCH_DIR,/tmp}/peer_gemma_experiments/logs",
        wandb_project="peer-gemma-9b-nscc",
        wandb_entity="naqibl-nus",
        hydra_defaults=[
            "_self_",
            {"model": "gemma_9b"},
            {"data": "c4"},
            {"training": "full"},
            {"system": "nscc"}
        ]
    )

    cs = store()
    cs(main_config, name="config")