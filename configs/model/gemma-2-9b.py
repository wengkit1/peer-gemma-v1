from typing import List, Optional
from dataclasses import dataclass

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
