from configs.config_schema import ModelConfig

gemma_2b_model = ModelConfig(
    model_name_or_path="google/gemma-2-2b",
    replace_layers=[3, 6],
    peer_config={
        "num_experts": 32768,
        "num_experts_per_head": 8,
        "heads": 8,
        "dim_key": 64
    },
    torch_dtype="bfloat16",
    low_cpu_mem_usage=True,
    device_map="auto"
)

gemma_7b_model = ModelConfig(
    model_name_or_path="google/gemma-2-7b",
    replace_layers=[6, 9, 12],  # More layers for bigger model
    peer_config={
        "num_experts": 65536,  # 256^2 as per paper
        "num_experts_per_head": 16,
        "heads": 16,
        "dim_key": 128
    },
    torch_dtype="bfloat16",
    low_cpu_mem_usage=True,
    device_map="auto"
)

gemma_9b_model = ModelConfig(
    model_name_or_path="google/gemma-2-9b",
    replace_layers=[6, 9],  # As in your original config
    peer_config={
        "num_experts": 65536,
        "num_experts_per_head": 16,
        "heads": 16,
        "dim_key": 128
    },
    torch_dtype="bfloat16",
    low_cpu_mem_usage=True,
    device_map="auto"
)
