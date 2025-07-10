from configs.config_schema import DataConfig

c4_data = DataConfig(
    sequence_length=2048,
    batch_size=2,
    num_samples=100000,
    dataset_name={"allenai/c4": 1.0},
    dataset_config=["en"],
    tokenizer="google/gemma-2-9b",
    streaming=True,
    cache_dir=None,  # Will use env var or default
    seed=42
)

c4_large_data = DataConfig(
    sequence_length=8192,
    batch_size=1,
    num_samples=1000000,
    dataset_name={"allenai/c4": 1.0},
    dataset_config=["en"],
    tokenizer="google/gemma-2-9b",
    streaming=True,
    cache_dir=None,
    seed=42
)




