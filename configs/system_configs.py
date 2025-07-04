from configs.config_schema import SystemConfig

nscc_system = SystemConfig(
    accelerator="gpu",
    devices=8,
    num_nodes=1,
    precision="bf16-mixed",
    benchmark=True,
    deterministic=False
)

local_system = SystemConfig(
    accelerator="gpu",
    devices=1,
    num_nodes=1,
    precision="bf16-mixed",
    benchmark=True,
    deterministic=False
)
