from configs.config_schema import TrainingConfig

full_training = TrainingConfig(
    max_steps=15000,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    gradient_clip_val=1.0,
    accumulate_grad_batches=64, # 64 * 8 nodes * mbs * 4096 seq len
    val_check_interval=500,
    save_top_k=3,
    monitor="eval_loss",
    mode="min",
)

quick_training = TrainingConfig(
    max_steps=300,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=100,
    gradient_clip_val=1.0,
    accumulate_grad_batches=64,
    val_check_interval=50,
    save_top_k=1,
    monitor="eval_loss",
    mode="min"
)
