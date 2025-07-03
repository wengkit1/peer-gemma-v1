from configs.config_schema import TrainingConfig

full_training = TrainingConfig(
    max_epochs=3,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
    val_check_interval=0.25,
    save_top_k=3,
    monitor="val_loss",
    mode="min",
)

quick_training = TrainingConfig(
    max_epochs=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=100,
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
    val_check_interval=0.5,
    save_top_k=1,
    monitor="val_loss",
    mode="min"
)
