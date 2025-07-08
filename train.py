"""
PEER Gemma Training Script with Hydra Configuration Management
Supports DeepSpeed, DDP, and NSCC PBS integration
"""

import os
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.integrations import TensorBoardCallback
import wandb
from loguru import logger
from hydra_zen import builds, store, zen
from omegaconf import OmegaConf, MISSING
from dotenv import load_dotenv

from configs.config_schema import DataConfig, Config
from models.peer_gemma import PEERGemmaForCausalLM
from data.data_module import TokenDataset
from trainers import GradualUnfreezingCallback

load_dotenv()
from configs.model_configs import gemma_2b_model, gemma_7b_model, gemma_9b_model, peered_model
from configs.data_configs import c4_data, c4_large_data
from configs.training_configs import full_training, quick_training
from configs.system_configs import nscc_system, local_system

cs = store(group="model")
cs(gemma_7b_model, name="gemma_7b")
cs(gemma_2b_model, name="gemma_2b")
cs(gemma_9b_model, name="gemma_9b")
cs(peered_model, name="peered_model")

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
    deepspeed_config="configs/deepspeed_z2.json",
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

def setup_pytorch_backend(cfg: Config):
    """Configure PyTorch backend settings safely based on available hardware"""

    if torch.cuda.is_available():
        if cfg.system.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("✅ CUDA deterministic mode enabled")
        else:
            torch.backends.cudnn.benchmark = cfg.system.benchmark
            logger.info(f"✅ CUDA benchmark: {cfg.system.benchmark}")
    else:
        logger.info(" Skipping CUDA backend config (CUDA not available)")

def setup_distributed():
    """Setup distributed training environment"""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger.info(f"Distributed setup: world_size={world_size}, rank={rank}, local_rank={local_rank}")

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return world_size, rank, local_rank

def setup_model_and_tokenizer(cfg: Config):
    """Setup PEER model and tokenizer"""
    model_config = cfg.model
    logger.info(f"Loading model: {model_config.model_name_or_path}")
    # Get HF token from environment
    hf_token = os.getenv('HF_TOKEN')
    hf_home = os.getenv('HF_HOME')
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        token=hf_token,
        cache_dir=hf_home,
        trust_remote_code=True
    )

    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model config
    config = AutoConfig.from_pretrained(
        model_config.model_name_or_path,
        token=hf_token,
        cache_dir=hf_home,
        trust_remote_code=True
    )

    # Create PEER model
    model = PEERGemmaForCausalLM.from_pretrained_with_peer_surgery(
        model_config.model_name_or_path,
        config=config,
        replace_layers=model_config.replace_layers,
        peer_config=model_config.peer_config,
        torch_dtype=getattr(torch, model_config.torch_dtype),
        device_map=model_config.device_map if model_config.device_map != "auto" else None,
        low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        token=hf_token,
        trust_remote_code=True
    )

    device = cfg.system.accelerator

    if device == "cuda":
        world_size, rank, local_rank = setup_distributed()
        if world_size > 1:
            model = model.to(f"cuda:{local_rank}")
        else:
            model = model.to("cuda")
    elif device == "cpu":
        model = model.to("cpu")

    logger.info(f"Model setup complete. Model device: {next(model.parameters()).device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")

    # Test forward pass
    try:
        dummy_input = torch.randint(0, 1000, (1, 10)).to(next(model.parameters()).device)
        with torch.no_grad():
            output = model(dummy_input)
        logger.info("✅ Forward pass test successful")
    except Exception as e:
        logger.error(f"❌ Forward pass test failed: {e}")
        raise

    return model, tokenizer, config


def create_dataset(data_config: DataConfig, tokenizer, split="train"):
    """Create dataset for training/validation"""
    return TokenDataset(
        sequence_length=data_config.sequence_length,
        batch_size=data_config.batch_size,
        num_samples=data_config.num_samples,
        dataset_name=data_config.dataset_name,
        dataset_config=data_config.dataset_config,
        tokenizer=tokenizer,
        cache_dir=data_config.cache_dir or os.getenv('HF_DATASETS_CACHE'),
        streaming=data_config.streaming,
        split=split,
        seed=data_config.seed
    )


def setup_training_args(cfg: Config, output_dir: str, logging_dir: str):
    """Setup TrainingArguments with DeepSpeed integration"""

    # Get actual batch size (with overrides)
    batch_size = cfg.data.batch_size
    learning_rate = cfg.training.learning_rate

    # Calculate gradient accumulation steps for effective batch size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_device_batch_size = batch_size
    gradient_accumulation_steps = max(1, batch_size // (per_device_batch_size * world_size * cfg.system.devices))

    logger.info(f"Batch size calculation:")
    logger.info(f"  Target batch size: {batch_size}")
    logger.info(f"  World size: {world_size}")
    logger.info(f"  Devices per node: {cfg.system.devices}")
    logger.info(f"  Per device batch size: {per_device_batch_size}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")

    run_name = f"{cfg.model.model_name_or_path.split('/')[-1]}"
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,

        # Training params
        # num_train_epochs=cfg.training.max_epochs,
        max_steps=25000,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        max_grad_norm=cfg.training.gradient_clip_val,

        # Distributed training
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),

        # Mixed precision
        fp16=cfg.system.precision == "fp16",
        bf16=cfg.system.precision in ["bf16", "bf16-mixed"],

        # Logging
        logging_steps=10,
        logging_first_step=True,
        eval_steps=500,
        save_steps=500,
        run_name=run_name,
        save_total_limit=cfg.training.save_top_k,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.training.monitor,
        greater_is_better=(cfg.training.mode == "max"),

        # Performance
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
        remove_unused_columns=False,

        # DeepSpeed integration
        deepspeed=cfg.deepspeed_config if cfg.deepspeed_config else None,

        # Reporting
        report_to=["wandb", "tensorboard"],

        # Debug
        disable_tqdm=False,
    )

    return training_args


def setup_wandb(cfg: Config):
    """Setup Weights & Biases logging"""
    if not cfg.debug:
        # Get WANDB API key from environment
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)

            config_dict = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=config_dict,
                tags=["peer", "gemma", "mixture-of-experts"]
            )
        else:
            logger.warning("WANDB_API_KEY not found, skipping W&B logging")


def create_unfreezing_callback(model):
    """Create unfreezing callback using model's PEER info"""

    peer_info = model.get_peer_info()
    peer_layer_indices = peer_info["replaced_layer_indices"]
    total_layers = len(model.model.layers)

    logger.info(f"Creating unfreezing callback for PEER layers: {peer_layer_indices}")

    # Create middle-out unfreezing schedule
    unfreezing_schedule = {}

    for i, step in enumerate(range(500, 5000, 500)):
        layers_to_unfreeze = []
        for peer_idx in peer_layer_indices:
            if peer_idx - i - 1 >= 0:
                layers_to_unfreeze.append(f'layers.{peer_idx - i - 1}')
            if peer_idx + i + 1 < total_layers:
                layers_to_unfreeze.append(f'layers.{peer_idx + i + 1}')

        if layers_to_unfreeze:
            unfreezing_schedule[step] = layers_to_unfreeze

    return GradualUnfreezingCallback(model, unfreezing_schedule)

def train_task(model, data, training, system, deepspeed_config, output_dir, logging_dir,
               wandb_project, wandb_entity, debug=False):
    """Main training function"""
    cfg = Config(
        model=model, data=data, training=training, system=system,
        deepspeed_config=deepspeed_config, output_dir=output_dir,
        logging_dir=logging_dir, wandb_project=wandb_project,
        wandb_entity=wandb_entity, debug=debug
    )

    # Setup logging
    logger.info("Starting PEER Gemma training")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Create output directories
    output_dir = Path(cfg.output_dir)
    logging_dir = Path(cfg.logging_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    setup_pytorch_backend(cfg)
    # Setup distribute training
    world_size, rank, local_rank = setup_distributed()

    # Setup W&B on rank 0 only
    if rank == 0:
        setup_wandb(cfg)

    model, tokenizer, model_config = setup_model_and_tokenizer(cfg)

    # Create datasets
    train_dataset = create_dataset(cfg.data, tokenizer, split="train")
    eval_data_config = cfg.data
    eval_data_config.num_samples = 1000
    eval_dataset = create_dataset(eval_data_config, tokenizer, split="validation")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )

    # Setup training arguments
    training_args = setup_training_args(cfg, str(output_dir), str(logging_dir))

    # unfreezing schedule
    callback = create_unfreezing_callback(model)
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.add_callback(callback)
    if not cfg.debug:
        trainer.remove_callback(TensorBoardCallback)

    # Save config
    if rank == 0:
        config_save_path = output_dir / "training_config.yaml"
        with open(config_save_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=False))
        logger.info(f"Config saved to {config_save_path}")

    # Start training
    try:
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        if rank == 0:
            final_model_path = output_dir / "final_model"
            trainer.save_model(str(final_model_path))
            tokenizer.save_pretrained(str(final_model_path))
            logger.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        # Cleanup
        if world_size > 1:
            dist.destroy_process_group()

        if rank == 0 and not cfg.debug:
            wandb.finish()


if __name__ == "__main__":
    # Add all store configs to Hydra
    store.add_to_hydra_store()

    # Use zen decorator to create CLI
    zen(train_task).hydra_main(
        config_path=None,
        config_name="config",
        version_base="1.1"
    )