from typing import Dict, List, Optional, Union
import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from loguru import logger
import random
import numpy as np


class TokenDataset(IterableDataset):
    def __init__(self,
                 sequence_length: int = 2048,
                 vocab_size: int = 256000,
                 batch_size: int = 2,
                 num_samples: int = 100000,
                 dataset_name: Optional[Union[str, Dict[str, float]]] = None,
                 dataset_config: Optional[List[str]] = None,
                 tokenizer = None,
                 cache_dir: Optional[str] = None,
                 streaming: bool = False,
                 split: str = "train",
                 seed: int = 42):

        # Set defaults
        if dataset_name is None:
            dataset_name = {"allenai/c4": 1.0}
        elif isinstance(dataset_name, str):
            dataset_name = {dataset_name: 1.0}

        if dataset_config is None:
            dataset_config = ["en"]

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_samples = num_samples

        _scratch_dir = os.getenv("SCRATCH_DIR", os.path.expanduser("~/scratch"))
        default_cache = f"{_scratch_dir}/.cache/huggingface/hub/datasets"
        self.datasets_cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE", default_cache)

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.streaming = streaming
        self.split = split
        self.seed = seed

        # Initialize tokenizer
        if tokenizer is None:
            # Only create new tokenizer if none provided
            tokenizer_name = "google/gemma-2-9b"
            logger.info(f"Initializing tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                token=os.getenv("HF_TOKEN"),
                cache_dir=os.getenv("HF_HOME"),
                trust_remote_code=True
            )
        else:
            # Use the passed tokenizer object
            self.tokenizer = tokenizer
            logger.info("Using provided tokenizer")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"Configured vocab size: {vocab_size}")

        # Load datasets
        self.datasets = self._load_datasets()

    def _load_datasets(self):
        """Load and combine multiple datasets according to proportions"""
        datasets = {}

        for dataset_name, proportion in self.dataset_name.items():
            logger.info(f"Loading dataset: {dataset_name} (proportion: {proportion})")

            # Handle different dataset configurations
            if dataset_name == "allenai/c4":
                # C4 dataset with language config
                for config in self.dataset_config:
                    logger.info(f"Loading C4 with config: {config}")
                    try:
                        dataset = load_dataset(
                            dataset_name,
                            config,
                            split=self.split,
                            streaming=self.streaming,
                            cache_dir=self.datasets_cache_dir,
                            token=os.getenv("HF_TOKEN")
                        )
                        datasets[f"{dataset_name}_{config}"] = {
                            "dataset": dataset,
                            "proportion": proportion
                        }
                        logger.success(f"Successfully loaded {dataset_name}_{config}")
                    except Exception as e:
                        logger.error(f"Failed to load {dataset_name}_{config}: {e}")
                        # Fallback to default C4
                        dataset = load_dataset(
                            "allenai/c4",
                            "en",
                            split=self.split,
                            streaming=self.streaming,
                            cache_dir=self.datasets_cache_dir,
                            token=os.getenv("HF_TOKEN")
                        )
                        datasets[f"c4_en"] = {
                            "dataset": dataset,
                            "proportion": proportion
                        }
            else:
                # Other datasets
                try:
                    dataset = load_dataset(
                        dataset_name,
                        split=self.split,
                        streaming=self.streaming,
                        cache_dir=self.datasets_cache_dir,
                        token=os.getenv("HF_TOKEN")
                    )
                    datasets[dataset_name] = {
                        "dataset": dataset,
                        "proportion": proportion
                    }
                    logger.success(f"Successfully loaded {dataset_name}")
                except Exception as e:
                    logger.error(f"Failed to load {dataset_name}: {e}")

        return datasets

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text and prepare for training"""
        # Tokenize with truncation and padding
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.sequence_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        labels = input_ids.clone()

        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def __iter__(self):
        """Iterator for the dataset"""
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        total_proportion = sum(data["proportion"] for data in self.datasets.values())
        sampling_probs = {
            name: data["proportion"] / total_proportion
            for name, data in self.datasets.items()
        }

        dataset_names = list(self.datasets.keys())
        probabilities = list(sampling_probs.values())

        # Create iterators for each dataset
        dataset_iters = {
            name: iter(data["dataset"])
            for name, data in self.datasets.items()
        }

        samples_yielded = 0
        while samples_yielded < self.num_samples:
            try:
                # Sample a dataset according to proportions
                selected_dataset = np.random.choice(dataset_names, p=probabilities)

                # Get next sample from selected dataset
                sample = next(dataset_iters[selected_dataset])

                # Extract text (handle different dataset formats)
                if "text" in sample:
                    text = sample["text"]
                elif "content" in sample:
                    text = sample["content"]
                else:
                    # Concatenate all string fields
                    text = " ".join([str(v) for v in sample.values() if isinstance(v, str)])

                if not text or len(text.strip()) < 10:
                    continue

                tokenized = self._tokenize_text(text)
                yield tokenized

                samples_yielded += 1

            except StopIteration:
                # If a dataset iterator is exhausted, recreate it
                dataset_iters[selected_dataset] = iter(self.datasets[selected_dataset]["dataset"])
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue


class DataModule:
    """DataModule for PEER-Gemma training"""

    def __init__(self,
                 sequence_length: int = 2048,
                 vocab_size: int = 256000,
                 batch_size: int = 2,
                 num_samples: int = 100000,
                 dataset_name: Optional[Union[str, Dict[str, float]]] = None,
                 dataset_config: Optional[List[str]] = None,
                 tokenizer: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 streaming: bool = False,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 seed: int = 42):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.streaming = streaming
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.seed = seed

        logger.info("Initializing DataModule with:")
        logger.info(f"  Sequence length: {sequence_length}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Num samples: {num_samples}")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Streaming: {streaming}")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        if stage == "fit" or stage is None:
            self.train_dataset = TokenDataset(
                sequence_length=self.sequence_length,
                vocab_size=self.vocab_size,
                batch_size=self.batch_size,
                num_samples=self.num_samples,
                dataset_name=self.dataset_name,
                dataset_config=self.dataset_config,
                tokenizer=self.tokenizer,
                cache_dir=self.cache_dir,
                streaming=self.streaming,
                split="train",
                seed=self.seed
            )

            # Create smaller validation dataset
            self.val_dataset = TokenDataset(
                sequence_length=self.sequence_length,
                vocab_size=self.vocab_size,
                batch_size=self.batch_size,
                num_samples=min(1000, self.num_samples // 10),  # 10% or 1000 samples max
                dataset_name=self.dataset_name,
                dataset_config=self.dataset_config,
                tokenizer=self.tokenizer,
                cache_dir=self.cache_dir,
                streaming=self.streaming,
                split="validation" if not self.streaming else "train",
                seed=self.seed + 1  # Different seed for validation
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """Collate function for batching"""
        # Stack tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    @classmethod
    def from_config(cls, data_config):
        """Create DataModule from configuration object"""
        return cls(
            sequence_length=data_config.sequence_length,
            vocab_size=data_config.vocab_size,
            batch_size=data_config.batch_size,
            num_samples=data_config.num_samples,
            dataset_name=data_config.dataset_name,
            dataset_config=data_config.dataset_config,
            tokenizer=data_config.tokenizer,
            cache_dir=data_config.cache_dir,
            streaming=data_config.streaming,
            seed=data_config.seed
        )


