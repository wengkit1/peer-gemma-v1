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

        # Load datasets
        self.datasets = self._load_datasets()

    def _load_datasets(self):
        """Load and combine multiple datasets according to proportions"""
        datasets = {}

        for dataset_name, proportion in self.dataset_name.items():
            logger.info(f"Loading dataset: {dataset_name} (proportion: {proportion})")

            if dataset_name == "allenai/c4":
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
                selected_dataset = np.random.choice(dataset_names, p=probabilities)

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
                dataset_iters[selected_dataset] = iter(self.datasets[selected_dataset]["dataset"])
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue



