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
                 dataset_name: Optional[Dict[str, float]] = None,
                 dataset_config: Optional[List[str]] = None,
                 tokenizer=None,
                 cache_dir: Optional[str] = None,
                 streaming: bool = False,
                 split: str = "train",
                 seed: int = 42):

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
        self.token_buffer = []

        if tokenizer is None:
            tokenizer_name = "google/gemma-2-9b"
            logger.info(f"Initializing tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                token=os.getenv("HF_TOKEN"),
                cache_dir=os.getenv("HF_HOME"),
                trust_remote_code=True
            )
        else:
            self.tokenizer = tokenizer
            logger.info("Using provided tokenizer")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")

        self.datasets = self._load_datasets()

    def _load_datasets(self):
        """Load and combine multiple datasets according to proportions"""
        datasets = {}

        for dataset_name, proportion in self.dataset_name.items():
            logger.info(f"Loading dataset: {dataset_name} (proportion: {proportion})")

            # Handle datasets with configs
            for config in self.dataset_config:
                logger.info(f"Loading {dataset_name} with config: {config}")
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

        return datasets

    def _get_text_from_sample(self, sample):
        """Extract text from sample, handling different dataset formats"""
        if "text" in sample:
            return sample["text"]
        elif "content" in sample:
            return sample["content"]
        else:
            return " ".join([str(v) for v in sample.values() if isinstance(v, str)])

    def _tokenize_and_pack(self, dataset_iters, dataset_names, probabilities):
        """Pack multiple documents into a single sequence"""
        packed_tokens = []

        while len(packed_tokens) < self.sequence_length:
            try:
                selected_dataset = np.random.choice(dataset_names, p=probabilities)
                sample = next(dataset_iters[selected_dataset])

                text = self._get_text_from_sample(sample)
                if not text or len(text.strip()) < 10:
                    continue

                doc_tokens = self.tokenizer.encode(text, add_special_tokens=False)

                if not doc_tokens:
                    continue

                if packed_tokens:
                    packed_tokens.append(self.tokenizer.eos_token_id)

                remaining_space = self.sequence_length - len(packed_tokens)
                if remaining_space <= 0:
                    break

                tokens_to_add = min(len(doc_tokens), remaining_space)
                packed_tokens.extend(doc_tokens[:tokens_to_add])

                if tokens_to_add < len(doc_tokens):
                    break

            except StopIteration:
                dataset_iters[selected_dataset] = iter(self.datasets[selected_dataset]["dataset"])
                continue
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue

        if len(packed_tokens) < self.sequence_length:
            pad_length = self.sequence_length - len(packed_tokens)
            packed_tokens.extend([self.tokenizer.pad_token_id] * pad_length)
        elif len(packed_tokens) > self.sequence_length:
            packed_tokens = packed_tokens[:self.sequence_length]

        input_ids = torch.tensor(packed_tokens, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
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
                packed_sample = self._tokenize_and_pack(dataset_iters, dataset_names, probabilities)
                yield packed_sample
                samples_yielded += 1

            except Exception as e:
                logger.warning(f"Error creating packed sample: {e}")
                continue