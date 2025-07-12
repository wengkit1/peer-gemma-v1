from typing import Dict, List, Optional
import os
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, get_dataset_config_info
from transformers import AutoTokenizer, TrainerCallback
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
                 seed: int = 42,
                 validation_split_ratio: float = 0.20,
                 validation_offset: int = 10_000_000):

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.validation_split_ratio = validation_split_ratio
        self.validation_offset = validation_offset

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


    def __len__(self):
        """Return the number of packed samples"""
        return self.num_samples


    def _load_datasets(self):
        """Load and combine multiple datasets according to proportions"""
        datasets = {}

        for dataset_name, proportion in self.dataset_name.items():
            logger.info(f"Loading dataset: {dataset_name} (proportion: {proportion})")

            # Handle datasets with configs
            for config in self.dataset_config:
                logger.info(f"Loading {dataset_name} with config: {config}")
                try:
                    # First, try to load the specified split
                    dataset = load_dataset(
                        dataset_name,
                        config,
                        split=self.split,
                        streaming=self.streaming,
                        cache_dir=self.datasets_cache_dir,
                        token=os.getenv("HF_TOKEN")
                    )

                    # If split is "validation" but dataset doesn't have validation split,
                    # create one from the training data
                    if self.split == "validation":
                        # Check if validation split exists by inspecting dataset info
                        validation_exists = self._check_validation_split_exists(dataset_name, config)

                        if validation_exists:
                            logger.info(f"Found existing validation split for {dataset_name}_{config}")
                        else:
                            # Validation split doesn't exist, create from train
                            logger.warning(
                                f"No validation split found for {dataset_name}_{config}, creating from train split")
                            dataset = self._create_validation_from_train(dataset_name, config)

                    datasets[f"{dataset_name}_{config}"] = {
                        "dataset": dataset,
                        "proportion": proportion
                    }
                    logger.success(f"Successfully loaded {dataset_name}_{config}")
                except Exception as e:
                    logger.error(f"Failed to load {dataset_name}_{config}: {e}")

        return datasets


    @staticmethod
    def _check_validation_split_exists(dataset_name: str, config: str) -> bool:
        """Check if validation split exists without downloading data"""
        try:

            # Get dataset info to check available splits
            dataset_info = get_dataset_config_info(
                dataset_name,
                config_name=config,
                token=os.getenv("HF_TOKEN")
            )

            # Check if 'validation' is in available splits
            available_splits = list(dataset_info.splits.keys())
            logger.info(f"Available splits for {dataset_name}/{config}: {available_splits}")

            return "validation" in available_splits

        except Exception as e:
            logger.warning(f"Could not check splits for {dataset_name}/{config}: {e}")
            return False

    def _create_validation_from_train(self, dataset_name: str, config: str):
        """Create validation split from training data using offset approach"""
        logger.info(f"Creating validation split from train data for {dataset_name}_{config}")

        # Load training dataset as streaming
        train_dataset = load_dataset(
            dataset_name,
            config,
            split="train",
            streaming=True,
            cache_dir=self.datasets_cache_dir,
            token=os.getenv("HF_TOKEN")
        )

        validation_dataset = train_dataset.skip(self.validation_offset)

        # Take only the number of samples you need for validation
        val_samples_target = int(self.num_samples * self.validation_split_ratio)
        validation_dataset = validation_dataset.take(val_samples_target)

        return validation_dataset


    @staticmethod
    def _get_text_from_sample(sample):
        """Extract text from sample, handling different data formats"""
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


class DynamicEvalDataset(IterableDataset):
    """A wrapper that provides different subsets of eval data for each evaluation"""

    def __init__(self, base_dataset, eval_samples_per_call: int = 1000, seed: int = 42):
        self.base_dataset = base_dataset
        self.eval_samples_per_call = eval_samples_per_call
        self.seed = seed
        self.call_count = 0


    def __len__(self):
        """Return the number of samples per evaluation call"""
        return self.eval_samples_per_call


    def __iter__(self):
        """Each call returns a different subset of the evaluation data"""
        current_seed = self.seed + self.call_count
        self.call_count += 1

        random.seed(current_seed)
        np.random.seed(current_seed)

        logger.info(f"Generating eval subset {self.call_count} with seed {current_seed}")

        base_iter = iter(self.base_dataset)

        for i, sample in enumerate(base_iter):
            if i >= self.eval_samples_per_call:
                break
            yield sample


class DynamicEvalCallback(TrainerCallback):
    """Callback that refreshes the eval dataset for each evaluation"""

    def __init__(self, dynamic_eval_dataset):
        self.dynamic_eval_dataset = dynamic_eval_dataset

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Called before each evaluation - refresh the eval dataset"""
        if hasattr(state, 'trainer') and hasattr(state.trainer, 'eval_dataset'):
            state.trainer.eval_dataset = self.dynamic_eval_dataset
            logger.info(f"Refreshed eval dataset for evaluation step {state.global_step}")

