import torch
from transformers import GemmaForCausalLM, AutoModelForCausalLM
from PEER_pytorch import PEER
from loguru import logger
from typing import List, Dict, Any
import os


class PEERGemmaForCausalLM(GemmaForCausalLM):
    """
    Gemma model with PEER layers following the paper: Mixture of A Million Experts

    Each replaced layer gets its own independent PEER layer with separate expert pools
    """

    def __init__(self, config, replace_layers: List[int], peer_config=None):
        super().__init__(config)
        self.replace_layers = replace_layers
        self.peer_config = peer_config or {}
        self.replace_layers = replace_layers

        self.peer_layers = {}

        self._replace_mlp_layers()
        logger.info(
            f" Replaced {len(self.replaced_layer_indices)} MLP layers with PEER layers")

    def _create_peer_layer(self, layer_idx: int):
        """Create an independent PEER layer for a specific transformer layer"""
        hidden_size = self.config.hidden_size # From model loaded config

        # Default PEER configuration based on paper
        default_peer_config = {
            "dim": hidden_size,
            "heads": min(16, hidden_size // 128),
            "num_experts": 65_536, # 256^2
            "num_experts_per_head": 16,
            "dim_key": 128,
            "pre_rmsnorm": True
        }

        final_peer_config = {**default_peer_config, **self.peer_config}

        logger.info(f" Creating PEER layer for transformer layer {layer_idx}:")
        logger.info(f"   Experts: {final_peer_config['num_experts']:,}")
        logger.info(f"   Heads: {final_peer_config['heads']}")
        logger.info(f"   Dim: {final_peer_config['dim']}")

        peer_layer = PEER(**final_peer_config)

        # Count parameters in this layer
        peer_params = sum(p.numel() for p in peer_layer.parameters())
        memory_gb = peer_params * 4 / (1024 ** 3)  # 4 bytes per float32

        logger.info(f"✨ PEER layer {layer_idx} created: {peer_params:,} parameters ({memory_gb:.2f} GB)")

        return peer_layer

    def _replace_mlp_layers(self):
        """Replace MLP layers with independent PEER layers (following the paper)"""
        num_layers = len(self.model.layers)

        logger.info(f"  Model surgery info:")
        logger.info(f"    Total layers: {num_layers}")
        logger.info(f"    Hidden size: {self.config.hidden_size}")
        logger.info(f"    Intermediate size: {self.config.intermediate_size}")
        logger.info(f"   Replacing layers: {self.replace_layers}")

        # Track parameter changes
        total_original_params = 0
        total_peer_params = 0

        for i in self.replace_layers:
            original_mlp = self.model.layers[i].mlp

            original_params = sum(p.numel() for p in original_mlp.parameters())
            total_original_params += original_params

            # Create independent PEER layer for this transformer layer
            peer_layer = self._create_peer_layer(i)
            peer_params = sum(p.numel() for p in peer_layer.parameters())
            total_peer_params += peer_params

            # Store the PEER layer
            self.peer_layers[i] = peer_layer

            # Replace the MLP with the PEER layer
            self.model.layers[i].mlp = peer_layer

            logger.info(f"   Layer {i}: MLP({original_params:,}) → PEER({peer_params:,})")

        logger.info(f"")
        logger.info(f"   Surgery Summary:")
        logger.info(f"   Original total: {total_original_params:,} parameters")
        logger.info(f"   PEER total: {total_peer_params:,} parameters")
        logger.info(f"   Parameter ratio: {total_peer_params / total_original_params:.2f}x")

    @classmethod
    def from_pretrained_with_peer_surgery(cls,
                                          model_path: str,
                                          replace_layers="middle",
                                          peer_config=None,
                                          **kwargs):
        """
        Load and modify model with independent PEER layers (following the paper)
        """
        logger.info(f" Loading model from: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=os.getenv("HF_TOKEN"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        # Convert to PEER model class
        model.__class__ = cls

        model.replace_layers = replace_layers
        model.peer_config = peer_config or {}
        model.replaced_layer_indices = []
        model.peer_layers = {}

        # Perform surgery on the existing model
        assert isinstance(model, PEERGemmaForCausalLM)
        model._replace_mlp_layers()

        logger.success(f" ✅PEER surgery completed successfully!")
        return model

    def get_peer_info(self) -> Dict[str, Any]:
        """Get information about the PEER surgery and memory usage"""
        if not self.peer_layers:
            return {"error": "No PEER surgery performed"}

        # Calculate total PEER parameters across all layers
        total_peer_params = 0
        layer_info = {}

        for layer_idx, peer_layer in self.peer_layers.items():
            peer_params = sum(p.numel() for p in peer_layer.parameters())
            total_peer_params += peer_params
            layer_info[f"layer_{layer_idx}"] = {
                "parameters": peer_params,
                "num_experts": peer_layer.num_experts,
                "heads": peer_layer.heads,
                "experts_per_head": peer_layer.num_experts_per_head
            }

        total_model_params = sum(p.numel() for p in self.parameters())

        return {
            "replaced_layers": len(self.replaced_layer_indices),
            "replaced_layer_indices": self.replaced_layer_indices,
            "total_peer_parameters": total_peer_params,
            "total_model_parameters": total_model_params,
            "peer_parameter_ratio": total_peer_params / total_model_params,
            "memory_estimate_gb": total_model_params * 4 / (1024 ** 3),
            "layer_details": layer_info,
            "experts_per_layer": self.peer_config.get('num_experts', 1_000_000),
            "total_experts_across_layers": len(self.peer_layers) * self.peer_config.get('num_experts', 1_000_000)
        }

    def get_peer_layer(self, layer_idx: int):
        """Access a specific PEER layer"""
        if layer_idx not in self.peer_layers:
            raise ValueError(f"Layer {layer_idx} was not replaced with PEER")
        return self.peer_layers[layer_idx]

    def get_all_peer_layers(self):
        """Access all PEER layers"""
        return self.peer_layers
