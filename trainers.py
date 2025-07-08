from transformers import TrainerCallback
from loguru import logger

class GradualUnfreezingCallback(TrainerCallback):
    def __init__(self, model, unfreezing_schedule):
        self.model = model
        self.unfreezing_schedule = unfreezing_schedule
        self._initial_freeze()


    def _initial_freeze(self):
        """Freeze all parameters except PEER layers - using parameter IDs"""

        peer_param_ids = set()
        if hasattr(self.model, 'peer_layers'):
            for idx, peer_layer in self.model.peer_layers.items():
                for param in peer_layer.parameters():
                    peer_param_ids.add(id(param))

        # Freeze/unfreeze based on parameter identity
        for name, param in self.model.named_parameters():
            if id(param) in peer_param_ids:
                param.requires_grad = True
                logger.info(f" PEER param kept trainable: {name}")
            else:
                param.requires_grad = False
                logger.info(f"❄️ Frozen: {name}")

        logger.info(f"Total PEER parameters: {len(peer_param_ids)}")

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step

        if current_step in self.unfreezing_schedule:
            layers_to_unfreeze = self.unfreezing_schedule[current_step]
            self._unfreeze_layers(layers_to_unfreeze)

    def _unfreeze_layers(self, layer_patterns):
        for name, param in self.model.named_parameters():
            for pattern in layer_patterns:
                if pattern in name:
                    param.requires_grad = True
                    logger.info(f"Unfrozen layer: {name}")
