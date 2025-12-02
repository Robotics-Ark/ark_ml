from typing import Dict, Any
import torch
import torch.nn as nn
from arkml.core.policy import BasePolicy
from arkml.core.app_context import ArkMLContext
from arkml.algos.vla.pi05.models import Pi05Policy


class Pi05Node(BasePolicy):
    """
    Policy node for Pi0.5 integration following Pi0 patterns.
    """

    def __init__(self, model=None, device="cpu", **kwargs):
        """
        Initialize the Pi0.5 policy node.

        Args:
            model: The Pi05Policy model instance (optional - loads from config if None)
            device: Device to run the model on
        """
        super().__init__()  # Initialize parent class (nn.Module)

        if model is None:
            # Load model from configuration (for registry usage)
            model_cfg = ArkMLContext.cfg.get("algo").get("model")
            self.model = Pi05Policy(
                policy_type=model_cfg.get("policy_type", "pi0.5"),
                model_path=model_cfg.get("model_path", ""),
                obs_dim=model_cfg.get("obs_dim", 256),  # Default to original value
                action_dim=model_cfg.get("action_dim", 8),  # Default to original value
                image_dim=model_cfg.get("image_dim", (3, 224, 224)),
                pred_horizon=model_cfg.get("pred_horizon", 1),
                hidden_dim=model_cfg.get("hidden_dim", 512),  # Original parameter for custom architecture
                vocab_size=model_cfg.get("vocab_size", 32000),
                fast_vocab_size=model_cfg.get("fast_vocab_size", 1000)
            )
        else:
            # Use provided model (for backward compatibility)
            self.model = model

        self.device = device

        # Move model to device
        self.model.to_device(device)

        # Internal state for sequence prediction
        self.reset()

    def reset(self):
        """Reset internal state for the policy node."""
        pass

    def predict(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Run model prediction on the observation.
        """
        # Set model to eval mode
        self.model.set_eval_mode()

        # Run prediction through the model
        with torch.no_grad():
            actions = self.model.predict(obs)

        # Ensure proper tensor format for output
        if torch.is_tensor(actions):
            if actions.dim() == 1:
                # If single action, return as-is
                return actions
            elif actions.dim() >= 2:
                # If batch of actions, take first in batch
                return actions[0] if actions.size(0) > 0 else actions
            else:
                # Fallback
                return actions
        else:
            # Fallback if not a tensor
            return torch.tensor(actions, device=self.device)

    def to_device(self, device: str):
        """
        Move the node and its components to the specified device.
        """
        self.device = device
        self.model.to_device(device)
        return self

    def set_eval_mode(self):
        """
        Set the model to evaluation mode.
        """
        self.model.set_eval_mode()

    def set_train_mode(self):
        """
        Set the model to training mode.
        """
        self.model.set_train_mode()