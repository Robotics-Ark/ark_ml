from typing import Dict, Any
import torch
from arkml.core.policy import BasePolicy


class Pi05Node(BasePolicy):
    """
    Policy node for Pi0.5 integration.
    Structurally identical to PiZeroPolicyNode, using Pi05Policy internally.
    """

    def __init__(self, model, device="cpu", **kwargs):
        """
        Initialize the Pi0.5 policy node.

        Args:
            model: The Pi05Policy model instance
            device: Device to run the model on
        """
        super().__init__()  # Initialize parent class first
        self.model = model
        self.device = device

        # Move model to device
        self.model.to_device(device)

        # Set to eval mode
        self.model.set_eval_mode()

        # Internal state for sequence prediction if needed
        self.reset()

    def reset(self):
        """Reset internal state for the policy node."""
        self.model.reset()

    def predict(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Main prediction method that calls the underlying model's predict method.

        Args:
            obs: Observation dictionary containing image, state, task, etc.

        Returns:
            Predicted action tensor
        """
        return self.model.predict(obs)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass for training that calls the underlying model's forward method.

        Args:
            batch: Batch of observations for training

        Returns:
            Loss tensor for training
        """
        return self.model.forward(batch)

    def predict_n_actions(self, obs: Dict[str, Any], n_actions: int = 10) -> torch.Tensor:
        """
        Generate multiple action predictions.

        Args:
            obs: Observation dictionary
            n_actions: Number of actions to predict

        Returns:
            Tensor of multiple predicted actions
        """
        return self.model.predict_n_actions(obs, n_actions)

    def to_device(self, device: str):
        """
        Move the model to specified device.

        Args:
            device: Target device string (e.g., "cpu", "cuda")

        Returns:
            Self for method chaining
        """
        self.device = device
        self.model.to_device(device)
        return self