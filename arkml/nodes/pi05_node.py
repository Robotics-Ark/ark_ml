from typing import Dict, Any
import torch
import numpy as np
from arkml.core.policy_node import PolicyNode
from arktypes import string_t


class Pi05Node(PolicyNode):
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
        policy_name = kwargs.get('policy_name', 'pi05_node')  # default policy name
        super().__init__(policy=model, policy_name=policy_name, device=device)

        self.model = model
        self.device = device

        # Move model to device
        self.model.to_device(device)

        # Set to eval mode
        self.model.set_eval_mode()

        # Register text input subscription
        self.create_subscription(string_t, "text_input", self.on_text_input, 10)

        # Internal state for sequence prediction if needed
        self.reset()

    def reset(self):
        """Reset internal state for the policy node."""
        self.model.reset()

    def predict(self, obs_seq: Dict[str, Any]) -> np.ndarray:
        """
        Compute the action for the given observation batch.

        The expected structure of ``obs_seq`` is dictated by the underlying VLA
        policy (typically a dict with batched tensors for images and state, and
        a list[str] for the task prompt).

        Args:
          obs_seq: Observation input to the policy (dict or tensor as required
            by the wrapped model).

        Returns:
          numpy.ndarray: Action vector for the first batch element.
        """
        obs = self.prepare_observation(obs_seq)

        with torch.no_grad():
            action = self.model.predict(obs)
            action = action.detach().cpu().numpy()

        return action

    def prepare_observation(self, ob: Dict[str, Any]):
        """
        Convert a single raw env observation into a batched policy input.
        This method should be implemented based on the expected observation format.

        Args:
          ob: Single observation dict from the environment.

        Returns:
          A batch dictionary compatible with the model.
        """
        # This needs to match the expected input format of the Pi05 model
        # Implementation depends on the specific observation format expected
        obs = {}

        # Handle state if available
        if 'state' in ob:
            state = torch.from_numpy(ob['state']).float().unsqueeze(0)  # (1, D)
            obs['state'] = state

        # Handle image if available
        if 'image' in ob:
            img = torch.from_numpy(ob['image']).float().unsqueeze(0)  # (1, C, H, W) or (1, H, W, C)
            obs['image'] = img

        # Handle task if available
        if 'task' in ob:
            obs['task'] = [ob['task']]  # List of strings expected

        return obs

    def on_text_input(self, msg):
        """Callback to receive text input from the text node."""
        if hasattr(self.model, "update_text_context"):
            self.model.update_text_context(msg.data)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass for training that calls the underlying model's forward method.

        Args:
            batch: Batch of observations for training

        Returns:
            Loss tensor for training
        """
        return self.model.forward(batch)