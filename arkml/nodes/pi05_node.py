from typing import Dict, Any
import torch
from arkml.core.policy import BasePolicy


class Pi05Node(BasePolicy):
    """
    Policy node for Pi0.5 integration.
    Implements the prediction pipeline: obs -> observation tokens -> subtask -> actions
    """

    def __init__(self, model, device="cpu", **kwargs):
        """
        Initialize the Pi0.5 policy node.

        Args:
            model: The Pi05Policy model instance
            device: Device to run the model on
        """
        self.model = model
        self.device = device

        # Move model to device
        self.model.to_device(device)

        # Internal state for sequence prediction
        self.reset()

    def reset(self):
        """Reset internal state for the policy node."""
        self._last_obs_tokens = None
        self._last_subtask_tokens = None
        self._action_buffer = []
        self._current_action_idx = 0

    def _obs_to_tokens(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Convert observation to observation tokens.
        TODO: Implement actual tokenization logic
        """
        # TODO: Implement actual observation tokenization
        # For now, return a placeholder tensor based on image input
        if "image" in obs:
            image_tensor = obs["image"]
            if not torch.is_tensor(image_tensor):
                image_tensor = torch.tensor(image_tensor)
            # Return shape that matches model expectations
            # Placeholder: flatten and return relevant features
            return image_tensor.flatten(start_dim=1).to(self.device)
        else:
            # If no image provided, return a zero tensor of expected size
            return torch.zeros(1, 512, device=self.device)  # Placeholder size

    def predict(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Main prediction pipeline:
        1. obs → observation tokens (TODO stub)
        2. subtask_tokens = model.sample_subtask(obs_tokens)
        3. actions = model.predict_with_flow(obs_tokens, subtask_tokens)
        4. return first action in chunk
        """
        # Set model to eval mode
        self.model.set_eval_mode()

        # Step 1: Convert observation to tokens
        # TODO: Implement actual tokenization logic for vision and language
        obs_tokens = self._obs_to_tokens(obs)

        # Step 2: Sample subtask using the model's subtask head
        with torch.no_grad():
            subtask_tokens = self.model.sample_subtask(obs_tokens)

        # Step 3: Predict actions using flow (note: in our current model implementation,
        # predict_with_flow doesn't take subtask_tokens as input, so we just use obs_tokens)
        # TODO: Update model to accept subtask_tokens if needed
        with torch.no_grad():
            actions = self.model.predict_with_flow(obs_tokens)

        # Step 4: Return first action in chunk (for now, return the single predicted action)
        if torch.is_tensor(actions):
            if actions.dim() == 1:
                # If single action, return as-is
                first_action = actions
            elif actions.dim() >= 2:
                # If batch of actions, take first in batch
                first_action = actions[0] if actions.size(0) > 0 else actions
            else:
                # Fallback
                first_action = actions
        else:
            # Fallback if not a tensor
            first_action = torch.tensor(actions, device=self.device)

        return first_action

    def predict_with_task(self, obs: Dict[str, Any], task_instruction: str = None) -> torch.Tensor:
        """
        Predict action with an optional task instruction.
        This could be used to condition the prediction on a specific task.
        """
        # Set model to eval mode
        self.model.set_eval_mode()

        # Convert observation to tokens
        # TODO: Implement actual tokenization logic for vision and language
        obs_tokens = self._obs_to_tokens(obs)

        # Sample subtask (could be influenced by task_instruction in more complex implementations)
        with torch.no_grad():
            subtask_tokens = self.model.sample_subtask(obs_tokens)

        # Predict actions using flow
        with torch.no_grad():
            actions = self.model.predict_with_flow(obs_tokens)

        # Return first action in chunk
        if torch.is_tensor(actions):
            if actions.dim() == 1:
                first_action = actions
            elif actions.dim() >= 2:
                first_action = actions[0] if actions.size(0) > 0 else actions
            else:
                first_action = actions
        else:
            first_action = torch.tensor(actions, device=self.device)

        return first_action