from typing import Dict, Any
import torch
import torch.nn as nn
from arkml.core.policy import BasePolicy
from arkml.algos.vla.pi05.pi05_processor import Pi05Processor
from arkml.core.app_context import ArkMLContext
from arkml.algos.vla.pi05.models import Pi05Policy


class Pi05Node(BasePolicy):
    """
    Policy node for Pi0.5 integration.
    Implements the prediction pipeline: obs -> observation tokens -> subtask -> actions
    """

    def __init__(self, model=None, device="cpu", processor_path=None, **kwargs):
        """
        Initialize the Pi0.5 policy node.

        Args:
            model: The Pi05Policy model instance (optional - loads from config if None)
            device: Device to run the model on
            processor_path: Path to processor configuration (optional)
        """
        super().__init__()  # Initialize parent class (nn.Module)

        if model is None:
            # Load model from configuration (for registry usage)
            model_cfg = ArkMLContext.cfg.get("algo").get("model")
            self.model = Pi05Policy(
                policy_type=model_cfg.get("policy_type", "pi0.5"),
                model_path=model_cfg.get("model_path", ""),
                obs_dim=model_cfg.get("obs_dim", 256),
                action_dim=model_cfg.get("action_dim", 8),
                image_dim=model_cfg.get("image_dim", (3, 224, 224)),
                pred_horizon=model_cfg.get("pred_horizon", 1),
                hidden_dim=model_cfg.get("hidden_dim", 512),
                vocab_size=model_cfg.get("vocab_size", 32000),
                fast_vocab_size=model_cfg.get("fast_vocab_size", 1000)
            )
        else:
            # Use provided model (for backward compatibility)
            self.model = model

        self.device = device

        # Initialize processor
        self.processor = Pi05Processor(device=device)

        # Move model to device
        self.model.to_device(device)

        # Internal state for sequence prediction
        self.reset()

    def reset(self):
        """Reset internal state for the policy node."""
        # Currently no internal state to reset for this simple implementation
        pass

    def run_once(self, obs):
        """
        Run a single inference step from raw observation to action.

        Args:
            obs: Raw observation dict containing 'image' and optionally 'language'/'instruction'

        Returns:
            Action vector as tensor
        """
        # Preprocess observation using the processor
        preprocessed = self.processor(obs)

        # Run model prediction in eval mode with no gradients
        with torch.no_grad():
            self.model.set_eval_mode()
            action = self.model.predict(preprocessed)
        return action

    def predict(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Main prediction pipeline:
        1. Preprocess observation using the processor
        2. Run model prediction on preprocessed observation
        3. Return action tensor
        """
        # Set model to eval mode
        self.model.set_eval_mode()

        # Use processor to preprocess observation
        preprocessed_obs = self.processor(obs)

        # Run prediction through the model
        with torch.no_grad():
            actions = self.model.predict(preprocessed_obs)

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

    def predict_with_task(self, obs: Dict[str, Any], task_instruction: str = None) -> torch.Tensor:
        """
        Predict action with an optional task instruction.
        This could be used to condition the prediction on a specific task.
        """
        # If task instruction is provided separately, update observation
        if task_instruction and isinstance(task_instruction, str):
            obs = obs.copy()
            obs['instruction'] = task_instruction

        # Set model to eval mode
        self.model.set_eval_mode()

        # Use processor to preprocess observation
        preprocessed_obs = self.processor(obs)

        # Run prediction through the model
        with torch.no_grad():
            actions = self.model.predict(preprocessed_obs)

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

    def to_device(self, device: str):
        """
        Move the node and its components to the specified device.
        """
        self.device = device
        self.model.to_device(device)
        # Update processor device
        self.processor.device = device
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