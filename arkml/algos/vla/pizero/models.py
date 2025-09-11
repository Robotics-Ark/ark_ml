import torch.nn as nn
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from arkml.core.registry import MODELS


@MODELS.register("PiZeroNet")
class PiZeroNet(nn.Module):
    """
    VLA policy wrapper that uses explicit lerobot policies with a switchable type.

    - policy_type: 'pi0' or 'smolvla'
    - pretrained_model: HF hub id or local path. If None, uses a sensible default per type.
    - Numeric state only is supported out-of-the-box (passed as 'observation.state').
      To use image-based policies like SmolVLA, pass a full observation dict with
      the required image tensors and task string.
    """

    def __init__(
            self,
            policy_type: str,
            model_path: str,
            obs_dim: int,
            action_dim: int,
            image_dim: tuple,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_dim = image_dim
        self.device = None

        kind = policy_type.lower()
        if kind not in {"pi0", "smolvla"}:
            raise ValueError(f"Unsupported policy_type '{policy_type}'. Use 'pi0' or 'smolvla'.")

        policy_class = PI0Policy if kind == "pi0" else SmolVLAPolicy

        try:
            self._policy = policy_class.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained {kind} policy '{model_path}': {e}")

        self._policy.config.input_features = {
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=self.image_dim
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.obs_dim,)
            )
        }
        self._policy.config.output_features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,)
            )
        }

    def load_mode_to_device(self, device: str):
        self.device = device
        self._policy.to(device)

    def set_eval_mode(self):
        self._policy.eval()

    def reset(self):
        self._policy.reset()

    def prepare_input(self,observation: dict[str, ...]):
        if not isinstance(observation, dict):
            raise TypeError("Observation expected in dict type")

        obs = {
            "observation.images.image": observation["image"].to(self.device),
            "observation.state": observation["state"].to(self.device),
            "task": observation["task"]
        }

        return obs

    def forward(self, observation):
        """
        Accepts either:
          - numeric vector or (H,D) array/tensor: wrapped into {'observation.state': tensor[1,D]}
          - full observation dict expected by the policy: passed through unchanged
        Returns: action tensor (action_dim,)
        """
        obs = self.prepare_input(observation=observation)
        action = self._policy.select_action(obs)
        return action
