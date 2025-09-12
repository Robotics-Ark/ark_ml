import torch

from arkml.algos.vla.pizero.models import PiZeroNet
from arkml.nodes.policy_node import PolicyNode

from ark_framework.ark.client.comm_infrastructure.base_node import BaseNode


class PiZeroPolicyNode(BaseNode):
    """Wrapper node for PiZero/SmolVLA.

    Args:
      model_cfg: Model configurations.
      device: Target device string (e.g., ``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, model_cfg, device="cuda"):
        self.task_prompt = None

        policy = PiZeroNet(
            policy_type=model_cfg.policy_type,
            model_path=model_cfg.model_path,
            obs_dim=model_cfg.obs_dim,
            action_dim=model_cfg.action_dim,
            image_dim=model_cfg.image_dim,
        )
        super().__init__(policy=policy, device=device)

    def predict(self, obs_seq):
        """Compute the action for the given observation batch.

        The expected structure of ``obs_seq`` is dictated by the underlying VLA
        policy (typically a dict with batched tensors for images and state, and
        a list[str] for the task prompt).

        Args:
          obs_seq: Observation input to the policy (dict or tensor as required
            by the wrapped model).

        Returns:
          numpy.ndarray: Action vector for the first batch element.
        """

        with torch.no_grad():
            action = self.policy.predict(obs_seq)
        return action.detach().cpu().numpy()[0]
