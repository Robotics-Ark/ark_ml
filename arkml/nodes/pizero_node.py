from collections import deque

import numpy as np
import torch
from arkml.algos.vla.pizero.models import PiZeroNet
from arkml.core.policy_node import PolicyNode
from arktypes import task_space_command_t, string_t
from arktypes.utils import pack


class PiZeroPolicyNode(PolicyNode):
    """Wrapper node for PiZero/SmolVLA.

    Args:
      model_cfg: Model configurations.
      device: Target device string (e.g., ``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, model_cfg, device="cuda", global_config=None):
        policy = PiZeroNet(
            policy_type=model_cfg.policy_type,
            model_path=model_cfg.model_path,
            obs_dim=model_cfg.obs_dim,
            action_dim=model_cfg.action_dim,
            image_dim=model_cfg.image_dim,
        )
        super().__init__(
            policy=policy,
            device=device,
            channel_type=string_t,
            message_type=task_space_command_t,
            global_config=global_config,
        )

        self.policy.to_device(device)
        self.policy.reset()
        self.policy.set_eval_mode()
        self.create_stepper(10, self.step)

        # Inference chunking: number of actions to prefetch from the model when queue is empty
        self.n_infer_actions = getattr(model_cfg, "pred_horizon", 10)
        self._action_queue: deque[np.ndarray] = deque()

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

        # Convert to tensors in the expected shapes
        obs = {}
        if "image" in obs_seq and obs_seq["image"] is not None:
            obs["image"] = torch.tensor(obs_seq["image"], dtype=torch.float32)
        if "state" in obs_seq and obs_seq["state"] is not None:
            obs["state"] = torch.tensor(obs_seq["state"], dtype=torch.float32)
        if "task" in obs_seq:
            obs["task"] = obs_seq["task"]

        # Serve one action per call. If queue is empty, prefetch n actions.
        if len(self._action_queue) == 0:
            with torch.no_grad():
                actions = self.policy.predict_n_actions(
                    obs, n_actions=self.n_infer_actions
                )
            actions_np = actions.detach().cpu().numpy()  # (n, action_dim)
            for i in range(actions_np.shape[0]):
                self._action_queue.append(actions_np[i])

        return self._action_queue.popleft()

    def publish_action(self, action: np.ndarray):
        """Pack and publish action to downstream consumers."""
        if action is None or action.shape[0] < 8:
            return

        xyz = np.asarray(action[:3], dtype=np.float32)
        quat = np.asarray(action[3:7], dtype=np.float32)
        grip = float(action[7])
        msg = pack.task_space_command("next_action", xyz, quat, grip)
        self.pub.publish(msg)
