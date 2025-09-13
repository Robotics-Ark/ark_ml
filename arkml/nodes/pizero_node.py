import json
import torch
import numpy as np
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
        self.task_prompt = None

        self.policy = PiZeroNet(
            policy_type=model_cfg.policy_type,
            model_path=model_cfg.model_path,
            obs_dim=model_cfg.obs_dim,
            action_dim=model_cfg.action_dim,
            image_dim=model_cfg.image_dim,
        )
        super().__init__(
            policy=self.policy,
            device=device,
            channel_type=string_t,  # generic observation as JSON string
            message_type=task_space_command_t,  # publish robot-agnostic task-space command
            global_config=global_config,
        )

        self.policy.to_device(device)
        self.policy.reset()
        self.policy.set_eval_mode()
        self.create_stepper(10, self.step)

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
        # breakpoint()
        # If observation comes as JSON string_t, convert to model input dict
        if isinstance(obs_seq, str):
            payload = json.loads(obs_seq)
        elif hasattr(obs_seq, "data") and isinstance(obs_seq.data, str):
            payload = json.loads(obs_seq.data)
        elif isinstance(obs_seq, dict):
            payload = obs_seq
        else:
            raise ValueError("Unsupported observation format for PiZeroPolicyNode")

        # Convert to tensors in the expected shapes
        obs = {}
        if "image" in payload and payload["image"] is not None:
            img = torch.tensor(payload["image"], dtype=torch.float32)
            obs["image"] = img
        if "state" in payload and payload["state"] is not None:
            st = torch.tensor(payload["state"], dtype=torch.float32)
            obs["state"] = st
        if "task" in payload:
            obs["task"] = payload["task"]

        with torch.no_grad():
            action = self.policy.predict(obs)
        return action.detach().cpu().numpy()[0]

    def step(self):
        if self.curr_observation is None:
            return

        # Run policy inference given the latest observation payload
        action = self.predict(obs_seq=self.curr_observation)
        print(f"[ACTION PREDICTION] {action}")
        # breakpoint()

        # Expect an 8D action: [x, y, z, qx, qy, qz, qw, gripper]
        if action.shape[0] < 8:
            return

        xyz = np.asarray(action[:3], dtype=np.float32)
        quat = np.asarray(action[3:7], dtype=np.float32)
        grip = float(action[7])

        msg = pack.task_space_command("franka", xyz, quat, grip)
        self.pub.publish(msg)
