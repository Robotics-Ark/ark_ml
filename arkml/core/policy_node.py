import importlib
import queue
import threading
import time
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ark.client.comm_infrastructure.base_node import BaseNode
from ark.env.spaces import ActionSpace, ObservationSpace
from ark.tools.log import log
from arktypes import flag_t
from arktypes.utils import pack, unpack
from torch import nn


class PolicyNode(ABC, BaseNode):
    """Abstract base class for policy wrappers with async inference.

    Args:
      policy: Underlying policy module to be executed.
      device: Target device identifier (e.g., ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        policy: nn.Module,
        device: str,
        stepper_frequency: int,
        global_config=None,
        channel_config_path: str | Path | None = None,
    ):
        super().__init__("Policy", global_config)

        # Channel config to publish and subscribe
        channel_cfg_path = Path(channel_config_path)
        if channel_cfg_path.exists():
            with open(channel_cfg_path, "r") as f:
                cfg_dict = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(
                f"Config file could not found {channel_config_path}"
            )
        observation_channels = cfg_dict.get("observation_channels", {})
        action_channels = cfg_dict.get("action_channels", {})
        self.obs_keys = list(observation_channels.keys())
        self.action_keys = list(action_channels.keys())

        obs_channels = self._resolve_channel_types(observation_channels)

        act_channels = self._resolve_channel_types(action_channels)

        if not obs_channels:
            raise NotImplementedError("No observation channels found")

        if not act_channels:
            raise NotImplementedError("No action channels found")

        self.action_space = ActionSpace(act_channels, self.action_packing, self._lcm)
        self.observation_space = ObservationSpace(
            obs_channels, self.observation_unpacking, self._lcm
        )

        self._multi_comm_handlers.append(self.action_space.action_space_publisher)
        self._multi_comm_handlers.append(
            self.observation_space.observation_space_listener
        )

        # Policy setup
        self.policy = policy
        self.policy.to_device(device=device)
        self.policy.set_eval_mode()
        self.latest_action = None
        self._stop_event = threading.Event()

        self.worker_thread = threading.Thread(
            target=self._inference_worker, daemon=True
        )
        self.worker_thread.start()

        # Stepper publishes actions at fixed control frequency
        self.create_stepper(stepper_frequency, self.step)

        # Reset service to reset policy state
        self._reset_service_name = f"{self.name}/policy/reset"
        self.create_service(
            self._reset_service_name, flag_t, flag_t, self._callback_reset_service
        )

    def _inference_worker(self):
        """Background thread to run model inference asynchronously."""
        while not self._stop_event.is_set():
            self.observation_space.wait_until_observation_space_is_ready()
            obs = self.observation_space.get_observation()
            if obs is not None:
                action = self.predict(obs)
                log.info(f"[ACTION PREDICTED] : {action}")
                self.latest_action = action

    def reset(self) -> None:
        """Reset the internal state of the policy."""
        try:
            self.observation_space.is_ready = False
            self.suspend_communications(services=False)
            self.policy.reset()
            # Allow subclasses to clear their own buffers
            reset_hook = getattr(self, "_on_reset", None)
            if callable(reset_hook):
                reset_hook()
            self.latest_action = None
            time.sleep(5)
            self.observation_space.wait_until_observation_space_is_ready()
            _ = self.observation_space.get_observation()
        finally:
            self.resume_communications(services=False)

    def _callback_reset_service(self, channel, msg):
        """Service callback to reset policy state.

        Returns an empty flag_t response after clearing internal buffers.
        """
        # Pause non-service comms to avoid races during reset
        log.info(f"[INFO] Received callback reset service")
        self.reset()
        return flag_t()

    def step(self):
        """Stepper loop: publish latest action if available."""
        if self.latest_action is None:
            return
        self.publish_action(self.latest_action)
        self.latest_action = None

    def publish_action(self, action: np.ndarray):
        """Pack and publish action to downstream consumers."""
        if action is None or action.shape[0] < 8:
            return

        self.action_space.pack_and_publish(action)

    @abstractmethod
    def predict(self, obs_seq: dict[str, Any]) -> np.ndarray:
        """Compute the action(s) from observations."""
        ...

    def _resolve_channel_types(self, mapping: dict[str, Any]) -> dict[str, type]:
        """Resolve type names from config into arktypes classes.

        Accepts either already-imported classes or string names present in the
        ``arktypes`` package. Returns a mapping of channel name to type.
        """
        if not mapping:
            return {}
        resolved: dict[str, type] = {}
        arktypes_mod = importlib.import_module("arktypes")
        for ch_name, t in mapping.items():
            if isinstance(t, str):
                resolved[ch_name] = getattr(arktypes_mod, t)
            else:
                resolved[ch_name] = t
        return resolved

    def observation_unpacking(self, observation_dict):
        """Unpack raw Ark observations into structured components.

        Converts incoming channel messages into a dictionary with primitive
        types useful for policies.

        Returns a dictionary with keys:
          - ``cube``: np.ndarray (3,) cube position
          - ``target``: np.ndarray (3,) target position
          - ``gripper``: list[float] gripper opening
          - ``franka_ee``: tuple(np.ndarray (3,), np.ndarray (4,)) EE position and quaternion
          - ``images``: tuple(rgb, depth) from the RGBD sensor

        Args:
          observation_dict: Mapping from channel name to serialized Ark message.

        Returns:
          dict: Structured observation dictionary as described above.
        """

        if not observation_dict or any(v is None for v in observation_dict.values()):
            return None
        cube_state = observation_dict[self.obs_keys[0]]
        target_state = observation_dict[self.obs_keys[1]]
        joint_state = observation_dict[self.obs_keys[2]]
        ee_state = observation_dict[self.obs_keys[3]]
        images = observation_dict[self.obs_keys[4]]
        _, cube_position, _, _, _ = unpack.rigid_body_state(cube_state)
        _, target_position, _, _, _ = unpack.rigid_body_state(target_state)
        _, _, franka_joint_position, _, _ = unpack.joint_state(joint_state)
        franka_ee_position, franka_ee_orientation = unpack.pose(ee_state)
        rgb, depth = unpack.rgbd(images)

        gripper_position = franka_joint_position[
            -2
        ]  # Assuming last two joints are gripper

        return {
            "cube": cube_position,
            "target": target_position,
            "gripper": [gripper_position],
            "franka_ee": (franka_ee_position, franka_ee_orientation),
            "images": (rgb, depth),
        }

    def action_packing(self, action: list) -> dict[str, Any]:
        """Pack action into Ark cartesian command message.

        Expects an 8D vector representing end-effector position, orientation
        quaternion, and gripper command in the following order:
        ``[x, y, z, qx, qy, qz, qw, gripper]``.

        Args:
          action: List describing the cartesian command.

        Returns:
          dict[str, ...]: Mapping with key pointing to a packed Ark message.
        """

        xyz_command = np.array(action[:3])
        quaternion_command = np.array(action[3:7])
        gripper_command = action[7]

        franka_cartesian_command = pack.task_space_command(
            "all", xyz_command, quaternion_command, gripper_command
        )
        return {self.action_keys[0]: franka_cartesian_command}
