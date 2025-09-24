from abc import abstractmethod, ABC
from typing import Any

import numpy as np
from ark.client.comm_infrastructure.base_node import BaseNode
from ark.env.spaces import ObservationSpace, ActionSpace
from ark.tools.log import log
from arkml.utils.schema_io import (
    get_channel_types,
    _dynamic_observation_unpacker,
    _dynamic_action_packer,
)
from arkml.utils.schema_io import load_yaml
from arktypes import flag_t, string_t
from torch import nn


class PolicyNode(ABC, BaseNode):
    """Abstract base class for policy wrappers with async inference.

    Args:
      policy: Underlying policy module to be executed.
      device: Target device identifier (e.g., ``"cpu"``, ``"cuda"``).
      observation_unpacking: Function to unpack observations.
      action_packing: Function to pack actions.
      global_config: Global configuration path of the simulator or robot
    """

    def __init__(
        self,
        policy: nn.Module,
        policy_name: str,
        device: str,
        global_config=None,
    ):
        super().__init__(policy_name, global_config)

        cfg_dict = load_yaml(config_path=global_config)

        if "channel_config" not in cfg_dict:
            raise ValueError("channel_config must not be empty and properly configured")

        self.mode = cfg_dict["policy_mode"]
        if self.mode not in ["service", "stepper"]:
            raise ValueError("Policy_mode must be 'service' or 'stepper'")

        log.info(f"Policy mode: {self.mode}")

        schema = load_yaml(config_path=cfg_dict["channel_config"])
        # Channel config to get observations
        obs_channels = get_channel_types(schema=schema, channel_type="observation")
        self.observation_unpacking = _dynamic_observation_unpacker(schema)

        if not obs_channels:
            raise NotImplementedError("No observation channels found")

        self.observation_space = ObservationSpace(
            obs_channels, self.observation_unpacking, self._lcm
        )
        self._multi_comm_handlers.append(
            self.observation_space.observation_space_listener
        )

        action_channels = get_channel_types(schema=schema, channel_type="action")
        self.action_packing = _dynamic_action_packer(schema)
        self.action_space = ActionSpace(action_channels, self.action_packing, self._lcm)

        self._multi_comm_handlers.append(self.action_space.action_space_publisher)

        # Policy setup
        self.policy = policy
        self.policy.to_device(device=device)
        self.policy.set_eval_mode()
        self.latest_action = None
        self._stop_service = True
        self._resetting = False

        # Create services - predict and reset policy state
        self._predict_service_name = f"{self.name}/policy/predict"
        self._reset_service_name = f"{self.name}/policy/reset"

        self._start_service_name = f"{self.name}/policy/start"
        self._stop_service_name = f"{self.name}/policy/stop"

        # predict
        self.create_service(
            self._predict_service_name,
            flag_t,
            flag_t,
            self._callback_predict_service,
        )

        # reset
        self.create_service(
            self._reset_service_name, flag_t, flag_t, self._callback_reset_service
        )

        # start
        self.create_service(
            self._start_service_name, flag_t, flag_t, self._callback_start_service
        )
        # stop
        self.create_service(
            self._stop_service_name, flag_t, flag_t, self._callback_stop_service
        )


        if self.mode == "stepper":
            self.create_stepper(
                cfg_dict["simulator"]["node_frequency"], self.step_policy
            )

    def reset(self) -> None:
        """Reset the internal state of the policy."""
        # Block publishing immediately and clear any pending action
        self._stop_service = True
        self._resetting = True
        self.observation_space.is_ready = False
        self.suspend_communications(services=False)
        self.latest_action = None
        self.policy.reset()
        # Allow subclasses to clear their own buffers
        reset_hook = getattr(self, "_on_reset", None)
        if callable(reset_hook):
            reset_hook()
        self.resume_communications(services=False)
        self.observation_space.wait_until_observation_space_is_ready()
        _ = self.observation_space.get_observation()
        self._resetting = False

    def _callback_reset_service(self, channel, msg) -> string_t:
        """Service callback to reset policy state."""
        log.info(f"[INFO] Received callback reset service")
        reset_status = string_t()
        try:
            self.reset()
            reset_status.data = "Successfully reset policy state"
            log.info(f"[INFO] Finished callback reset service")
        except Exception as e:
            reset_status.data = f"Failed to reset policy state: {e}"
            log.error(f"[ERROR] Failed to reset policy state: {e}")
        return reset_status

    def _callback_predict_service(self, channel, msg):
        self.get_next_action()
        return flag_t()

    def step_policy(self):
        if self._stop_service:
            return
        self.get_next_action()

    def _callback_start_service(self, channel, msg) -> flag_t:
        """Start policy prediction service"""
        log.info(f"[INFO] Received callback to start service")
        self._stop_service = False
        return flag_t()

    def _callback_stop_service(self, channel, msg) -> flag_t:
        """Stop policy prediction service"""
        log.info(f"[INFO] Received callback to stop service")
        self._stop_service = True
        return flag_t()

    def get_next_action(self):
        """Compute the next action from observations."""
        self.observation_space.wait_until_observation_space_is_ready()
        obs = self.observation_space.get_observation()
        action = self.predict(obs)
        log.info(f"[ACTION PREDICTED] : {action}")
        self.action_space.pack_and_publish(action)

    @abstractmethod
    def predict(self, obs_seq: dict[str, Any]) -> np.ndarray:
        """Compute the action(s) from observations."""
        ...
