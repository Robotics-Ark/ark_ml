import importlib
import threading
import time
from abc import abstractmethod, ABC
from functools import partial
from pathlib import Path
from typing import Any
from collections.abc import Callable
import numpy as np
import yaml
from ark.client.comm_infrastructure.base_node import BaseNode
from ark.env.spaces import ActionSpace, ObservationSpace
from ark.tools.log import log
from arktypes import flag_t
from torch import nn


class PolicyNode(ABC, BaseNode):
    """Abstract base class for policy wrappers with async inference.

    Args:
      policy: Underlying policy module to be executed.
      device: Target device identifier (e.g., ``"cpu"``, ``"cuda"``).
      observation_unpacking: Function to unpack observations.
      action_packing: Function to pack actions.
      stepper_frequency: Frequency of stepper.
      global_config: Global configuration path of the simulator or robot
    """

    def __init__(
        self,
        policy: nn.Module,
        device: str,
        observation_unpacking: Callable,
        action_packing: Callable,
        stepper_frequency: int,
        global_config=None,
    ):
        super().__init__("Policy", global_config)

        # Channel config to publish and subscribe
        channel_cfg_path = Path(global_config)
        if channel_cfg_path.exists():
            with open(channel_cfg_path, "r") as f:
                cfg_dict = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file could not found {global_config}")

        if "channel_config" not in cfg_dict:
            raise ValueError("channel_config must not be empty and properly configured")

        channel_config = cfg_dict["channel_config"]
        observation_channels = channel_config.get("observation_channels", {})
        action_channels = channel_config.get("action_channels", {})

        obs_keys = list(observation_channels.keys())
        action_keys = list(action_channels.keys())

        self.observation_unpacking = partial(observation_unpacking, obs_keys=obs_keys)
        self.action_packing = partial(action_packing, action_keys=action_keys)

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
        self._resetting = False
        self._stop_event = threading.Event()
        self._publish_lock = threading.Lock()

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
            if self._resetting:
                self.latest_action = None
                time.sleep(0.05)
                continue
            self.observation_space.wait_until_observation_space_is_ready()
            obs = self.observation_space.get_observation()
            if obs is not None:
                action = self.predict(obs)
                log.info(f"[ACTION PREDICTED] : {action}")
                self.latest_action = action

    def reset(self) -> None:
        """Reset the internal state of the policy."""
        # Block publishing immediately and clear any pending action
        with self._publish_lock:
            self.observation_space.is_ready = False
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

    def _callback_reset_service(self, channel, msg):
        """Service callback to reset policy state."""
        log.info(f"[INFO] Received callback reset service")

        self._resetting = False
        # Suspend comms (stops publisher/listener at LCM level)
        self.suspend_communications(services=False)
        # Schedule the actual reset after 2 second
        t = threading.Timer(2.0, self.reset)
        t.daemon = True
        t.start()
        time.sleep(2.0)
        log.info(f"[INFO] Finished callback reset service")
        return flag_t()

    def step(self):
        """Stepper loop: publish latest action if available."""
        with self._publish_lock:
            if self._resetting or self.latest_action is None:
                return
            self.publish_action(self.latest_action)
            self.latest_action = None

    def publish_action(self, action: np.ndarray):
        """Pack and publish action to downstream consumers."""
        if action is None:
            return

        self.action_space.pack_and_publish(action)

    @abstractmethod
    def predict(self, obs_seq: dict[str, Any]) -> np.ndarray:
        """Compute the action(s) from observations."""
        ...

    @staticmethod
    def _resolve_channel_types(mapping: dict[str, Any]) -> dict[str, type]:
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
