import threading
import queue
import numpy as np
from abc import abstractmethod, ABC
from typing import Any
from torch import nn
import json
import importlib
from pathlib import Path
import yaml
from ark.tools.log import log

from ark.client.comm_infrastructure.base_node import BaseNode
from arktypes.utils import pack
from arktypes import flag_t


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
        stepper_frequency:int,
        global_config=None,
        channel_config_path: str | Path | None = None,
    ):
        super().__init__("Policy", global_config)

        # Channel config to publish and subscribe
        channel_cfg_path = Path(channel_config_path)
        if not channel_cfg_path.exists():
            with open(channel_cfg_path, 'r') as f:
                cfg_dict = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file could not found {channel_config_path}")

        obs_channels = self._resolve_channel_types(
            cfg_dict.get("observation_channels", {})
        )
        act_channels = self._resolve_channel_types(cfg_dict.get("action_channels", {}))

        if obs_channels:
            self.obs_listener = self.create_multi_channel_listener(obs_channels)
        else:
            raise NotImplementedError("No observation channels found")

        if act_channels:
            self.action_pub = self.create_multi_channel_publisher(act_channels)
        else:
            raise NotImplementedError("No action channels found")

        # Policy setup
        self.policy = policy
        self.policy.to_device(device=device)
        self.policy.set_eval_mode()

        # Async inference infra
        self.obs_queue = queue.Queue(maxsize=1)  # only keep latest obs
        self.latest_action = None
        self._stop_event = threading.Event()

        self.worker_thread = threading.Thread(
            target=self._inference_worker, daemon=True
        )
        self.worker_thread.start()

        # Stepper publishes actions at fixed control frequency
        self.create_stepper(stepper_frequency, self.step)

        # Expose a reset service so external clients can reset policy state
        self._reset_service_name = f"{self.name}/policy/reset"
        self.create_service(
            self._reset_service_name, flag_t, flag_t, self._callback_reset_service
        )

    def _inference_worker(self):
        """Background thread to run model inference asynchronously."""
        while not self._stop_event.is_set():
            # Poll latest messages across all observation channels
            obs_dict = self.obs_listener.get()
            print(f"obs_dict : {obs_dict}")
            if not obs_dict or any(v is None for v in obs_dict.values()):
                # Incomplete observation set; try again shortly
                continue
            action = self.predict(obs_dict)
            self.latest_action = action

    def reset(self) -> None:
        """Reset the internal state of the policy."""
        self.policy.reset()
        self.latest_action = None
        with self.obs_queue.mutex:
            self.obs_queue.queue.clear()
        # Allow subclasses to clear their own buffers
        reset_hook = getattr(self, "_on_reset", None)
        if callable(reset_hook):
            reset_hook()

    def _callback_reset_service(self, channel, msg):
        """Service callback to reset policy state.

        Returns an empty flag_t response after clearing internal buffers.
        """
        # Pause non-service comms to avoid races during reset
        self.suspend_communications(services=False)
        try:
            self.reset()
        finally:
            self.resume_communications(services=False)
        return flag_t()

    def callback(self, t, channel_name, msg):
        """Subscriber callback for new observations."""
        # Drop old obs
        try:
            self.obs_queue.get_nowait()
        except queue.Empty:
            pass

        if isinstance(msg, str):
            payload = json.loads(msg)
        elif hasattr(msg, "data") and isinstance(msg.data, str):
            payload = json.loads(msg.data)
        elif isinstance(msg, dict):
            payload = msg
        else:
            raise ValueError("Unsupported observation format")

        if "episode_over" in payload and payload["episode_over"]:
            self.reset()
            log.info("[EPISODE OVER]: Current episode is over")
        else:
            self.obs_queue.put_nowait(payload)

    def step(self):
        """Stepper loop: publish latest action if available."""
        if self.latest_action is None:
            return
        log.info(f"[ACTION PREDICTED] : {self.latest_action}")
        self.publish_action(self.latest_action)
        self.latest_action = None

    def publish_action(self, action: np.ndarray):
        """Pack and publish action to downstream consumers."""
        if action is None or action.shape[0] < 8:
            return

        xyz = np.asarray(action[:3], dtype=np.float32)
        quat = np.asarray(action[3:7], dtype=np.float32)
        grip = float(action[7])
        msg = pack.task_space_command("all", xyz, quat, grip)

        self.action_pub.publish({"nex_action": msg})

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
