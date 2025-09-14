import threading
import queue
import numpy as np
from abc import abstractmethod, ABC
from typing import Any
from torch import nn
import json


from ark.client.comm_infrastructure.base_node import BaseNode


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
        channel_type,
        message_type,
        global_config=None,
    ):
        super().__init__("Policy", global_config)

        # Publishers/subscribers
        self.pub = self.create_publisher("next_action", message_type)
        self.create_subscriber("observation", channel_type, self.callback)

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
        self.create_stepper(5, self.step)

    def _inference_worker(self):
        """Background thread to run model inference asynchronously."""
        while not self._stop_event.is_set():
            try:
                obs_seq = self.obs_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            action = self.predict(obs_seq)
            self.latest_action = action

    def reset(self) -> None:
        """Reset the internal state of the policy."""
        self.policy.reset()
        self.latest_action = None
        # Clear any queued observations
        with self.obs_queue.mutex:
            self.obs_queue.queue.clear()
        # Allow subclasses to clear their own buffers
        reset_hook = getattr(self, "_on_reset", None)
        if callable(reset_hook):
            reset_hook()

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
            print("[EPISODE OVER]: Current episode is over")
        else:
            self.obs_queue.put_nowait(payload)

    def step(self):
        """Stepper loop: publish latest action if available."""
        if self.latest_action is None:
            return
        print(f"[ACTION PREDICTED] : {self.latest_action}")
        self.publish_action(self.latest_action)
        self.latest_action = None

    def publish_action(self, action: np.ndarray):
        """Publish action message. Subclasses may override for custom packing."""
        ...

    @abstractmethod
    def predict(self, obs_seq: dict[str, Any]) -> np.ndarray:
        """Compute the action(s) from observations."""
        ...
