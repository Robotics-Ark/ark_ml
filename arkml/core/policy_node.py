import os
from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from ark.client.comm_infrastructure.base_node import BaseNode
from ark.env.ark_env import ArkEnv
from ark.tools.log import log
from ark.utils.video_recorder import VideoRecorder
from arkml.core.app_context import ArkMLContext
from arktypes import flag_t, string_t
from torch import nn


class PolicyEnv(ArkEnv):
    def __init__(self, config_path: str, channel_schema: str):
        super().__init__(
            environment_name="policy_env",
            channel_schema=channel_schema,
            global_config=config_path,
            sim=True,
        )

    @staticmethod
    def _create_termination_conditions():
        return {}

    @staticmethod
    def _create_reward_functions():
        return {}

    def reset_objects(self):
        pass


class PolicyNode(ABC, BaseNode):
    """Abstract base class for policy wrappers with async inference.

    Args:
      policy: Underlying policy module to be executed.
      policy_name: Name of the policy service.
      device: Target device identifier (e.g., ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        policy: nn.Module,
        policy_name: str,
        device: str,
    ):
        super().__init__(policy_name, ArkMLContext.cfg["global_config"])

        cfg_dict = ArkMLContext.global_config

        if "channel_config" not in cfg_dict:
            raise ValueError("channel_config must not be empty and properly configured")

        self.mode = cfg_dict["policy_mode"]
        if self.mode not in ["service", "stepper"]:
            raise ValueError("Policy_mode must be 'service' or 'stepper'")

        log.info(f"Policy mode: {self.mode}")
        self._env = PolicyEnv(
            channel_schema=ArkMLContext.cfg["channel_schema"],
            config_path=ArkMLContext.cfg["global_config"],
        )
        self.obs = None
        self.debug = os.getenv("ARK_DEBUG", "").lower() in ("1", "true")

        # Policy setup
        self.policy = policy
        self.policy.to_device(device=device)
        self.policy.set_eval_mode()
        self.latest_action = None
        self._stop_service = True
        self._resetting = False
        self._video_recorder: VideoRecorder | None = None
        self._record_video = ArkMLContext.cfg["write_video"]
        self._video_cfg = ArkMLContext.cfg["video_cfg"]

        self.step_count = 0
        self.video_save_step = self._video_cfg["save_steps"]

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

        if self._record_video:
            self._start_video_recorder()

    def reset(self) -> None:
        """Reset the internal state of the policy."""
        self._stop_service = True
        self._resetting = True
        self.latest_action = None
        self.policy.reset()
        # Allow subclasses to clear their own buffers
        reset_hook = getattr(self, "_on_reset", None)
        if callable(reset_hook):
            reset_hook()
        self.obs, _ = self._env.reset()
        # Start a fresh recording after reset
        if self._video_recorder:
            self._video_recorder.close()
            self._start_video_recorder()
        self._resetting = False

    def _callback_reset_service(self, channel: str, msg: Any) -> string_t:
        """
        Service callback to reset policy state.
        Args:
            channel: Service channel id.
            msg: Message

        Returns:
            Reset status
        """
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

    def _callback_predict_service(self, channel: str, msg: Any) -> flag_t:
        """
        Service callback to predict the next action.
        Args:
            channel: Service channel id.
            msg: Message

        Returns:
            Returns dummy flag
        """
        self.get_next_action()
        return flag_t()

    def step_policy(self) -> None:
        """
        Step function to predict the next action.
        Returns:
            None
        """
        if self._stop_service:
            return
        self.get_next_action()

    def _callback_start_service(self, channel: str, msg: Any) -> flag_t:
        """
        Start policy prediction .
        Args:
            channel: Service channel id.
            msg: Message

        Returns:
            Returns dummy flag
        """
        log.info(f"[INFO] Received callback to start service")
        self._stop_service = False
        return flag_t()

    def _callback_stop_service(self, channel: str, msg: Any) -> flag_t:
        """
        Stop policy prediction service.
        Args:
            channel: Service channel id.
            msg: Message

        Returns:
            Returns dummy flag

        """
        log.info(f"[INFO] Received callback to stop service")
        self._stop_service = True
        return flag_t()

    def get_next_action(self) -> None:
        """
        Compute the next action from observations.
        Returns:
            None, Publishes the next action.
        """
        if self._resetting:
            return

        if self.obs is None:
            log.warning(f"Observation is None")
            return

        if self._record_video and self.step_count >= self.video_save_step:
            self._start_video_recorder()
            self.step_count = 0

        action = self.predict(self.obs)
        self.obs, _, _, _, _ = self._env.step(action)
        self.step_count += 1

        if self.debug:
            log.info(f"[ACTION PREDICTED] : {action}")
        if action is None:
            log.warning(f"Predicted action is None")
            return

        if self._video_recorder and self._record_video:
            self._video_recorder.add_frame(self.obs)
            if self.step_count >= self.video_save_step:
                self._video_recorder.close()

    @abstractmethod
    def predict(self, obs_seq: dict[str, Any]) -> np.ndarray:
        """
        Compute the action(s) from observations.
        Args:
            obs_seq: Observation sequence.

        Returns:
            Predicted next action.
        """
        ...

    def _start_video_recorder(self) -> None:
        output_dir = Path(ArkMLContext.cfg["output_dir"]) / "videos"
        filename = self._video_cfg.get(
            "filename", f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        fps = int(self._video_cfg.get("fps", 20))
        obs_key = self._video_cfg.get("obs_key", "rgb")
        out_path = output_dir / filename
        try:
            self._video_recorder = VideoRecorder(
                out_path=out_path, fps=fps, obs_rgb_key=obs_key
            )
            self._video_recorder.start()
            log.info(
                f"Video recording enabled: {out_path} (fps={fps}, obs_key={obs_key})"
            )
        except Exception as e:
            log.warning(
                f"Failed to start video recorder ({e}); disabling video recording"
            )
            self._video_recorder = None
            self._record_video = False
