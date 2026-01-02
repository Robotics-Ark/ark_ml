from __future__ import annotations

import numpy as np
import torch
from arkml.algos.rl.sb3_algorithm import build_vector_env, _SB3_ALGOS
from arkml.core.app_context import ArkMLContext
from arkml.core.policy_node import PolicyNode

from ark_ml.arkml.core.policy_node import PolicyEnv


class SB3RLPolicyNode(PolicyNode):
    """Wrapper node for Stable Baseline"""

    def __init__(self, device: str):
        cfg = ArkMLContext.cfg["algo"]
        sb3_cfg = cfg["sb3"]
        model_cfg = cfg["model"]
        model_path = model_cfg["model_path"]
        if not model_path:
            raise ValueError(
                "SB3RLPolicyNode requires 'model_path' in cfg.algo.sb3.model_path "
            )

        algo_name = sb3_cfg["algo_name"]
        self._smoothing_cfg = sb3_cfg.get("action_smoothing")
        algo_cls = _SB3_ALGOS.get(algo_name)

        policy = algo_cls.load(model_path)

        super().__init__(
            policy=policy,
            device=device,
            policy_name=ArkMLContext.cfg.get("node_name"),
        )

        self._env = build_vector_env(
            env_cls=PolicyEnv,
            channel_schema=ArkMLContext.cfg["channel_schema"],
            global_config=ArkMLContext.cfg["global_config"],
            num_envs=1,
            asynchronous=False,
            sim=False,
            action_smoothing=self._smoothing_cfg,
        )

        self._prev_action: np.ndarray | None = None
        self._action_dim = None
        if hasattr(self._env, "action_space") and getattr(
            self._env.action_space, "shape", None
        ):
            self._action_dim = int(np.prod(self._env.action_space.shape))

    def _on_reset(self) -> None:
        """
        Policy specific reset function.

        Returns:
            None
        """
        self._prev_action = None

    def predict(self, obs) -> np.ndarray:
        """Compute the action for the given observation batch.

        Args:
          obs: Observation input to the policy.

        Returns:
          numpy.ndarray: Action vector.
        """
        with torch.no_grad():
            action, _ = self.policy.predict(obs, deterministic=True)
        return action
