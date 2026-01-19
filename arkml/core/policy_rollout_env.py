from __future__ import annotations

from typing import Any

from ark.env.ark_env import ArkEnv
from ark.utils.config_utils import resolve_class
from arkml.core.rl.termination_conditions.base_termination_conditions import (
    BaseTerminationCondition,
)
from arkml.core.rl.termination_conditions.termination_provider import (
    TerminationProvider,
)
from arkml.core.rl.termination_conditions.timeout import Timeout
from arkml.nodes.policy_registry import get_policy_model
from omegaconf import DictConfig


class PolicyRolloutEnv(ArkEnv):
    """
    Environment wrapper that owns policy rollout and termination handling.
    """

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        rollout_cfg = cfg.rollout_env
        self._termination_cfg = cfg.termination
        self._policy_cfg = cfg.policy
        self._timeout_steps = getattr(rollout_cfg, "max_steps", None)
        self._rollout_terminations: dict[str, BaseTerminationCondition] = {}

        super().__init__(
            environment_name=rollout_cfg.env_name,
            channel_schema=rollout_cfg.channel_schema,
            global_config=rollout_cfg.global_config,
            sim=rollout_cfg.sim,
            namespace=rollout_cfg.namespace,
        )

        self.policy = self._build_policy(self._policy_cfg)
        self._policy_device = self._policy_cfg.get("device", "cpu")
        if hasattr(self.policy, "to_device"):
            self.policy.to_device(self._policy_device)
        if hasattr(self.policy, "set_eval_mode"):
            self.policy.set_eval_mode()

    def _create_termination_conditions(self) -> dict[str, BaseTerminationCondition]:
        terminations: dict[str, BaseTerminationCondition] = {}
        term_name = getattr(self._termination_cfg, "name", "termination")
        class_path = getattr(self._termination_cfg, "class_path", None)
        kwargs = dict(getattr(self._termination_cfg, "kwargs", {}) or {})

        if class_path:
            term_cls = resolve_class(class_path)
            term_obj = term_cls(**kwargs)
            if isinstance(term_obj, BaseTerminationCondition):
                terminations[term_name] = term_obj
            elif isinstance(term_obj, TerminationProvider):
                terminations = term_obj.build(cfg=self._cfg) or {}
            else:
                raise TypeError(
                    "Termination class must be a BaseTerminationCondition or "
                    "TerminationProvider."
                )

        if "timeout" not in terminations:
            timeout_steps = kwargs.get("max_steps", self._timeout_steps)
            terminations.setdefault("timeout", Timeout(max_steps=timeout_steps or 500))

        self._rollout_terminations = terminations
        return terminations

    @staticmethod
    def _create_reward_functions() -> dict[str, Any]:
        return {}

    def reset_objects(self) -> None:
        pass

    def _build_policy(self, policy_cfg: DictConfig):
        policy_cls = get_policy_model(self._cfg)
        policy_kwargs = dict(getattr(policy_cfg, "kwargs", {}) or {})
        return policy_cls(**policy_kwargs)

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        for condition in self._rollout_terminations.values():
            condition.reset()
        return obs

    def _step_terminations(self, obs: dict[str, Any]) -> tuple[bool, bool]:
        done = False
        success = False
        for condition in self._rollout_terminations.values():
            cond_done, cond_success = condition.step(obs)
            done = done or cond_done
            success = success or cond_success
        return done, success

    def rollout_forever(self, max_episodes: int | None = None) -> None:
        """
        Run policy inference until termination/timeout, then reset and repeat.
        """
        episodes = 0
        obs = self.reset()

        while True:
            action = self.policy.predict(obs)
            result = self.step(action)

            if isinstance(result, tuple) and len(result) == 5:
                obs, reward, terminated, truncated, info = result
                env_done = terminated or truncated
            elif isinstance(result, tuple) and len(result) == 4:
                obs, reward, env_done, info = result
            else:
                raise ValueError(f"Unexpected env.step return size: {len(result)}")

            term_done, term_success = self._step_terminations(obs)
            if env_done or term_done:
                episodes += 1
                if max_episodes is not None and episodes >= max_episodes:
                    return
                obs = self.reset()
