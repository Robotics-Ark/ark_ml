import torch


class RobotNode:
    """Manages env-policy interactions.

    Args:
      env: Environment object.
      policy_node: Policy wrapper.
      obs_horizon: Number of most recent observations to stack for the policy.
    """

    def __init__(self, env, policy_node, obs_horizon=8):
        self.env = env
        self.policy_node = policy_node
        self.obs_horizon = obs_horizon
        self.obs_history = []

    def reset(self):
        """Reset environment and bootstrap observation history.

        Returns:
          ``(obs, info)`` as returned by ``env.reset()``
        """
        obs, info = self.env.reset()
        self.obs_history = [obs] * self.obs_horizon  # bootstrap with repeated obs
        return obs, info

    def step(self, action):
        """Step the environment once and update history.

        Args:
          action: Action to apply to the environment.

        Returns:
          ``(obs, reward, terminated, truncated, info)`` from the env.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_history.append(obs)
        if len(self.obs_history) > self.obs_horizon:
            self.obs_history.pop(0)  # keep window size fixed
        return obs, reward, terminated, truncated, info

    def get_obs_sequence(self, task_prompt:str):
        """Prepare a stacked observation batch for the policy.

        Args:
          task_prompt: Natural language instruction to include in the batch.

        Returns:
          Prepared observation batch, typically containing keys like
          ``image`` (Tensor ``[B,C,H,W]``), ``state`` (Tensor ``[B,D]``), and
          ``task`` (list of length ``B``).
        """
        obs_history = self.obs_history[-self.obs_horizon:]
        return self.prepare_observations(observations=obs_history, task_prompt=task_prompt)

    @staticmethod
    def prepare_observations(observations:list[dict[str,...]], task_prompt:str):
        """Convert raw env observations into a batched policy input.

        Args:
          observations: List of observation dicts from the env. Each dict is
            expected to contain keys ``images`` (tuple with RGB as HxWxC),
            ``cube``, ``target``, ``gripper``, and ``franka_ee``.
          task_prompt: Natural language task description to include in the
            batch.

        Returns:
          A batch dictionary with:
            - ``image``: ``torch.FloatTensor`` of shape ``[B, C, H, W]``.
            - ``state``: ``torch.FloatTensor`` of shape ``[B, D]``.
            - ``task``: ``list[str]`` repeated across the batch (length ~= ``B``).
        """
        imgs = []
        states = []
        obs = {}
        for ob in observations:
            # ---- Image ----
            img = ob["images"][0]  # (H, W, C)
            img = torch.from_numpy(img.copy()).permute(2, 0, 1)  # (C, H, W)
            img = img.float() / 255.0
            imgs.append(img)

            # ---- State ----
            state = []
            state.extend(list(ob["cube"]))
            state.extend(list(ob["target"]))
            state.extend(list(ob["gripper"]))
            state.extend(list(ob["franka_ee"][0]))
            states.append(torch.tensor(state, dtype=torch.float32))

        obs['image'] = torch.stack(imgs, dim=0)
        obs['state'] = torch.stack(states, dim=0)
        obs['task'] =  [task_prompt] * obs['state'].shape[0]
        return obs

    def run_episode(self, max_steps: int, action_horizon: int, step_sleep: float, start_offset: int, task_prompt: str):
        """Run one episode using the current policy and environment.

        Args:
          max_steps: Maximum number of env steps to execute.
          action_horizon: Number of actions to apply from a predicted sequence.
          step_sleep: Optional delay (seconds) between executed actions.
          start_offset: Index into the predicted sequence to start from.
          task_prompt: Natural language instruction to include in observations.

        Returns:
          ``(obs_history, terminated, truncated)`` where
          ``obs_history`` is the internal observation buffer, and the booleans
          indicate episode termination status.
        """
        _, _ = self.reset()
        terminated = False
        truncated = False

        for _ in range(max_steps):
            model_input = self.get_obs_sequence(task_prompt=task_prompt)
            actions = self.policy_node.predict(model_input)

            # Normalize to a sequence of actions [T, action_dim]
            if actions is None:
                raise RuntimeError("Policy returned None for actions.")
            if actions.ndim == 1:
                actions_seq = actions[None, :]
            else:
                actions_seq = actions[start_offset:start_offset + action_horizon]

            for a in actions_seq:
                _, _, terminated, truncated, _ = self.step(a)
                if step_sleep:
                    import time as _t
                    _t.sleep(step_sleep)
                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        return self.obs_history, terminated, truncated
