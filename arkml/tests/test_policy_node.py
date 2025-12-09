import numpy as np

from ark.utils.scene_status_utils import task_space_action_from_obs
from arkml.core.policy_node import PolicyNode


class DummyPolicy:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.last_obs = None

    def to_device(self, device: str) -> None:
        self.device = device

    def set_eval_mode(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def predict(self, obs: dict) -> np.ndarray:
        self.last_obs = obs
        return task_space_action_from_obs(obs, self.action_dim)


class StubObservationSpace:
    def __init__(self, obs: dict):
        self._obs = obs

    def get_observation(self) -> dict:
        return self._obs

    def wait_until_observation_space_is_ready(self) -> None:
        pass


class StubActionSpace:
    def __init__(self):
        self.published = None

    def pack_and_publish(self, action) -> None:
        self.published = action


class DummyPolicyNode(PolicyNode):
    def __init__(self, policy: DummyPolicy, obs: dict, env):
        # Avoid BaseNode/LCM setup; only install what get_next_action needs.
        self.policy = policy
        self.name = "dummy_policy_node"
        self.observation_space = StubObservationSpace(obs)
        self.action_space = StubActionSpace()
        self._video_recorder = None
        self._record_video = False
        self.env = env

    def predict(self, obs_seq: dict) -> np.ndarray:
        return self.policy.predict(obs_seq)


class DummyEnv:
    def __init__(self):
        self.last_action = None

    def step(self, action):
        self.last_action = action


def test_policy_node_uses_observation_to_publish_action():
    obs = {
        "proprio::pose::position": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "proprio::pose::orientation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    action_dim = 7
    policy = DummyPolicy(action_dim=action_dim)
    env = DummyEnv()
    node = DummyPolicyNode(policy=policy, obs=obs, env=env)

    node.get_next_action()

    expected_action = task_space_action_from_obs(obs, action_dim)
    assert policy.last_obs is obs
    assert node.action_space.published is not None
    np.testing.assert_array_equal(node.action_space.published, expected_action)

    env.step(node.action_space.published)
    np.testing.assert_array_equal(env.last_action, expected_action)


if __name__ == "__main__":
    test_policy_node_uses_observation_to_publish_action()
