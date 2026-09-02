"""PointMass1D toy env in NM512/dreamerv3-torch interface.

Deterministic 1D point mass: move to origin. Optimal ~ -1, random ~ -5.
Matches the dict-obs, 4-tuple step interface used elsewhere in this repo.
"""

import gym
import numpy as np


class PointMass1D:
    metadata = {}

    def __init__(self, seed=0):
        self._rng = np.random.RandomState(seed)
        self._max_steps = 20
        self._x = 0.0
        self._t = 0

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "state": gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
            "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        })

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def _obs(self, is_first=False, is_terminal=False):
        return {
            "state": np.array([self._x], dtype=np.float32),
            "image": np.zeros((64, 64, 3), dtype=np.uint8),
            "is_first": is_first,
            "is_terminal": is_terminal,
        }

    def reset(self):
        self._x = float(self._rng.uniform(-0.9, 0.9))
        self._t = 0
        return self._obs(is_first=True, is_terminal=False)

    def step(self, action):
        assert np.isfinite(action).all(), action
        v = float(np.clip(action[0], -1.0, 1.0))
        self._x = float(np.clip(self._x + 0.1 * v, -1.0, 1.0))
        self._t += 1
        reward = -self._x ** 2
        done = self._t >= self._max_steps
        info = {"discount": np.float32(1.0)}
        return self._obs(is_first=False, is_terminal=False), reward, done, info

    def render(self, *args, **kwargs):
        return np.zeros((64, 64, 3), dtype=np.uint8)
