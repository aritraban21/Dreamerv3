"""
pointmass.py — 1D PointMass toy environment for sanity-checking DreamerV3.

Deterministic dynamics, dense reward, 1D obs, 1D action. If DreamerV3 fails
to learn this in a few thousand env steps, the algorithm has a bug — Pendulum
would never work either.

Optimal return per episode ≈ -0.5 (reach origin, hold).
Random-policy return    ≈ -7.
Episode length          = 20 steps.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PointMass1D(gym.Env):
    """1D point mass: move to origin. Optimal ≈ -0.5, random ≈ -7."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # obs = agent position ∈ [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # action = velocity ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._x = 0.0
        self._t = 0
        self._max_steps = 20

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._x = float(self.np_random.uniform(-0.9, 0.9))
        self._t = 0
        return np.array([self._x], dtype=np.float32), {}

    def step(self, action):
        v = float(np.clip(action[0], -1.0, 1.0))
        self._x = float(np.clip(self._x + 0.1 * v, -1.0, 1.0))
        self._t += 1
        reward = -self._x ** 2
        terminated = False
        truncated = self._t >= self._max_steps
        return (
            np.array([self._x], dtype=np.float32),
            reward,
            terminated,
            truncated,
            {},
        )


gym.register(id="PointMass1D-v0", entry_point=PointMass1D)
