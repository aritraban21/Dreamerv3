"""Pendulum-v1 in the NM512/dreamerv3-torch env interface (state-based obs).

Wraps classic gym Pendulum-v1 for DYNAMICS only, and exposes the same dict-obs,
4-tuple-step interface used by envs/pointmass.py so the reference DreamerV3 can
train on it with a purely state-based (MLP) encoder/decoder — matching our own
Pendulum setup (obs = [cos theta, sin theta, theta_dot], no image).

Notes:
  - obs['state'] = [cos, sin, theta_dot]  (identical to gym Pendulum-v1 obs).
  - A dummy zero 'image' key is included only because WorldModel.preprocess()
    unconditionally does obs['image']/255.0; the encoder/decoder ignore it
    (mlp_keys='state', cnn_keys='$^').
  - theta is recoverable from the state as arctan2(sin, cos); theta=0 is upright.
    We keep it for rendering the pendulum in the parity videos.
"""

import gym
import numpy as np


class PendulumState:
    metadata = {}

    def __init__(self, seed=0):
        # Underlying classic-control Pendulum. Its own TimeLimit is bypassed —
        # the reference wraps make_env's result in wrappers.TimeLimit(config.time_limit).
        self._env = gym.make("Pendulum-v1").unwrapped
        self._seed = seed
        try:
            self._env.reset(seed=seed)
        except TypeError:
            # very old gym
            self._env.seed(seed)
            self._env.reset()
        self._last_state = np.zeros(3, dtype=np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "state": gym.spaces.Box(-8.0, 8.0, (3,), dtype=np.float32),
            "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        })

    @property
    def action_space(self):
        # Real torque range; wrappers.NormalizeActions maps agent's [-1,1] to this.
        return gym.spaces.Box(-2.0, 2.0, (1,), dtype=np.float32)

    def _obs(self, raw_obs, is_first=False, is_terminal=False):
        state = np.asarray(raw_obs, dtype=np.float32)
        self._last_state = state
        return {
            "state": state,
            "image": np.zeros((64, 64, 3), dtype=np.uint8),
            "is_first": is_first,
            "is_terminal": is_terminal,
        }

    def reset(self):
        out = self._env.reset()
        raw = out[0] if isinstance(out, tuple) else out
        return self._obs(raw, is_first=True, is_terminal=False)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        out = self._env.step(action)
        if len(out) == 5:                       # gymnasium-style, just in case
            raw, reward, term, trunc, info = out
            done = bool(term or trunc)
        else:                                   # classic gym 4-tuple
            raw, reward, done, info = out
        # Pendulum never terminates early; the reference TimeLimit sets done at
        # time_limit. is_terminal stays False so the cont head learns discount=1.
        obs = self._obs(raw, is_first=False, is_terminal=False)
        return obs, float(reward), False, {"discount": np.float32(1.0)}

    @property
    def theta(self):
        """Angle from upright (radians), recovered from obs. theta=0 is up."""
        cos_t, sin_t = float(self._last_state[0]), float(self._last_state[1])
        return float(np.arctan2(sin_t, cos_t))

    def render(self, *args, **kwargs):
        return np.zeros((64, 64, 3), dtype=np.uint8)
