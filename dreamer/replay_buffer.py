"""
replay_buffer.py — Circular replay buffer for storing environment transitions.

Stores transitions and samples random contiguous subsequences for world model training.
Pre-allocated numpy arrays minimize memory fragmentation.

Key design decisions:
  - Circular buffer: when full, overwrites oldest transitions.
  - Sequences may span episode boundaries — the is_first flag marks resets so the
    RSSM can reset its hidden state at those positions.
  - sequence_length (T=64) must be <= total stored steps for sampling to work.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity circular buffer for RL transitions.

    All transitions are stored as flat numpy arrays indexed by a write pointer.
    Sampling returns contiguous subsequences of length `sequence_length`.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        sequence_length: int = 64,
    ):
        """
        Args:
            capacity:        maximum number of transitions (1,000,000 in paper)
            obs_dim:         observation dimensionality (3 for Pendulum, 17 for HalfCheetah)
            action_dim:      action dimensionality (1 for Pendulum, 6 for HalfCheetah)
            sequence_length: length T of sampled sequences (64 in paper)
        """
        # store capacity, obs_dim, action_dim, sequence_length as attributes
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        # initialize write pointer: self._ptr = 0
        self._ptr = 0
        # initialize fill counter: self._size = 0
        self._size = 0
        # pre-allocate storage arrays:
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._done = np.zeros(capacity, dtype=bool)
        self._is_first = np.zeros(capacity, dtype=bool)
        #   self._obs     = np.zeros((capacity, obs_dim), dtype=np.float32)
        #   self._action  = np.zeros((capacity, action_dim), dtype=np.float32)
        #   self._reward  = np.zeros(capacity, dtype=np.float32)
        #   self._done    = np.zeros(capacity, dtype=bool)
        #   self._is_first = np.zeros(capacity, dtype=bool)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        is_first: bool,
    ) -> None:
        """
        Adds a single transition to the buffer. Overwrites oldest entry when full.

        Args:
            obs:      shape (obs_dim,) — current observation
            action:   shape (action_dim,) — action taken
            reward:   scalar — reward received after taking action
            done:     bool — True if episode ended after this step
            is_first: bool — True if obs is the first observation of a new episode
        """
        # write obs, action, reward, done, is_first to self._ptr index in each array
        self._obs[self._ptr] = obs
        self._action[self._ptr] = action
        self._reward[self._ptr] = reward
        self._done[self._ptr] = done
        self._is_first[self._ptr] = is_first
        # advance write pointer: self._ptr = (self._ptr + 1) % self.capacity
        self._ptr = (self._ptr + 1) % self.capacity
        # update fill size: self._size = min(self._size + 1, self.capacity)
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """
        Samples a batch of contiguous subsequences of length sequence_length.

        Args:
            batch_size: number of sequences to return (B in training loop)
        Returns:
            dict with keys (all torch.float32 Tensors on CPU):
                'obs':      shape (B, T, obs_dim)
                'action':   shape (B, T, action_dim)
                'reward':   shape (B, T)
                'done':     shape (B, T)      — float (0.0 or 1.0)
                'is_first': shape (B, T)      — float (0.0 or 1.0)
        Raises:
            ValueError: if buffer has fewer than sequence_length transitions
        Notes:
            - Valid start indices: positions where index + sequence_length <= self._size
            - Use np.random.randint to sample start indices uniformly
            - Gather subsequences using fancy indexing:
                indices = np.arange(start, start + sequence_length) % capacity
              to handle wrap-around correctly.
        """
        # raise ValueError if self._size < self.sequence_length
        if self._size < self.sequence_length:
            raise ValueError(
                f"Not enough transitions to sample a sequence of length {self.sequence_length}. "
                f"Current size: {self._size}."
            )
        # compute max valid start: valid_range = self._size - self.sequence_length + 1
        valid_start_range = self._size - self.sequence_length + 1
        # sample batch_size start indices uniformly from [0, valid_range)
        start_indices = np.random.randint(0, valid_start_range, size = batch_size)
        idx = (start_indices[:, None] + np.arange(self.sequence_length)[None, :]) % self.capacity
        obs_batch = self._obs[idx]
        action_batch = self._action[idx]
        reward_batch = self._reward[idx]
        done_batch = self._done[idx].astype(np.float32)
        is_first_batch = self._is_first[idx].astype(np.float32)
        return {
            'obs': torch.from_numpy(obs_batch),
            'action': torch.from_numpy(action_batch),
            'reward': torch.from_numpy(reward_batch),
            'done': torch.from_numpy(done_batch),
            'is_first': torch.from_numpy(is_first_batch)
        }
        

    def __len__(self) -> int:
        """Returns the current number of stored transitions."""
        # return self._size
        return self._size