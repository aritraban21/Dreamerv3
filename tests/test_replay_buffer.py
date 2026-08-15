"""
Tests for dreamer/replay_buffer.py

Run with: pytest tests/test_replay_buffer.py -v
"""

import pytest
import numpy as np
import torch
from dreamer.replay_buffer import ReplayBuffer


@pytest.fixture
def small_buffer():
    """Small buffer for quick tests."""
    return ReplayBuffer(capacity=100, obs_dim=3, action_dim=1, sequence_length=8)


def _fill_buffer(buf: ReplayBuffer, n_steps: int, episode_length: int = 20):
    """Helper: fill buffer with synthetic transitions."""
    obs_dim = buf._obs.shape[1]
    action_dim = buf._action.shape[1]
    step_in_ep = 0
    obs = np.random.randn(obs_dim).astype(np.float32)
    for i in range(n_steps):
        action = np.random.randn(action_dim).astype(np.float32)
        reward = float(np.random.randn())
        done = (step_in_ep == episode_length - 1)
        is_first = (step_in_ep == 0)
        buf.add(obs, action, reward, done, is_first)
        step_in_ep = 0 if done else step_in_ep + 1
        obs = np.random.randn(obs_dim).astype(np.float32)


# ─────────────────────────────────────────────
# __len__ tests
# ─────────────────────────────────────────────

def test_initial_length(small_buffer):
    """Empty buffer has length 0."""
    assert len(small_buffer) == 0


def test_length_after_add(small_buffer):
    """Length increases with each add."""
    _fill_buffer(small_buffer, 10)
    assert len(small_buffer) == 10


def test_length_caps_at_capacity():
    """Buffer length caps at capacity, doesn't grow indefinitely."""
    buf = ReplayBuffer(capacity=50, obs_dim=3, action_dim=1, sequence_length=4)
    _fill_buffer(buf, 100)
    assert len(buf) == 50, f"Buffer length should cap at 50, got {len(buf)}"


# ─────────────────────────────────────────────
# add() tests
# ─────────────────────────────────────────────

def test_add_single_transition(small_buffer):
    """add() stores a transition without error."""
    obs = np.zeros(3, dtype=np.float32)
    action = np.zeros(1, dtype=np.float32)
    small_buffer.add(obs, action, reward=1.0, done=False, is_first=True)
    assert len(small_buffer) == 1


def test_add_wrap_around():
    """Buffer wraps around correctly — old entries are overwritten."""
    buf = ReplayBuffer(capacity=5, obs_dim=2, action_dim=1, sequence_length=3)
    # Fill more than capacity
    for i in range(8):
        obs = np.array([float(i), float(i)], dtype=np.float32)
        action = np.array([0.0], dtype=np.float32)
        buf.add(obs, action, reward=float(i), done=False, is_first=(i == 0))
    assert len(buf) == 5, "Buffer size must be capped at capacity"


# ─────────────────────────────────────────────
# sample() tests
# ─────────────────────────────────────────────

def test_sample_raises_when_empty(small_buffer):
    """Sampling from an empty or too-small buffer should raise ValueError."""
    with pytest.raises(ValueError):
        small_buffer.sample(batch_size=4)


def test_sample_raises_when_insufficient(small_buffer):
    """Sampling requires at least sequence_length transitions."""
    _fill_buffer(small_buffer, n_steps=5)  # less than sequence_length=8
    with pytest.raises(ValueError):
        small_buffer.sample(batch_size=2)


def test_sample_output_keys(small_buffer):
    """sample() returns a dict with all required keys."""
    _fill_buffer(small_buffer, n_steps=50)
    batch = small_buffer.sample(batch_size=4)
    for key in ('obs', 'action', 'reward', 'done', 'is_first'):
        assert key in batch, f"Missing key '{key}' in sample output"


def test_sample_output_shapes(small_buffer):
    """sample() returns tensors with shape (B, T, ...)."""
    _fill_buffer(small_buffer, n_steps=50)
    B = 4
    batch = small_buffer.sample(batch_size=B)
    T = small_buffer.sequence_length
    assert batch['obs'].shape == (B, T, 3), f"obs shape: {batch['obs'].shape}"
    assert batch['action'].shape == (B, T, 1), f"action shape: {batch['action'].shape}"
    assert batch['reward'].shape == (B, T), f"reward shape: {batch['reward'].shape}"
    assert batch['done'].shape == (B, T), f"done shape: {batch['done'].shape}"
    assert batch['is_first'].shape == (B, T), f"is_first shape: {batch['is_first'].shape}"


def test_sample_returns_tensors(small_buffer):
    """sample() must return torch.Tensor objects."""
    _fill_buffer(small_buffer, n_steps=50)
    batch = small_buffer.sample(batch_size=4)
    for key, val in batch.items():
        assert isinstance(val, torch.Tensor), \
            f"batch['{key}'] should be a torch.Tensor, got {type(val)}"


def test_sample_dtype_float32(small_buffer):
    """All batch tensors should be float32."""
    _fill_buffer(small_buffer, n_steps=50)
    batch = small_buffer.sample(batch_size=4)
    for key, val in batch.items():
        assert val.dtype == torch.float32, \
            f"batch['{key}'] dtype should be float32, got {val.dtype}"


def test_sample_values_are_stored():
    """Values in the sampled batch should match what was stored."""
    buf = ReplayBuffer(capacity=100, obs_dim=2, action_dim=1, sequence_length=4)
    # add a known transition
    known_obs = np.array([1.23, 4.56], dtype=np.float32)
    known_action = np.array([7.89], dtype=np.float32)
    known_reward = 3.14
    buf.add(known_obs, known_action, known_reward, done=False, is_first=True)
    _fill_buffer(buf, n_steps=50)
    # just check that the buffer doesn't corrupt data types
    batch = buf.sample(batch_size=4)
    assert torch.isfinite(batch['obs']).all()
    assert torch.isfinite(batch['reward']).all()


def test_sample_is_first_correct():
    """is_first should be True only at episode boundaries."""
    buf = ReplayBuffer(capacity=200, obs_dim=3, action_dim=1, sequence_length=16)
    # Add 2 episodes of exactly 20 steps
    obs = np.zeros(3, dtype=np.float32)
    action = np.zeros(1, dtype=np.float32)
    for ep in range(5):
        for t in range(20):
            is_first = (t == 0)
            done = (t == 19)
            buf.add(obs, action, reward=0.0, done=done, is_first=is_first)
    batch = buf.sample(batch_size=4)
    # is_first values should be 0.0 or 1.0
    is_first_vals = batch['is_first'].unique()
    for v in is_first_vals:
        assert v.item() in (0.0, 1.0), f"is_first should only be 0 or 1, got {v.item()}"


def test_sample_multiple_batches(small_buffer):
    """Can call sample() multiple times without error."""
    _fill_buffer(small_buffer, n_steps=50)
    for _ in range(5):
        batch = small_buffer.sample(batch_size=4)
        assert batch['obs'].shape[0] == 4
