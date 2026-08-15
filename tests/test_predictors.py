"""
Tests for dreamer/models/predictors.py

Run with: pytest tests/test_predictors.py -v
"""

import pytest
import torch
from dreamer.models.predictors import RewardPredictor, ContinuePredictor
from dreamer.utils.math_utils import symlog


@pytest.fixture
def state_dim():
    return 128 + 8 * 8  # deter=128, stoch=8*8=64 → 192


@pytest.fixture
def reward_pred(state_dim):
    return RewardPredictor(state_dim=state_dim, hidden_dim=64, num_layers=1, num_bins=51)


@pytest.fixture
def cont_pred(state_dim):
    return ContinuePredictor(state_dim=state_dim, hidden_dim=64, num_layers=1)


# ─────────────────────────────────────────────
# RewardPredictor tests
# ─────────────────────────────────────────────

def test_reward_pred_returns_dist(reward_pred, state_dim):
    """RewardPredictor.forward() should return a TwoHotDist."""
    from dreamer.utils.distributions import TwoHotDist
    feature = torch.randn(8, state_dim)
    dist = reward_pred(feature)
    assert isinstance(dist, TwoHotDist), \
        f"RewardPredictor must return TwoHotDist, got {type(dist)}"


def test_reward_pred_log_prob_shape(reward_pred, state_dim):
    """log_prob returns shape (B,) for (B,) targets."""
    feature = torch.randn(8, state_dim)
    rewards = torch.randn(8)
    dist = reward_pred(feature)
    lp = dist.log_prob(symlog(rewards))
    assert lp.shape == (8,), f"Expected (8,), got {lp.shape}"


def test_reward_pred_log_prob_sequence(reward_pred, state_dim):
    """RewardPredictor works on (B, T, state_dim) and returns (B, T) log_prob."""
    B, T = 4, 16
    feature = torch.randn(B, T, state_dim)
    rewards = torch.randn(B, T)
    dist = reward_pred(feature)
    lp = dist.log_prob(symlog(rewards))
    assert lp.shape == (B, T), f"Expected ({B}, {T}), got {lp.shape}"


def test_reward_pred_mean_shape(reward_pred, state_dim):
    """mean() returns (B,) in original reward scale."""
    feature = torch.randn(8, state_dim)
    dist = reward_pred(feature)
    mean = dist.mean()
    assert mean.shape == (8,), f"Expected (8,), got {mean.shape}"


def test_reward_pred_mean_finite(reward_pred, state_dim):
    """mean() must be finite."""
    feature = torch.randn(16, state_dim)
    dist = reward_pred(feature)
    assert torch.isfinite(dist.mean()).all()


def test_reward_pred_loss_finite_and_backward(reward_pred, state_dim):
    """Reward prediction loss should be finite and gradients should flow."""
    feature = torch.randn(8, state_dim)
    rewards = torch.randn(8)
    dist = reward_pred(feature)
    loss = -dist.log_prob(symlog(rewards)).mean()
    assert torch.isfinite(loss), f"Reward loss is not finite: {loss.item()}"
    loss.backward()
    grads = [p.grad for p in reward_pred.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed through RewardPredictor"


def test_reward_pred_output_init_near_zero(state_dim):
    """With zero-initialized output weights, initial predictions should be near zero."""
    pred = RewardPredictor(state_dim=state_dim, hidden_dim=64, num_layers=1, num_bins=51)
    with torch.no_grad():
        feature = torch.zeros(1, state_dim)
        dist = pred(feature)
        # mean should be near 0 if output weights are zero-initialized
        mean = dist.mean()
    assert abs(mean.item()) < 5.0, \
        f"Initial reward prediction should be near 0 (zero-init), got {mean.item()}"


# ─────────────────────────────────────────────
# ContinuePredictor tests
# ─────────────────────────────────────────────

def test_cont_pred_returns_bernoulli(cont_pred, state_dim):
    """ContinuePredictor returns a Bernoulli distribution."""
    feature = torch.randn(8, state_dim)
    dist = cont_pred(feature)
    assert isinstance(dist, torch.distributions.Bernoulli), \
        f"ContinuePredictor must return Bernoulli, got {type(dist)}"


def test_cont_pred_probs_in_01(cont_pred, state_dim):
    """Bernoulli probs must be in (0, 1)."""
    feature = torch.randn(16, state_dim)
    dist = cont_pred(feature)
    probs = dist.probs
    assert (probs > 0).all() and (probs < 1).all(), \
        f"Bernoulli probs must be in (0, 1). Got min={probs.min().item()}, max={probs.max().item()}"


def test_cont_pred_log_prob_shape(cont_pred, state_dim):
    """log_prob returns shape (B,)."""
    feature = torch.randn(8, state_dim)
    continues = torch.ones(8)  # non-terminal
    dist = cont_pred(feature)
    lp = dist.log_prob(continues)
    assert lp.shape == (8,), f"Expected (8,), got {lp.shape}"


def test_cont_pred_log_prob_sequence(cont_pred, state_dim):
    """ContinuePredictor works on (B, T, state_dim) input."""
    B, T = 4, 16
    feature = torch.randn(B, T, state_dim)
    continues = torch.ones(B, T)
    dist = cont_pred(feature)
    lp = dist.log_prob(continues)
    assert lp.shape == (B, T), f"Expected ({B}, {T}), got {lp.shape}"


def test_cont_pred_loss_backward(cont_pred, state_dim):
    """Continue loss should be finite with backprop."""
    feature = torch.randn(4, state_dim)
    continues = torch.ones(4)
    dist = cont_pred(feature)
    loss = -dist.log_prob(continues).mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in cont_pred.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed through ContinuePredictor"
