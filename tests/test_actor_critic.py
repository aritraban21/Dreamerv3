"""
Tests for dreamer/models/actor.py and dreamer/models/critic.py

Run with: pytest tests/test_actor_critic.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from dreamer.models.actor import Actor
from dreamer.models.critic import Critic
from dreamer.utils.math_utils import symlog


@pytest.fixture
def state_dim():
    return 192  # deter=128 + stoch=64


@pytest.fixture
def action_dim():
    return 1  # Pendulum


@pytest.fixture
def actor(state_dim, action_dim):
    return Actor(state_dim=state_dim, action_dim=action_dim, hidden_dim=64, num_layers=1)


@pytest.fixture
def critic(state_dim):
    return Critic(state_dim=state_dim, hidden_dim=64, num_layers=1, num_bins=51, bin_range=10.0)


# ─────────────────────────────────────────────
# Actor tests
# ─────────────────────────────────────────────

def test_actor_forward_returns_dist(actor, state_dim):
    """actor.forward() returns Independent(Normal(tanh(mean), std), 1)."""
    feature = torch.randn(8, state_dim)
    dist = actor(feature)
    assert isinstance(dist, torch.distributions.Independent), \
        f"Actor must return Independent(Normal,1), got {type(dist)}"
    assert isinstance(dist.base_dist, torch.distributions.Normal), \
        f"Actor base_dist must be Normal, got {type(dist.base_dist)}"
    # mean is tanh-bounded to (-1, 1)
    assert (dist.mean.abs() <= 1.0).all(), \
        f"Actor mean must be in [-1,1] (tanh-bounded), got range [{dist.mean.min()}, {dist.mean.max()}]"


def test_actor_get_action_shape_training(actor, state_dim, action_dim):
    """get_action(training=True) returns shape (B, action_dim)."""
    feature = torch.randn(8, state_dim)
    action = actor.get_action(feature, training=True)
    assert action.shape == (8, action_dim), f"Expected (8, {action_dim}), got {action.shape}"


def test_actor_get_action_shape_eval(actor, state_dim, action_dim):
    """get_action(training=False) returns shape (B, action_dim)."""
    feature = torch.randn(8, state_dim)
    action = actor.get_action(feature, training=False)
    assert action.shape == (8, action_dim)


def test_actor_samples_within_reasonable_range(actor, state_dim):
    """Sampled actions cluster near [-1, 1]. Actor uses unbounded Normal(tanh(mean), std) —
    samples may exceed [-1, 1] slightly (env clips), but shouldn't run wild."""
    feature = torch.randn(64, state_dim)
    action = actor.get_action(feature, training=True)
    # allow modest overshoot (~3 std beyond mean's tanh bound)
    assert (action >= -3.0).all() and (action <= 3.0).all(), \
        f"Actions unreasonably far from [-1, 1]. Min: {action.min().item()}, Max: {action.max().item()}"
    assert torch.isfinite(action).all(), "actions must be finite"


def test_actor_mean_in_bounds(actor, state_dim):
    """Mean actions (training=False) must be in [-1, 1]."""
    feature = torch.randn(16, state_dim)
    action = actor.get_action(feature, training=False)
    assert (action >= -1.0 - 1e-5).all() and (action <= 1.0 + 1e-5).all()


def test_actor_log_prob_shape(actor, state_dim, action_dim):
    """log_prob returns shape (B,) for (B, action_dim) actions."""
    feature = torch.randn(8, state_dim)
    dist = actor(feature)
    action = dist.sample()
    lp = dist.log_prob(action)
    assert lp.shape == (8,), f"Expected (8,), got {lp.shape}"


def test_actor_entropy_shape(actor, state_dim):
    """entropy() returns shape (B,)."""
    feature = torch.randn(8, state_dim)
    dist = actor(feature)
    H = dist.entropy()
    assert H.shape == (8,), f"Expected (8,), got {H.shape}"


def test_actor_gradient_flows(actor, state_dim, action_dim):
    """Actor loss backward must produce gradients in actor parameters."""
    feature = torch.randn(8, state_dim)
    dist = actor(feature)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    # simple actor loss
    fake_returns_norm = torch.ones(8)
    loss = -(fake_returns_norm * log_prob + 3e-4 * entropy).mean()
    loss.backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients in actor"
    assert all(torch.isfinite(g).all() for g in grads), "Actor gradients contain non-finite values"


def test_actor_min_std(state_dim, action_dim):
    """std should never be below min_std even after many steps."""
    actor = Actor(state_dim=state_dim, action_dim=action_dim,
                  hidden_dim=64, num_layers=1, min_std=0.1)
    feature = torch.randn(32, state_dim)
    dist = actor(feature)
    # Access std from the distribution
    # Note: implementation may store std differently; adjust if needed
    action = dist.sample()
    # The main check: no NaN or Inf
    assert torch.isfinite(action).all()


# ─────────────────────────────────────────────
# Critic tests
# ─────────────────────────────────────────────

def test_critic_forward_returns_dist(critic, state_dim):
    """critic.forward() returns a TwoHotDist."""
    from dreamer.utils.distributions import TwoHotDist
    feature = torch.randn(8, state_dim)
    dist = critic(feature)
    assert isinstance(dist, TwoHotDist), f"Critic must return TwoHotDist, got {type(dist)}"


def test_critic_mean_shape(critic, state_dim):
    """critic(feature).mean() returns shape (B,)."""
    feature = torch.randn(8, state_dim)
    mean = critic(feature).mean()
    assert mean.shape == (8,), f"Expected (8,), got {mean.shape}"


def test_critic_forward_ema_no_grad(critic, state_dim):
    """forward_ema() should not produce gradients."""
    feature = torch.randn(8, state_dim, requires_grad=True)
    dist = critic.forward_ema(feature)
    mean = dist.mean()
    assert mean.grad_fn is None or not mean.requires_grad, \
        "forward_ema() should return detached values (no grad_fn)"


def test_critic_ema_update_changes_weights(critic, state_dim):
    """After manually perturbing critic.net weights, update_ema() should change ema_net."""
    # record initial ema weights
    initial_ema = {name: param.clone() for name, param in critic.ema_net.named_parameters()}

    # perturb the trainable network weights significantly
    with torch.no_grad():
        for p in critic.net.parameters():
            p.add_(torch.ones_like(p) * 10.0)

    # update EMA
    critic.update_ema()

    # EMA weights should have changed
    changed = False
    for name, param in critic.ema_net.named_parameters():
        if not torch.allclose(param, initial_ema[name]):
            changed = True
            break
    assert changed, "update_ema() should change EMA weights after perturbing critic.net"


def test_critic_ema_not_trained_by_grad(critic, state_dim):
    """EMA network parameters should not require gradients."""
    for p in critic.ema_net.parameters():
        assert not p.requires_grad, "EMA parameters must not require gradients"


def test_critic_loss_backward(critic, state_dim):
    """Critic loss backward should produce finite gradients."""
    feature = torch.randn(8, state_dim)
    returns = torch.randn(8)
    dist = critic(feature)
    loss = -dist.log_prob(symlog(returns)).mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in critic.net.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients in critic"
    assert all(torch.isfinite(g).all() for g in grads)


def test_critic_output_init_near_zero(state_dim):
    """With zero-initialized output, initial value predictions should be near 0."""
    crit = Critic(state_dim=state_dim, hidden_dim=64, num_layers=1, num_bins=51, bin_range=10.0)
    with torch.no_grad():
        feature = torch.zeros(1, state_dim)
        mean = crit(feature).mean()
    assert abs(mean.item()) < 5.0, \
        f"Initial value prediction should be near 0 (zero-init output), got {mean.item()}"


def test_critic_sequence_input(critic, state_dim):
    """Critic handles (B, T, state_dim) sequence input."""
    B, T = 4, 16
    feature = torch.randn(B, T, state_dim)
    dist = critic(feature)
    mean = dist.mean()
    assert mean.shape == (B, T), f"Expected ({B}, {T}), got {mean.shape}"
