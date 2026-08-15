"""
Tests for dreamer/models/rssm.py

Run with: pytest tests/test_rssm.py -v

This is the most critical test file. The RSSM is the architectural core.
Test observe() and imagine() individually before testing the sequence methods.
"""

import pytest
import torch
from dreamer.models.rssm import RSSM, BlockGRU
from dreamer.models.actor import Actor


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def rssm_config():
    """Small RSSM config for fast testing."""
    return dict(
        embed_dim=64,
        action_dim=1,
        deter_dim=128,
        stoch_dim=8,
        stoch_classes=8,
        hidden_dim=64,
        unimix=0.01,
    )


@pytest.fixture
def rssm(rssm_config):
    return RSSM(**rssm_config)


@pytest.fixture
def actor(rssm_config):
    state_dim = rssm_config['deter_dim'] + rssm_config['stoch_dim'] * rssm_config['stoch_classes']
    return Actor(state_dim=state_dim, action_dim=rssm_config['action_dim'],
                 hidden_dim=64, num_layers=1)


# ─────────────────────────────────────────────
# BlockGRU tests
# ─────────────────────────────────────────────

def test_blockgru_output_shape():
    """BlockGRU returns h_next of shape (B, deter_dim)."""
    gru = BlockGRU(input_dim=16, deter_dim=32)
    x = torch.randn(4, 16)
    h = torch.zeros(4, 32)
    h_next = gru(x, h)
    assert h_next.shape == (4, 32), f"Expected (4, 32), got {h_next.shape}"


def test_blockgru_changes_state():
    """BlockGRU output should differ from the zero initial state."""
    gru = BlockGRU(input_dim=8, deter_dim=16)
    x = torch.randn(2, 8)
    h = torch.zeros(2, 16)
    h_next = gru(x, h)
    assert not torch.allclose(h_next, h), "GRU output should change from zero state"


# ─────────────────────────────────────────────
# RSSM initial_state tests
# ─────────────────────────────────────────────

def test_initial_state_shapes(rssm, rssm_config):
    """initial_state(B) returns h of (B, deter_dim) and z of (B, stoch_dim, stoch_classes)."""
    B = 4
    state = rssm.initial_state(B)
    assert 'h' in state and 'z' in state
    assert state['h'].shape == (B, rssm_config['deter_dim'])
    assert state['z'].shape == (B, rssm_config['stoch_dim'], rssm_config['stoch_classes'])


def test_initial_state_zeros(rssm):
    """initial_state should start at zero."""
    state = rssm.initial_state(2)
    assert torch.all(state['h'] == 0), "Initial h must be zero"
    assert torch.all(state['z'] == 0), "Initial z must be zero"


# ─────────────────────────────────────────────
# RSSM.observe tests
# ─────────────────────────────────────────────

def test_observe_output_shapes(rssm, rssm_config):
    """observe() returns state and dists with correct shapes."""
    B = 4
    embed = torch.randn(B, rssm_config['embed_dim'])
    action = torch.randn(B, rssm_config['action_dim'])
    prev_state = rssm.initial_state(B)
    state, dists = rssm.observe(embed, action, prev_state)

    assert state['h'].shape == (B, rssm_config['deter_dim'])
    assert state['z'].shape == (B, rssm_config['stoch_dim'], rssm_config['stoch_classes'])
    assert 'posterior_logits' in dists and 'prior_logits' in dists
    assert dists['posterior_logits'].shape == (B, rssm_config['stoch_dim'], rssm_config['stoch_classes'])
    assert dists['prior_logits'].shape == (B, rssm_config['stoch_dim'], rssm_config['stoch_classes'])


def test_observe_output_finite(rssm, rssm_config):
    """observe() outputs must be finite."""
    B = 4
    embed = torch.randn(B, rssm_config['embed_dim'])
    action = torch.randn(B, rssm_config['action_dim'])
    prev_state = rssm.initial_state(B)
    state, dists = rssm.observe(embed, action, prev_state)
    assert torch.isfinite(state['h']).all()
    assert torch.isfinite(dists['posterior_logits']).all()
    assert torch.isfinite(dists['prior_logits']).all()


def test_observe_multiple_steps(rssm, rssm_config):
    """Running observe() multiple times should work without errors."""
    B = 2
    prev_state = rssm.initial_state(B)
    for _ in range(5):
        embed = torch.randn(B, rssm_config['embed_dim'])
        action = torch.randn(B, rssm_config['action_dim'])
        prev_state, _ = rssm.observe(embed, action, prev_state)
    assert prev_state['h'].shape == (B, rssm_config['deter_dim'])


# ─────────────────────────────────────────────
# RSSM.imagine tests
# ─────────────────────────────────────────────

def test_imagine_output_shapes(rssm, rssm_config):
    """imagine() returns state and dists with correct shapes."""
    B = 4
    action = torch.randn(B, rssm_config['action_dim'])
    prev_state = rssm.initial_state(B)
    state, dists = rssm.imagine(action, prev_state)
    assert state['h'].shape == (B, rssm_config['deter_dim'])
    assert state['z'].shape == (B, rssm_config['stoch_dim'], rssm_config['stoch_classes'])
    assert 'prior_logits' in dists
    assert 'posterior_logits' not in dists, \
        "imagine() should not return posterior_logits (no observation available)"


def test_imagine_has_gradients(rssm, rssm_config):
    """imagine() output tensors must retain gradient connections (for actor training)."""
    B = 2
    action = torch.randn(B, rssm_config['action_dim'], requires_grad=True)
    prev_state = rssm.initial_state(B)
    state, _ = rssm.imagine(action, prev_state)
    feature = rssm.get_state_feature(state)
    loss = feature.sum()
    loss.backward()
    assert action.grad is not None, "Gradient must flow through imagine() to action"


# ─────────────────────────────────────────────
# RSSM.get_state_feature tests
# ─────────────────────────────────────────────

def test_get_state_feature_shape(rssm, rssm_config):
    """get_state_feature returns (B, deter_dim + stoch_dim * stoch_classes)."""
    B = 4
    state = rssm.initial_state(B)
    feature = rssm.get_state_feature(state)
    expected_dim = rssm_config['deter_dim'] + rssm_config['stoch_dim'] * rssm_config['stoch_classes']
    assert feature.shape == (B, expected_dim), \
        f"Expected ({B}, {expected_dim}), got {feature.shape}"


def test_get_state_feature_is_cat(rssm, rssm_config):
    """Feature should be a concatenation of h and flattened z."""
    B = 2
    state = rssm.initial_state(B)
    # manually set h and z to known values
    state['h'] = torch.ones(B, rssm_config['deter_dim'])
    state['z'] = torch.zeros(B, rssm_config['stoch_dim'], rssm_config['stoch_classes'])
    feature = rssm.get_state_feature(state)
    # first deter_dim elements should be 1
    assert torch.allclose(feature[:, :rssm_config['deter_dim']], torch.ones(B, rssm_config['deter_dim']))
    # remaining elements should be 0
    assert torch.allclose(feature[:, rssm_config['deter_dim']:], torch.zeros(B, rssm_config['stoch_dim'] * rssm_config['stoch_classes']))


# ─────────────────────────────────────────────
# RSSM.observe_sequence tests
# ─────────────────────────────────────────────

def test_observe_sequence_shapes(rssm, rssm_config):
    """observe_sequence returns features (B, T, state_dim) and logits (B, T, stoch, classes)."""
    B, T = 4, 8
    embeds = torch.randn(B, T, rssm_config['embed_dim'])
    actions = torch.randn(B, T, rssm_config['action_dim'])
    is_first = torch.zeros(B, T, dtype=torch.bool)
    initial_state = rssm.initial_state(B)

    features, posteriors, priors = rssm.observe_sequence(embeds, actions, is_first, initial_state)

    state_dim = rssm_config['deter_dim'] + rssm_config['stoch_dim'] * rssm_config['stoch_classes']
    assert features.shape == (B, T, state_dim)
    assert posteriors['posterior_logits'].shape == (B, T, rssm_config['stoch_dim'], rssm_config['stoch_classes'])
    assert priors['prior_logits'].shape == (B, T, rssm_config['stoch_dim'], rssm_config['stoch_classes'])


def test_observe_sequence_is_first_resets(rssm, rssm_config):
    """
    When is_first=True at step t, the state should be reset to zero at that step.
    Check: run a non-trivial sequence, then compare with is_first=True at t=5.
    The state at t=5 (and beyond) should differ from the case where is_first=False.
    """
    B, T = 2, 10
    embeds = torch.randn(B, T, rssm_config['embed_dim'])
    actions = torch.randn(B, T, rssm_config['action_dim'])
    initial_state = rssm.initial_state(B)

    # Run without any resets
    is_first_no_reset = torch.zeros(B, T, dtype=torch.bool)
    features_no_reset, _, _ = rssm.observe_sequence(embeds, actions, is_first_no_reset, initial_state)

    # Run with is_first=True at step 5 for batch element 0
    is_first_with_reset = torch.zeros(B, T, dtype=torch.bool)
    is_first_with_reset[0, 5] = True
    features_with_reset, _, _ = rssm.observe_sequence(embeds, actions, is_first_with_reset, initial_state)

    # Features at step 5 for batch element 0 should differ (state was reset)
    assert not torch.allclose(
        features_no_reset[0, 5], features_with_reset[0, 5]
    ), "is_first reset should change the feature at that step"


def test_observe_sequence_finite(rssm, rssm_config):
    """observe_sequence outputs must be finite for random inputs."""
    B, T = 4, 16
    embeds = torch.randn(B, T, rssm_config['embed_dim'])
    actions = torch.randn(B, T, rssm_config['action_dim'])
    is_first = torch.zeros(B, T, dtype=torch.bool)
    initial_state = rssm.initial_state(B)
    features, posteriors, priors = rssm.observe_sequence(embeds, actions, is_first, initial_state)
    assert torch.isfinite(features).all(), "features from observe_sequence has non-finite values"


# ─────────────────────────────────────────────
# RSSM.imagine_sequence tests
# ─────────────────────────────────────────────

def test_imagine_sequence_shapes(rssm, rssm_config, actor):
    """imagine_sequence returns features (B, H, state_dim) and actions (B, H, action_dim)."""
    B, H = 8, 16
    initial_state = rssm.initial_state(B)
    rollout = rssm.imagine_sequence(initial_state, actor, horizon=H)

    state_dim = rssm_config['deter_dim'] + rssm_config['stoch_dim'] * rssm_config['stoch_classes']
    assert rollout['features'].shape == (B, H, state_dim)
    assert rollout['actions'].shape == (B, H, rssm_config['action_dim'])


def test_imagine_sequence_finite(rssm, rssm_config, actor):
    """imagine_sequence outputs must be finite."""
    B, H = 4, 8
    initial_state = rssm.initial_state(B)
    rollout = rssm.imagine_sequence(initial_state, actor, horizon=H)
    assert torch.isfinite(rollout['features']).all()
    assert torch.isfinite(rollout['actions']).all()


def test_imagine_sequence_gradient_to_actor(rssm, rssm_config, actor):
    """Gradients must flow from a loss on features back to actor parameters."""
    B, H = 4, 4
    initial_state = rssm.initial_state(B)
    rollout = rssm.imagine_sequence(initial_state, actor, horizon=H)
    loss = rollout['features'].sum() + rollout['actions'].sum()
    loss.backward()
    actor_grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert len(actor_grads) > 0, \
        "No gradients flowed to actor from imagine_sequence. Check straight_through_sample."


# ─────────────────────────────────────────────
# KL sanity check
# ─────────────────────────────────────────────

def test_posterior_prior_kl_nonneg(rssm, rssm_config):
    """KL(posterior || prior) should be non-negative."""
    from dreamer.utils.distributions import UnimixCategorical
    B, T = 2, 4
    embeds = torch.randn(B, T, rssm_config['embed_dim'])
    actions = torch.randn(B, T, rssm_config['action_dim'])
    is_first = torch.zeros(B, T, dtype=torch.bool)
    initial_state = rssm.initial_state(B)
    _, posteriors, priors = rssm.observe_sequence(embeds, actions, is_first, initial_state)

    post = UnimixCategorical(posteriors['posterior_logits'])
    prior = UnimixCategorical(priors['prior_logits'])
    kl = post.kl_divergence(prior)
    assert (kl >= -1e-5).all(), f"KL must be non-negative. Min: {kl.min().item()}"


# ─────────────────────────────────────────────
# Additional coverage
# ─────────────────────────────────────────────

def test_observe_sequence_is_first_no_cross_contamination(rssm, rssm_config):
    """
    Resetting one batch element via is_first must NOT change other batch elements'
    features. Guards against a bug where the mask accidentally resets the whole
    batch instead of only the flagged elements.
    """
    B, T = 4, 8
    embeds = torch.randn(B, T, rssm_config['embed_dim'])
    actions = torch.randn(B, T, rssm_config['action_dim'])
    initial_state = rssm.initial_state(B)

    is_first_none = torch.zeros(B, T, dtype=torch.bool)
    torch.manual_seed(0)
    features_baseline, _, _ = rssm.observe_sequence(embeds, actions, is_first_none, initial_state)

    is_first_one = torch.zeros(B, T, dtype=torch.bool)
    is_first_one[2, 4] = True
    torch.manual_seed(0)
    features_reset, _, _ = rssm.observe_sequence(embeds, actions, is_first_one, initial_state)

    for b in [0, 1, 3]:
        assert torch.allclose(features_baseline[b], features_reset[b]), \
            f"Batch element {b} changed despite not being flagged in is_first"
    assert not torch.allclose(features_baseline[2, 4], features_reset[2, 4]), \
        "Batch element 2 at step 4 should differ (was reset)"


def test_observe_imagine_prior_agree(rssm, rssm_config):
    """
    observe() and imagine() must produce identical prior_logits when called from
    the same prev_state and action. Both compute prior_net(gru(x, prev_h)) with
    the same inputs — if they diverge, one branch has drifted from the other.
    """
    B = 4
    embed = torch.randn(B, rssm_config['embed_dim'])
    action = torch.randn(B, rssm_config['action_dim'])
    prev_state = rssm.initial_state(B)

    _, obs_dists = rssm.observe(embed, action, prev_state)
    _, img_dists = rssm.imagine(action, prev_state)

    assert torch.allclose(obs_dists['prior_logits'], img_dists['prior_logits']), \
        "prior_logits from observe() and imagine() must match for identical inputs"


def test_observe_sequence_gradient_to_embed(rssm, rssm_config):
    """
    Gradients from a loss on features must reach the embed inputs — this is the
    training path for the encoder. Mirrors test_imagine_sequence_gradient_to_actor
    but on the observe (world-model training) side.
    """
    B, T = 2, 4
    embeds = torch.randn(B, T, rssm_config['embed_dim'], requires_grad=True)
    actions = torch.randn(B, T, rssm_config['action_dim'])
    is_first = torch.zeros(B, T, dtype=torch.bool)
    initial_state = rssm.initial_state(B)

    features, _, _ = rssm.observe_sequence(embeds, actions, is_first, initial_state)
    loss = features.sum()
    loss.backward()

    assert embeds.grad is not None, \
        "No gradient flowed from features back to embeds. Encoder cannot train."
    assert torch.isfinite(embeds.grad).all(), "Gradient to embeds contains non-finite values"


def test_observe_z_is_one_hot(rssm, rssm_config):
    """After observe(), z should be a valid one-hot per stoch_dim (sums to 1)."""
    B = 4
    embed = torch.randn(B, rssm_config['embed_dim'])
    action = torch.randn(B, rssm_config['action_dim'])
    prev_state = rssm.initial_state(B)
    state, _ = rssm.observe(embed, action, prev_state)
    z_sums = state['z'].sum(dim=-1)
    assert torch.allclose(z_sums, torch.ones_like(z_sums), atol=1e-5), \
        f"z is not one-hot per stoch_dim. Sums: {z_sums}"


def test_imagine_sequence_state_evolves(rssm, rssm_config, actor):
    """
    Over an imagined horizon, consecutive state features should not be identical.
    Guards against a bug where state fails to propagate through the imagine loop.
    """
    B, H = 2, 6
    initial_state = rssm.initial_state(B)
    rollout = rssm.imagine_sequence(initial_state, actor, horizon=H)
    features = rollout['features']
    for t in range(H - 1):
        assert not torch.allclose(features[:, t], features[:, t + 1]), \
            f"Features at t={t} and t={t+1} are identical — state is not evolving"


def test_actor_get_action_eval_returns_tensor(rssm, rssm_config, actor):
    """
    actor.get_action(feat, training=False) must return a tensor of shape
    (B, action_dim), not a bound-method object. Guards against forgetting the
    parentheses on dist.mean() (which is a method, not a property).
    """
    B = 4
    state = rssm.initial_state(B)
    feature = rssm.get_state_feature(state)
    action = actor.get_action(feature, training=False)
    assert isinstance(action, torch.Tensor), \
        f"get_action(training=False) must return a Tensor, got {type(action)}"
    assert action.shape == (B, rssm_config['action_dim']), \
        f"Expected shape ({B}, {rssm_config['action_dim']}), got {action.shape}"
    assert torch.isfinite(action).all(), "Actions must be finite"
    assert (action >= -1.0).all() and (action <= 1.0).all(), \
        "Actions must lie in [-1, 1]"
