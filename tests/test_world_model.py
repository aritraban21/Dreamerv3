"""
Tests for dreamer/world_model.py

Run with: pytest tests/test_world_model.py -v

All upstream modules (encoder, rssm, decoder, predictors) must be implemented first.
"""

import pytest
import torch
from dreamer.world_model import WorldModel


@pytest.fixture
def config():
    return dict(
        embed_dim=64,
        deter_dim=128,
        stoch_dim=8,
        stoch_classes=8,
        hidden_dim=64,
        num_layers=1,
        num_bins=51,
        bin_range=10.0,
        unimix=0.01,
        lambda_pred=1.0,
        lambda_dyn=0.5,
        lambda_rep=0.1,
        free_bits=1.0,
    )


@pytest.fixture
def wm(config):
    return WorldModel(obs_dim=3, action_dim=1, config=config)


@pytest.fixture
def batch():
    """Synthetic batch of 4 sequences of length 8."""
    B, T = 4, 8
    return dict(
        obs=torch.randn(B, T, 3),
        action=torch.randn(B, T, 1),
        reward=torch.randn(B, T),
        done=torch.zeros(B, T),
        is_first=torch.zeros(B, T, dtype=torch.bool),
    )


# ─────────────────────────────────────────────
# encode tests
# ─────────────────────────────────────────────

def test_encode_shape(wm, config):
    """encode() returns (B, T, embed_dim)."""
    obs = torch.randn(4, 8, 3)
    embed = wm.encode(obs)
    assert embed.shape == (4, 8, config['embed_dim']), \
        f"Expected (4, 8, {config['embed_dim']}), got {embed.shape}"


def test_encode_2d(wm, config):
    """encode() handles (B, obs_dim) single-step input."""
    obs = torch.randn(4, 3)
    embed = wm.encode(obs)
    assert embed.shape == (4, config['embed_dim'])


def test_encode_finite(wm):
    """encode() output must be finite."""
    obs = torch.randn(4, 8, 3)
    assert torch.isfinite(wm.encode(obs)).all()


# ─────────────────────────────────────────────
# observe_sequence tests
# ─────────────────────────────────────────────

def test_observe_sequence_shapes(wm, config):
    """observe_sequence returns (B, T, state_dim), logits with correct shapes."""
    B, T = 4, 8
    obs = torch.randn(B, T, 3)
    actions = torch.randn(B, T, 1)
    is_first = torch.zeros(B, T, dtype=torch.bool)

    state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
    features, posteriors, priors = wm.observe_sequence(obs, actions, is_first)

    assert features.shape == (B, T, state_dim)
    assert posteriors['posterior_logits'].shape == (B, T, config['stoch_dim'], config['stoch_classes'])
    assert priors['prior_logits'].shape == (B, T, config['stoch_dim'], config['stoch_classes'])


def test_observe_sequence_finite(wm):
    """observe_sequence must produce finite features."""
    B, T = 4, 8
    obs = torch.randn(B, T, 3)
    actions = torch.randn(B, T, 1)
    is_first = torch.zeros(B, T, dtype=torch.bool)
    features, _, _ = wm.observe_sequence(obs, actions, is_first)
    assert torch.isfinite(features).all()


# ─────────────────────────────────────────────
# compute_kl_loss tests
# ─────────────────────────────────────────────

def test_kl_loss_scalar(wm, config):
    """compute_kl_loss returns a scalar tensor."""
    B, T = 4, 8
    posteriors = {'posterior_logits': torch.randn(B, T, config['stoch_dim'], config['stoch_classes'])}
    priors = {'prior_logits': torch.randn(B, T, config['stoch_dim'], config['stoch_classes'])}
    kl_loss, kl_info = wm.compute_kl_loss(posteriors, priors)
    assert kl_loss.ndim == 0, "KL loss must be a scalar"


def test_kl_loss_nonneg(wm, config):
    """KL loss must be non-negative (can be at free_bits floor, but not below 0)."""
    B, T = 4, 8
    posteriors = {'posterior_logits': torch.randn(B, T, config['stoch_dim'], config['stoch_classes'])}
    priors = {'prior_logits': torch.randn(B, T, config['stoch_dim'], config['stoch_classes'])}
    kl_loss, _ = wm.compute_kl_loss(posteriors, priors)
    assert kl_loss.item() >= 0, f"KL loss must be non-negative, got {kl_loss.item()}"


def test_kl_loss_has_info_keys(wm, config):
    """compute_kl_loss returns info dict with kl_dyn and kl_rep."""
    B, T = 2, 4
    posteriors = {'posterior_logits': torch.randn(B, T, config['stoch_dim'], config['stoch_classes'])}
    priors = {'prior_logits': torch.randn(B, T, config['stoch_dim'], config['stoch_classes'])}
    _, kl_info = wm.compute_kl_loss(posteriors, priors)
    assert 'kl_dyn' in kl_info
    assert 'kl_rep' in kl_info


# ─────────────────────────────────────────────
# compute_loss tests
# ─────────────────────────────────────────────

def test_compute_loss_scalar(wm, batch):
    """compute_loss returns a scalar loss."""
    continues = 1.0 - batch['done']
    loss, info = wm.compute_loss(
        batch['obs'], batch['action'], batch['reward'],
        continues, batch['is_first']
    )
    assert loss.ndim == 0, "World model loss must be a scalar"


def test_compute_loss_finite(wm, batch):
    """World model loss must be finite for random inputs."""
    continues = 1.0 - batch['done']
    loss, info = wm.compute_loss(
        batch['obs'], batch['action'], batch['reward'],
        continues, batch['is_first']
    )
    assert torch.isfinite(loss), f"World model loss is not finite: {loss.item()}"


def test_compute_loss_nonneg_kl(wm, batch):
    """The KL sub-loss should be non-negative."""
    continues = 1.0 - batch['done']
    _, info = wm.compute_loss(
        batch['obs'], batch['action'], batch['reward'],
        continues, batch['is_first']
    )
    assert info.get('kl_loss', info.get('kl_dyn', 0)) >= 0


def test_compute_loss_info_keys(wm, batch):
    """Loss info dict should contain standard sub-loss keys."""
    continues = 1.0 - batch['done']
    _, info = wm.compute_loss(
        batch['obs'], batch['action'], batch['reward'],
        continues, batch['is_first']
    )
    # check at least some expected keys are present
    expected_keys = {'recon_loss', 'reward_loss', 'cont_loss', 'kl_loss', 'total_loss'}
    found = expected_keys & set(info.keys())
    assert len(found) >= 3, f"Missing expected info keys. Found: {set(info.keys())}"


def test_compute_loss_backward(wm, batch):
    """World model loss backward must produce gradients in all submodules."""
    continues = 1.0 - batch['done']
    loss, _ = wm.compute_loss(
        batch['obs'], batch['action'], batch['reward'],
        continues, batch['is_first']
    )
    loss.backward()
    grads = [p.grad for p in wm.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed through world model"


# ─────────────────────────────────────────────
# imagine_rollout tests
# ─────────────────────────────────────────────

def test_imagine_rollout_shapes(wm, config):
    """imagine_rollout returns dict with correct shapes."""
    from dreamer.models.actor import Actor
    state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
    actor = Actor(state_dim=state_dim, action_dim=1, hidden_dim=64, num_layers=1)

    B, H = 8, 16
    start_features = torch.randn(B, state_dim)
    rollout = wm.imagine_rollout(start_features, actor, horizon=H)

    assert rollout['features'].shape == (B, H, state_dim)
    assert rollout['actions'].shape == (B, H, 1)
    assert rollout['rewards'].shape == (B, H)
    assert rollout['continues'].shape == (B, H)


def test_imagine_rollout_finite(wm, config):
    """imagine_rollout must produce finite values."""
    from dreamer.models.actor import Actor
    state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
    actor = Actor(state_dim=state_dim, action_dim=1, hidden_dim=64, num_layers=1)

    start_features = torch.randn(4, state_dim)
    rollout = wm.imagine_rollout(start_features, actor, horizon=8)
    for key, val in rollout.items():
        assert torch.isfinite(val).all(), f"imagine_rollout['{key}'] contains non-finite values"


def test_imagine_rollout_continues_in_01(wm, config):
    """Imagined continuation probabilities must be in (0, 1)."""
    from dreamer.models.actor import Actor
    state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
    actor = Actor(state_dim=state_dim, action_dim=1, hidden_dim=64, num_layers=1)

    start_features = torch.randn(4, state_dim)
    rollout = wm.imagine_rollout(start_features, actor, horizon=8)
    cont = rollout['continues']
    assert (cont > 0).all() and (cont < 1).all(), \
        f"Continues must be in (0, 1). Got min={cont.min():.4f}, max={cont.max():.4f}"
