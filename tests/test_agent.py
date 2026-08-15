"""
Tests for dreamer/agent.py

Run with: pytest tests/test_agent.py -v

These are integration-level tests — they exercise the full agent pipeline.
All upstream modules must be implemented before running these.
These tests are slower than unit tests; use -k to run subsets during development.
"""

import pytest
import os
import numpy as np
import torch
from dreamer.agent import DreamerV3
from dreamer.replay_buffer import ReplayBuffer


@pytest.fixture
def config():
    return dict(
        # model
        embed_dim=64,
        deter_dim=128,
        stoch_dim=8,
        stoch_classes=8,
        hidden_dim=64,
        num_layers=1,
        num_bins=51,
        bin_range=10.0,
        unimix=0.01,
        # losses
        lambda_pred=1.0,
        lambda_dyn=0.5,
        lambda_rep=0.1,
        free_bits=1.0,
        # actor-critic
        discount=0.99,
        lambda_return=0.95,
        imagination_horizon=4,
        actor_entropy=3e-4,
        actor_min_std=0.1,
        critic_ema_decay=0.98,
        critic_repval_scale=0.3,
        critic_slowreg=1.0,
        # normalizer
        ret_norm_decay=0.99,
        ret_norm_lower_pct=5.0,
        ret_norm_upper_pct=95.0,
        ret_norm_min=1.0,
        # training
        lr=1e-4,
        grad_clip=100.0,
        batch_size=4,
        batch_length=8,
        device='cpu',
    )


@pytest.fixture
def obs_dim():
    return 3  # Pendulum-like


@pytest.fixture
def action_dim():
    return 1


@pytest.fixture
def agent(config, obs_dim, action_dim):
    return DreamerV3(obs_dim=obs_dim, action_dim=action_dim, config=config)


@pytest.fixture
def filled_buffer(obs_dim, action_dim):
    """Replay buffer with enough data for one train_step."""
    buf = ReplayBuffer(capacity=500, obs_dim=obs_dim, action_dim=action_dim, sequence_length=8)
    obs = np.zeros(obs_dim, dtype=np.float32)
    action = np.zeros(action_dim, dtype=np.float32)
    for i in range(200):
        is_first = (i % 20 == 0)
        done = (i % 20 == 19)
        buf.add(obs + np.random.randn(obs_dim).astype(np.float32),
                np.random.randn(action_dim).astype(np.float32),
                float(np.random.randn()),
                done, is_first)
    return buf


# ─────────────────────────────────────────────
# step() tests
# ─────────────────────────────────────────────

def test_step_returns_action_and_state(agent, obs_dim, action_dim):
    """step() returns (action array, state dict) with correct shapes."""
    obs = np.zeros(obs_dim, dtype=np.float32)
    action, state = agent.step(obs, state=None, training=True)
    assert isinstance(action, np.ndarray), "Action must be a numpy array"
    assert action.shape == (action_dim,), f"Action shape wrong: {action.shape}"
    assert isinstance(state, dict), "State must be a dict"
    assert 'h' in state and 'z' in state


def test_step_action_in_bounds(agent, obs_dim):
    """Actions from step() should be in [-1, 1]."""
    obs = np.zeros(obs_dim, dtype=np.float32)
    for _ in range(10):
        action, state = agent.step(obs, state=None, training=True)
        assert (action >= -1.0 - 1e-5).all() and (action <= 1.0 + 1e-5).all(), \
            f"Action out of bounds: {action}"


def test_step_stateful(agent, obs_dim):
    """State updates between consecutive steps."""
    obs = np.zeros(obs_dim, dtype=np.float32)
    action, state1 = agent.step(obs, state=None, training=False)
    action, state2 = agent.step(obs, state=state1, training=False)
    # h should have changed between steps
    assert not torch.allclose(state1['h'], state2['h']), \
        "RSSM state h should change between consecutive steps"


def test_step_eval_mode_deterministic(agent, obs_dim):
    """In eval mode (training=False), action should be deterministic (mean)."""
    obs = np.zeros(obs_dim, dtype=np.float32)
    actions = []
    for _ in range(5):
        action, _ = agent.step(obs, state=None, training=False)
        actions.append(action.copy())
    # All actions should be identical (mean action, no sampling)
    for i in range(1, len(actions)):
        assert np.allclose(actions[0], actions[i], atol=1e-5), \
            "Eval mode should return deterministic mean actions"


# ─────────────────────────────────────────────
# compute_lambda_returns tests
# ─────────────────────────────────────────────

def test_compute_lambda_returns_shape(agent, config):
    """compute_lambda_returns returns (B, H)."""
    B, H = 8, config['imagination_horizon']
    state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
    rollout = {
        'features': torch.randn(B, H, state_dim),
        'rewards': torch.randn(B, H),
        'continues': torch.rand(B, H),
        'actions': torch.randn(B, H, config.get('action_dim', 1)),
    }
    with torch.no_grad():
        returns = agent.compute_lambda_returns(rollout)
    assert returns.shape == (B, H), f"Expected ({B}, {H}), got {returns.shape}"


def test_compute_lambda_returns_finite(agent, config):
    """compute_lambda_returns must produce finite values."""
    B, H = 4, config['imagination_horizon']
    state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
    rollout = {
        'features': torch.randn(B, H, state_dim),
        'rewards': torch.randn(B, H),
        'continues': torch.ones(B, H),
        'actions': torch.randn(B, H, 1),
    }
    with torch.no_grad():
        returns = agent.compute_lambda_returns(rollout)
    assert torch.isfinite(returns).all(), "Lambda returns contain non-finite values"


# ─────────────────────────────────────────────
# train_step tests
# ─────────────────────────────────────────────

def test_train_step_returns_dict(agent, filled_buffer):
    """train_step() returns a dict of metrics."""
    metrics = agent.train_step(filled_buffer)
    assert isinstance(metrics, dict), "train_step must return a dict"


def test_train_step_contains_losses(agent, filled_buffer):
    """train_step metrics should include world model and actor-critic losses."""
    metrics = agent.train_step(filled_buffer)
    # check for some expected keys
    expected = {'wm_loss', 'actor_loss', 'critic_loss'}
    found = expected & set(metrics.keys())
    assert len(found) >= 2, \
        f"Expected loss keys not found. Got: {set(metrics.keys())}"


def test_train_step_metrics_finite(agent, filled_buffer):
    """All metrics from train_step should be finite floats."""
    metrics = agent.train_step(filled_buffer)
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            assert np.isfinite(val), f"Metric '{key}' is not finite: {val}"


def test_train_step_updates_weights(agent, filled_buffer):
    """train_step should change model weights (verify learning is happening).

    Note: actor trunk starts with zero-init output heads (standard DreamerV3),
    so trunk grads are zero on step 1 — only the output heads update. Check
    mean_head.weight, which receives non-zero gradient from the actor loss.
    """
    initial_head = agent.actor.mean_head.weight.clone().detach()
    agent.train_step(filled_buffer)
    final_head = agent.actor.mean_head.weight.detach()
    assert not torch.allclose(initial_head, final_head), \
        "Actor mean_head weights should change after train_step"


# ─────────────────────────────────────────────
# save/load tests
# ─────────────────────────────────────────────

def test_save_and_load(agent, filled_buffer, tmp_path, config, obs_dim, action_dim):
    """Agent saves and loads correctly — predictions should be identical."""
    # do one training step to get non-trivial weights
    agent.train_step(filled_buffer)

    # save
    ckpt_path = str(tmp_path / "test_agent.pt")
    agent.save(ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint file was not created"

    # load into a fresh agent and compare
    agent2 = DreamerV3(obs_dim=obs_dim, action_dim=action_dim, config=config)
    agent2.load(ckpt_path)

    # Seed RNG before each step so both agents draw the same RSSM z sample —
    # otherwise stochastic z sampling inside RSSM.observe produces different
    # features → different mean actions, even with identical weights.
    obs = np.zeros(obs_dim, dtype=np.float32)
    torch.manual_seed(42)
    with torch.no_grad():
        action1, _ = agent.step(obs, state=None, training=False)
    torch.manual_seed(42)
    with torch.no_grad():
        action2, _ = agent2.step(obs, state=None, training=False)
    assert np.allclose(action1, action2, atol=1e-5), \
        f"Loaded agent produces different actions. {action1} vs {action2}"


# ─────────────────────────────────────────────
# step() state-contract tests (new agent-state {'h', 'z', 'action'})
# ─────────────────────────────────────────────

def test_step_state_contains_action(agent, obs_dim, action_dim):
    """step() must carry the last sampled action in the returned state dict."""
    obs = np.zeros(obs_dim, dtype=np.float32)
    _, state = agent.step(obs, state=None, training=True)
    assert 'action' in state, "step() must carry last action in state"
    assert state['action'].shape == (1, action_dim), \
        f"Expected action shape (1, {action_dim}), got {state['action'].shape}"


def test_step_prev_action_affects_state(agent, obs_dim):
    """Different prev_actions in the state dict must produce different next h."""
    obs = np.zeros(obs_dim, dtype=np.float32)
    _, s1 = agent.step(obs, state=None, training=False)
    s1_alt = {**s1, 'action': torch.ones_like(s1['action'])}
    _, s2a = agent.step(obs, state=s1, training=False)
    _, s2b = agent.step(obs, state=s1_alt, training=False)
    assert not torch.allclose(s2a['h'], s2b['h']), \
        "Different prev_action must produce different next h (prev_action reaches RSSM)"


# ─────────────────────────────────────────────
# gradient-isolation tests
# ─────────────────────────────────────────────

def test_actor_critic_does_not_update_world_model(agent, filled_buffer):
    """update_actor_critic() must NOT modify world model weights.

    features are required to be .detach()'d before AC losses; if not, gradients
    leak back into the world model and it silently double-trains.
    """
    batch = filled_buffer.sample(agent.config['batch_size'])
    _, features = agent.world_model.update(
        batch, agent.wm_optimizer, agent.device, agent.config['grad_clip'])
    start_feats = features.detach().reshape(-1, agent.state_dim)
    if start_feats.shape[0] > agent.config['batch_size']:
        start_feats = start_feats[:agent.config['batch_size']]
    rollout = agent.world_model.imagine_rollout(
        start_feats, agent.actor, agent.config['imagination_horizon'])

    wm_snap = {n: p.clone() for n, p in agent.world_model.named_parameters()}
    agent.update_actor_critic(rollout)
    for n, p in agent.world_model.named_parameters():
        assert torch.allclose(wm_snap[n], p), \
            f"WM param '{n}' changed during actor-critic update — features not detached?"


def test_train_step_updates_ema(agent, filled_buffer):
    """train_step must invoke critic.update_ema() so the EMA copy drifts."""
    ema_snap = [p.clone() for p in agent.critic.ema_net.parameters()]
    agent.train_step(filled_buffer)
    ema_after = list(agent.critic.ema_net.parameters())
    changed = any(not torch.allclose(a, b) for a, b in zip(ema_snap, ema_after))
    assert changed, "Critic EMA network should drift after train_step (update_ema called)"


# ─────────────────────────────────────────────
# update_actor_critic contract tests
# ─────────────────────────────────────────────

def test_update_actor_critic_return_keys(agent, filled_buffer):
    """update_actor_critic returns all documented keys, each finite."""
    batch = filled_buffer.sample(agent.config['batch_size'])
    _, features = agent.world_model.update(
        batch, agent.wm_optimizer, agent.device, agent.config['grad_clip'])
    start_feats = features.detach().reshape(-1, agent.state_dim)
    if start_feats.shape[0] > agent.config['batch_size']:
        start_feats = start_feats[:agent.config['batch_size']]
    rollout = agent.world_model.imagine_rollout(
        start_feats, agent.actor, agent.config['imagination_horizon'])
    info = agent.update_actor_critic(rollout)
    for k in ('actor_loss', 'critic_loss', 'mean_return', 'return_scale'):
        assert k in info, f"missing key: {k}"
        assert np.isfinite(info[k]), f"non-finite value for {k}: {info[k]}"


# ─────────────────────────────────────────────
# compute_lambda_returns contract tests
# ─────────────────────────────────────────────

def test_lambda_returns_do_not_leak_grad_to_critic(agent, config):
    """Bootstrap values must be detached — otherwise critic learns to shrink
    its own targets (self-referential collapse). Rewards/continues legitimately
    carry grad (for actor learning); only the CRITIC path must be severed.
    """
    B, H = 4, config['imagination_horizon']
    state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
    rollout = {
        'features': torch.randn(B, H, state_dim, requires_grad=True),
        'rewards': torch.randn(B, H, requires_grad=True),
        'continues': torch.ones(B, H, requires_grad=True),
        'actions': torch.randn(B, H, 1),
    }
    # zero any stale gradient on critic params
    for p in agent.critic.parameters():
        p.grad = None
    returns = agent.compute_lambda_returns(rollout)
    if returns.requires_grad:
        returns.sum().backward()
    for name, p in agent.critic.named_parameters():
        assert p.grad is None or p.grad.abs().max().item() == 0.0, \
            f"Critic param '{name}' received gradient from compute_lambda_returns — bootstrap not detached"


def test_lambda_returns_respects_termination(agent, config):
    """Zero continues (episode terminated) should truncate future returns."""
    B, H = 2, config['imagination_horizon']
    state_dim = config['deter_dim'] + config['stoch_dim'] * config['stoch_classes']
    features = torch.randn(B, H, state_dim)
    rewards = torch.ones(B, H) * 5.0
    actions = torch.randn(B, H, 1)
    # row 0: terminates after step 0; row 1: never terminates
    continues = torch.ones(B, H)
    continues[0, 1:] = 0.0
    rollout = {
        'features': features, 'rewards': rewards,
        'continues': continues, 'actions': actions,
    }
    with torch.no_grad():
        returns = agent.compute_lambda_returns(rollout)
    assert returns[0].sum() < returns[1].sum(), \
        "Terminated trajectory should have smaller total return than non-terminated one"


# ─────────────────────────────────────────────
# long-run stability
# ─────────────────────────────────────────────

def test_train_step_stability(agent, filled_buffer):
    """Multiple train_steps in a row must not produce NaN/Inf anywhere."""
    for _ in range(5):
        info = agent.train_step(filled_buffer)
        assert np.isfinite(info['wm_loss']), f"non-finite wm_loss: {info['wm_loss']}"
    for name, p in agent.named_parameters():
        assert torch.isfinite(p).all(), f"NaN/Inf in parameter '{name}' after 5 train_steps"


def test_step_after_train_step(agent, filled_buffer, obs_dim):
    """step() must still return valid outputs after training has occurred."""
    agent.train_step(filled_buffer)
    action, state = agent.step(np.zeros(obs_dim, dtype=np.float32), state=None, training=True)
    assert action.shape == (agent.action_dim,), f"Bad action shape: {action.shape}"
    assert np.isfinite(action).all(), "Action contains non-finite values"
    assert torch.isfinite(state['h']).all(), "state['h'] contains non-finite values"
    assert torch.isfinite(state['z']).all(), "state['z'] contains non-finite values"
