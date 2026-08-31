"""
video.py — Env-rollout and imagination-rollout video recording for DreamerV3 eval.

Two entry points:
  - record_env_video(agent, video_env, path, seed) -> starting_state
        Runs one full real-env episode with the (deterministic) actor, saves
        rendered frames as an mp4, and returns the env's initial internal state
        so an imagination video can start from the same point.

  - record_imag_video(agent, video_env, path, starting_state, horizon, env_name)
        Rolls out the world model from starting_state for `horizon` steps,
        decodes each imagined feature back to an observation, reconstructs the
        env's internal state from it, and renders. Only implemented for envs in
        IMAG_VIDEO_HOOKS (Pendulum-v1, InvertedPendulum-v4).

Frame save pattern (single approach everywhere):
    frames.append(video_env.render())
    ...
    imageio.mimsave(path, frames, fps=30)
"""

import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch

try:
    import imageio.v2 as imageio
except ImportError:                      # older imageio API
    import imageio

from dreamer.utils.math_utils import symexp


# ─────────────────────────────────────────────────────────────────────────────
# Env-specific obs↔state hooks for imagination video
# ─────────────────────────────────────────────────────────────────────────────

def _pendulum_state_to_obs(state) -> np.ndarray:
    """Pendulum-v1: state=[theta, theta_dot] → obs=[cos(theta), sin(theta), theta_dot]."""
    theta, theta_dot = float(state[0]), float(state[1])
    return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)


def _pendulum_set_state(env, obs_real: np.ndarray) -> None:
    theta = float(np.arctan2(obs_real[1], obs_real[0]))
    theta_dot = float(np.clip(obs_real[2], -8.0, 8.0))
    env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float32)


def _pendulum_capture_state(env):
    return env.unwrapped.state.copy()


def _ip_state_to_obs(state) -> np.ndarray:
    """InvertedPendulum-v4: obs is concat(qpos, qvel) directly."""
    qpos, qvel = state
    return np.concatenate([np.asarray(qpos, dtype=np.float32),
                           np.asarray(qvel, dtype=np.float32)])


def _ip_set_state(env, obs_real: np.ndarray) -> None:
    qpos = np.clip(obs_real[:2], [-1.0, -np.pi], [1.0, np.pi]).astype(np.float64)
    qvel = np.clip(obs_real[2:], -10.0, 10.0).astype(np.float64)
    env.unwrapped.set_state(qpos, qvel)


def _ip_capture_state(env):
    return (env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy())


# Registry: env_name -> (state_to_obs, set_state, capture_state)
IMAG_VIDEO_HOOKS: dict = {
    'Pendulum-v1':         (_pendulum_state_to_obs, _pendulum_set_state, _pendulum_capture_state),
    'InvertedPendulum-v4': (_ip_state_to_obs,       _ip_set_state,       _ip_capture_state),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def record_env_video(agent, video_env, path: str, seed: int, env_name: str):
    """
    Runs one deterministic-policy episode on the real env with rgb_array rendering,
    saves the frames as an mp4, and returns the env's starting internal state so an
    imagination video can begin from the same point.

    Args:
        agent:     trained DreamerV3 agent (must be built with render/eval methods).
        video_env: gym env constructed with render_mode='rgb_array'.
        path:      output mp4 path.
        seed:      seed for env.reset (also seeds initial state).
        env_name:  used to look up state-capture hook.
    Returns:
        starting_state: env's internal state right after reset(seed=seed).
                        Format depends on env (see IMAG_VIDEO_HOOKS). None if the
                        env has no imag hook.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    obs, _ = video_env.reset(seed=seed)
    capture = IMAG_VIDEO_HOOKS.get(env_name, (None, None, None))[2]
    starting_state = capture(video_env) if capture is not None else None

    frames = []
    frame = video_env.render()
    if frame is not None:
        frames.append(frame)

    state = None
    done = False
    while not done:
        action, state = agent.step(obs, state, training=False)
        obs, _, terminated, truncated, _ = video_env.step(action)
        frame = video_env.render()
        if frame is not None:
            frames.append(frame)
        done = bool(terminated or truncated)

    if frames:
        imageio.mimsave(path, frames, fps=30)
    return starting_state


def record_imag_video(agent, video_env, path: str, starting_state,
                      horizon: int, env_name: str, seed: int) -> None:
    """
    Rolls out the world model from `starting_state` for `horizon` imagined steps,
    decodes each feature back to an observation, reconstructs the env's internal
    state via IMAG_VIDEO_HOOKS[env_name], and renders each frame.

    Args:
        agent:          trained DreamerV3 agent.
        video_env:      gym env with render_mode='rgb_array'.
        path:           output mp4 path.
        starting_state: internal env state to start from (matches capture format).
        horizon:        number of imagined steps.
        env_name:       key into IMAG_VIDEO_HOOKS.
        seed:           seed for the pre-reset (real env is reset then overridden).
    """
    if env_name not in IMAG_VIDEO_HOOKS:
        print(f'[video] imagination video not implemented for {env_name}, skipping', flush=True)
        return
    if starting_state is None:
        print(f'[video] no starting_state captured for {env_name}, skipping imag video', flush=True)
        return

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    state_to_obs, set_state, _ = IMAG_VIDEO_HOOKS[env_name]

    # Reset env then override to starting_state so frame 0 matches env-rollout video.
    video_env.reset(seed=seed)
    obs0_np = state_to_obs(starting_state)          # obs vector at t=0
    set_state(video_env, obs0_np)                   # env re-set from that vector

    frames = []
    frame = video_env.render()
    if frame is not None:
        frames.append(frame)

    # Build the RSSM posterior at t=0 from obs0.
    device = agent.device
    action_dim = agent.action_dim
    with torch.no_grad():
        obs0_t = torch.tensor(obs0_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_dim)
        embed = agent.world_model.encode(obs0_t)                                          # (1, embed_dim)
        s0 = agent.world_model.rssm.initial_state(1)
        prev_action = torch.zeros(1, action_dim, device=device)
        rssm_state, _ = agent.world_model.rssm.observe(embed, prev_action, s0)
        feat0 = agent.world_model.rssm.get_state_feature(rssm_state)                     # (1, state_dim)

        # Imagined rollout under the current actor.
        rollout = agent.world_model.imagine_rollout(feat0, agent.actor, horizon)
        feats = rollout['features']                                                       # (1, H, state_dim)

        # Decode each imagined feature back to a symlog obs, then symexp to real obs.
        for t in range(horizon):
            obs_symlog = agent.world_model.decoder(feats[:, t])                           # (1, obs_dim)
            obs_real = symexp(obs_symlog).squeeze(0).cpu().numpy()
            try:
                set_state(video_env, obs_real)
                frame = video_env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                # early-training garbage decodes can drive the env into an invalid state;
                # skip that frame rather than crashing the whole video.
                print(f'[video] imag frame {t} skipped ({type(e).__name__}: {e})', flush=True)

    if frames:
        imageio.mimsave(path, frames, fps=30)
