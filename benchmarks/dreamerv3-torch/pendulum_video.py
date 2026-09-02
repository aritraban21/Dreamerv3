"""pendulum_video.py — parity videos for the reference DreamerV3 on Pendulum-v1.

Mirrors what our implementation saves each eval:
  - env_step_{N}.mp4   : one real-env eval-policy episode (true physics)
  - imag_step_{N}.mp4  : one imagined rollout from the SAME start state (world model)

The pendulum is drawn with matplotlib (Agg) directly from `theta`, so it needs no
GL/pyglet/xvfb — the SAME drawer renders both the real and imagined angle, so the two
videos are visually comparable frame-for-frame. theta=0 is upright.

Used by dreamer.py's eval loop when --render_video True, and runnable standalone
against a saved checkpoint (see benchmarks/COLAB.md).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio


def draw_pendulum(theta, label="", subtitle=""):
    """Render a single pendulum frame (uint8 HxWx3) at angle `theta` (0 = up)."""
    fig = plt.figure(figsize=(3.2, 3.4), dpi=100)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    # rod: origin -> (sin theta, cos theta); theta=0 points straight up
    x, y = np.sin(theta), np.cos(theta)
    ax.plot([0, x], [0, y], "-", lw=6, color="#1f77b4", solid_capstyle="round")
    ax.plot([0], [0], "o", ms=8, color="#333333")
    ax.plot([x], [y], "o", ms=12, color="#d62728")
    # faint "upright" reference marker
    ax.plot([0, 0], [0, 1.15], ":", lw=1, color="#bbbbbb")
    if label:
        ax.set_title(label, fontsize=11)
    if subtitle:
        ax.text(0, -1.2, subtitle, ha="center", fontsize=8, color="#666666")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frame = buf[..., :3].copy()
    plt.close(fig)
    return frame


def _theta_from_state(state_vec):
    cos_t, sin_t = float(state_vec[0]), float(state_vec[1])
    return float(np.arctan2(sin_t, cos_t))


def record_env_video(agent, env, path, seed=0, fps=30):
    """Run one deterministic (actor.mode) eval episode; render from true theta.

    Returns the start observation dict (to seed the imagination rollout).
    """
    obs = env.reset()
    start_obs = {k: np.array(v) for k, v in obs.items() if "log_" not in k}
    done_flag = np.array([True])
    state = None
    frames = []
    step = 0
    while True:
        obs_b = {k: np.stack([obs[k]]) for k in obs if "log_" not in k}
        with torch.no_grad():
            action, state = agent(obs_b, done_flag, state, training=False)
        act = {k: np.array(action[k][0].detach().cpu()) for k in action}
        theta = _theta_from_state(obs["state"])
        frames.append(draw_pendulum(theta, "real env (policy)", f"step {step}"))
        obs, reward, done, info = env.step(act)
        done_flag = np.array([done])
        step += 1
        if done:
            break
    imageio.mimsave(path, frames, fps=fps)
    return start_obs


def record_imag_video(agent, path, start_obs, horizon=200, fps=30):
    """Imagine `horizon` steps from start_obs using the world model + actor, render
    each decoded state's theta. No environment is stepped — pure model dream."""
    wm = agent._wm
    dyn = wm.dynamics
    actor = agent._task_behavior.actor
    obs_b = {k: np.stack([start_obs[k]]) for k in start_obs if "log_" not in k}
    frames = []
    with torch.no_grad():
        obs_p = wm.preprocess(obs_b)
        embed = wm.encoder(obs_p)
        latent, _ = dyn.obs_step(None, None, embed, obs_p["is_first"])
        for t in range(horizon):
            feat = dyn.get_feat(latent)
            action = actor(feat).mode()
            latent = dyn.img_step(latent, action)
            feat2 = dyn.get_feat(latent)
            state_pred = wm.heads["decoder"](feat2)["state"].mode()  # real space
            theta = _theta_from_state(state_pred[0].detach().cpu().numpy())
            frames.append(draw_pendulum(theta, "imagination (world model)", f"step {t}"))
    imageio.mimsave(path, frames, fps=fps)
