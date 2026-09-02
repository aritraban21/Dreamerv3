"""
train.py — Main training script for DreamerV3.

Entry point for both Part 1 (Pendulum-v1) and Part 2 (HalfCheetah-v4).

Usage:
    python train.py --config configs/pendulum.yaml          # Part 1
    python train.py --config configs/halfcheetah.yaml       # Part 2
    python train.py --config configs/pendulum.yaml --seed 42

Training loop overview (per environment step):
    1. step():  encode obs → run RSSM posterior → sample action from actor
    2. env.step(action): collect (obs, reward, done)
    3. replay_buffer.add(): store transition
    4. train_step(): sample batch → update world model → imagine → update actor/critic
    5. evaluate(): periodically run greedy rollouts and log return

See README.md for expected learning curves and common debugging tips.
"""

import argparse
import os
from datetime import datetime
import yaml
import numpy as np
import torch
import gymnasium as gym

from dreamer.agent import DreamerV3
from dreamer.replay_buffer import ReplayBuffer
from dreamer.utils.logging import tee_stdout
import dreamer.envs  # noqa: F401  (side-effect: registers custom envs like PointMass1D-v0)


def _wandb_init(config: dict, run_ts: str):
    """Starts a WandB run if config['wandb'] is truthy. Returns the wandb module or None.

    Requires WANDB_API_KEY to be set in the environment (never in code/config) —
    wandb.init() reads it automatically. We check for it up front rather than letting
    wandb.init() discover it's missing, because wandb falls back to an interactive
    login prompt (which hangs forever with no stdin, e.g. under a training script).
    Any other init failure (bad key, network) is also caught so it can't kill training.
    """
    if not config.get('wandb', False):
        return None
    if not os.environ.get('WANDB_API_KEY'):
        print("[wandb] --wandb passed but WANDB_API_KEY is not set in the environment. "
              "Continuing without wandb.", flush=True)
        return None
    import wandb
    try:
        wandb.init(
            project=config.get('wandb_project', 'dreamerv3'),
            entity=config.get('wandb_entity') or None,
            name=f"{config['env']}_{run_ts}",
            group=config['env'],
            config=config,
        )
    except Exception as e:
        print(f"[wandb] failed to start run ({e}). Is WANDB_API_KEY set? Continuing without wandb.",
              flush=True)
        return None
    return wandb


def load_config(config_path: str, overrides: dict = None) -> dict:
    """
    Loads a yaml config file, then merges in any CLI overrides.

    Args:
        config_path: path to the yaml config file (e.g., 'configs/pendulum.yaml')
        overrides:   dict of key→value pairs to override config values (from CLI args)
    Returns:
        config: dict with all hyperparameters
    """
    # load default config first: open('configs/default.yaml') and yaml.safe_load
    with open('configs/default.yaml', 'r') as f:
        default_config = yaml.safe_load(f)
    # load the specified config: open(config_path) and yaml.safe_load
    with open(config_path, 'r') as f:
        env_config = yaml.safe_load(f)
    # merge: default.update(env_config) — env config overrides defaults
    default_config.update(env_config)
    # if overrides: config.update(overrides) — CLI args override everything
    if overrides:
        default_config.update(overrides)
    # return merged config dict
    return default_config


def make_env(env_name: str, seed: int = 0, render_mode: str = None) -> gym.Env:
    """
    Creates a gymnasium environment with the given seed.
    Wraps the action space to [-1, 1] using RescaleAction if the action space is not already bounded.

    Args:
        env_name:    gymnasium environment ID (e.g., 'Pendulum-v1', 'HalfCheetah-v4')
        seed:        random seed for reproducibility
        render_mode: passed through to gym.make (e.g., 'rgb_array' for video recording)
    Returns:
        env: wrapped gymnasium environment
    """
    env = gym.make(env_name, render_mode=render_mode) if render_mode else gym.make(env_name)
    # wrap with RescaleAction if action space is continuous (Box) and not already [-1, 1]:
    #   env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    if isinstance(env.action_space, gym.spaces.Box):
        low = env.action_space.low
        high = env.action_space.high
        if not (np.allclose(low, -1.0) and np.allclose(high, 1.0)):
            env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    # return the environment
    return env


def collect_random_episodes(
    env: gym.Env,
    replay_buffer: ReplayBuffer,
    num_steps: int = 1000,
) -> None:
    """
    Pre-fills the replay buffer with random-policy transitions before training starts.
    This ensures the world model sees diverse data from the first gradient step.

    Args:
        env:           the gymnasium environment
        replay_buffer: buffer to fill
        num_steps:     number of random steps to collect
    Notes:
        - Use env.action_space.sample() for random actions.
        - Track is_first (True at episode starts) for proper RSSM resets.
        - Reset environment at episode termination and truncation.
    """
    # reset the environment; is_first = True for the first step
    obs, _ = env.reset()
    is_first = True
    # loop for num_steps:
    for _ in range(num_steps):
        # sample a random action
        action = env.action_space.sample()
        # step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # buffer stores TERMINATED (not done). Truncation is not termination — the
        # episode could have continued. This distinction matters for cont_pred:
        # continues = 1 - terminated. Matches reference (obs['cont'] = 1 - is_terminal).
        replay_buffer.add(obs, action, float(reward), terminated, is_first)
        obs = next_obs
        is_first = False
        if done:
            obs, _ = env.reset()
            is_first = True

def evaluate_returns(
    agent: DreamerV3,
    env: gym.Env,
    num_episodes: int = 5,
) -> list:
    """
    Runs num_episodes greedy episodes and returns the list of per-episode returns.
    Shared rollout loop between train-time eval and standalone evaluate.py.
    """
    agent.eval()
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = None
        episode_return = 0.0
        while True:
            action, state = agent.step(obs, state, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        returns.append(episode_return)
    agent.train()
    return returns


def evaluate(
    agent: DreamerV3,
    env: gym.Env,
    num_episodes: int = 5,
) -> float:
    """Mean-return wrapper around evaluate_returns() for train-time logging."""
    return float(np.mean(evaluate_returns(agent, env, num_episodes)))


def train(config: dict, run_ts: str = None) -> None:
    """
    Main DreamerV3 training loop.

    Pseudocode:
        env = make_env(config['env'], config['seed'])
        agent = DreamerV3(obs_dim, action_dim, config)
        replay_buffer = ReplayBuffer(...)
        collect_random_episodes(env, replay_buffer, config['prefill_steps'])

        # train_ratio uses the DreamerV3 paper convention: number of REPLAY
        # observations seen per ENV observation. Grad steps per env step is
        # therefore train_ratio / (batch_size * batch_length). E.g. train_ratio=512
        # with B*T=1024 → 0.5 grad steps per env step (1 update every 2 env steps).
        # We accumulate a fractional "debt" and drain it whenever it reaches 1.0.
        grad_per_env = config['train_ratio'] / (config['batch_size'] * config['batch_length'])
        train_debt = 0.0

        for global_step in range(config['total_steps']):
            action, state = agent.step(obs, state, training=True)
            step env; store in replay_buffer; handle episode resets
            if len(replay_buffer) >= config['batch_size'] * config['batch_length']:
                train_debt += grad_per_env
                while train_debt >= 1.0:
                    metrics = agent.train_step(replay_buffer)
                    train_debt -= 1.0
            if global_step % eval_every == 0:
                mean_return = evaluate(agent, eval_env, config['eval_episodes'])
                print and log metrics
                agent.save(config['save_path'])

    Args:
        config: dict loaded and merged from yaml configs
    """
    seed = config.get('seed', 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(config.get('device', 'cpu'))

    if run_ts is None:
        run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    wandb = _wandb_init(config, run_ts)
    verbose = bool(config.get('verbose', False))

    grad_per_env = config['train_ratio'] / (config['batch_size'] * config['batch_length'])
    print(f"[startup] env={config['env']}  device={device}  seed={seed}", flush=True)
    print(f"[startup] total_steps={config['total_steps']}  batch={config['batch_size']}x{config['batch_length']}"
          f"  train_ratio={config['train_ratio']} -> {grad_per_env:.3f} grad/env", flush=True)

    train_env = make_env(config['env'], seed=seed)
    eval_env = make_env(config['env'], seed=seed)
    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    print(f"[startup] obs_dim={obs_dim}  action_dim={action_dim}", flush=True)

    # Optional video env (rgb_array render). Only built when recording is on.
    record_video = bool(config.get('record_video', False))
    video_env = None
    video_dir = None
    if record_video:
        from dreamer.utils.video import record_env_video, record_imag_video, IMAG_VIDEO_HOOKS  # noqa: F401
        video_env = make_env(config['env'], seed=seed, render_mode='rgb_array')
        video_dir = config.get('video_dir', 'videos/{env}').format(env=config['env'])
        os.makedirs(video_dir, exist_ok=True)
        print(f"[startup] video recording ON -> {video_dir}", flush=True)

    # Per-env checkpoint dir.
    ckpt_dir = config.get('save_dir', 'checkpoints/{env}').format(env=config['env'])
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[startup] checkpoints -> {ckpt_dir}", flush=True)

    agent = DreamerV3(obs_dim, action_dim, config).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[startup] agent params: {n_params/1e6:.2f}M", flush=True)

    replay_buffer = ReplayBuffer(capacity=config['replay_capacity'],
                                 obs_dim=obs_dim, action_dim=action_dim,
                                 sequence_length=config['batch_length'])

    print(f"[prefill] collecting {config['prefill_steps']} random steps...", flush=True)
    collect_random_episodes(env=train_env, replay_buffer=replay_buffer,
                            num_steps=config['prefill_steps'])
    print(f"[prefill] done. buffer size = {len(replay_buffer)}", flush=True)

    # LP-01: pretrain phase — N WM+AC update steps on prefill data before env interaction.
    pretrain_steps = config.get('pretrain', 0)
    if pretrain_steps > 0 and len(replay_buffer) >= config['batch_size'] * config['batch_length']:
        print(f"[pretrain] running {pretrain_steps} update steps on prefill data...", flush=True)
        for _pt in range(pretrain_steps):
            metrics = agent.train_step(replay_buffer)
            if _pt % max(1, pretrain_steps // 10) == 0:
                print(f"[pretrain {_pt}/{pretrain_steps}] wm={metrics.get('wm_loss', float('nan')):.3f} "
                      f"actor={metrics.get('actor_loss', float('nan')):.3f} "
                      f"critic={metrics.get('critic_loss', float('nan')):.3f}", flush=True)
        print("[pretrain] done.", flush=True)

    obs, _ = train_env.reset(seed=seed)
    is_first = True
    state = None
    train_debt = 0.0
    best_eval = -float('inf')
    print("[train] entering main loop.", flush=True)
    # main loop over global_step in range(config['total_steps']):
    for global_step in range(config['total_steps']):
        #     collect one environment step:
        #         action, state = agent.step(obs, state, training=True)
        action, state = agent.step(obs, state, training=True)
        #         next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated
        # Store TERMINATED (not done). Truncation is not termination.
        replay_buffer.add(obs, action, float(reward), terminated, is_first)
        #         obs = next_obs; is_first = False
        obs = next_obs
        is_first = False
        #         if done: obs, _ = env.reset(); state = None; is_first = True
        if done:
            obs, _ = train_env.reset()
            state = None
            is_first = True
        #     training updates (paper convention: train_ratio = replay obs per env obs):
        #         if buffer has enough data: accumulate train_debt += grad_per_env, then
        #         drain via while train_debt >= 1.0: agent.train_step(...); train_debt -= 1
        if len(replay_buffer) >= config['batch_size'] * config['batch_length']:

            train_debt += grad_per_env
            while train_debt >= 1.0:
                metrics = agent.train_step(replay_buffer)
                train_debt -= 1.0
        # periodic training progress log
        if global_step % config['log_every'] == 0 and 'metrics' in locals():
            print(f"[step {global_step}] wm_loss={metrics.get('wm_loss', float('nan')):.3f}  "
                  f"actor_loss={metrics.get('actor_loss', float('nan')):.3f}  "
                  f"critic_loss={metrics.get('critic_loss', float('nan')):.3f}", flush=True)
            if verbose:
                print(f"[step {global_step}] wm/recon={metrics.get('wm/recon_loss', float('nan')):.4f} "
                      f"wm/reward={metrics.get('wm/reward_loss', float('nan')):.4f} "
                      f"wm/cont={metrics.get('wm/cont_loss', float('nan')):.4f} "
                      f"wm/kl_dyn={metrics.get('wm/kl_dyn', float('nan')):.4f} "
                      f"wm/kl_rep={metrics.get('wm/kl_rep', float('nan')):.4f} "
                      f"wm/kl_dyn_raw={metrics.get('wm/kl_dyn_raw', float('nan')):.4f} "
                      f"wm/kl_rep_raw={metrics.get('wm/kl_rep_raw', float('nan')):.4f} "
                      f"wm_grad={metrics.get('wm_grad_norm', float('nan')):.2f} "
                      f"actor_grad={metrics.get('actor_grad_norm', float('nan')):.2f} "
                      f"critic_grad={metrics.get('critic_grad_norm', float('nan')):.2f} "
                      f"policy_entropy={metrics.get('policy_entropy', float('nan')):.3f} "
                      f"mean_return={metrics.get('mean_return', float('nan')):.2f} "
                      f"return_scale={metrics.get('return_scale', float('nan')):.2f} "
                      f"buffer={len(replay_buffer)} train_debt={train_debt:.2f}", flush=True)
                print(f"[step {global_step}] value_mean={metrics.get('value_mean', float('nan')):.2f} "
                      f"value_std={metrics.get('value_std', float('nan')):.2f} "
                      f"target_p05={metrics.get('target_p05', float('nan')):.2f} "
                      f"target_p95={metrics.get('target_p95', float('nan')):.2f} "
                      f"ret_ema_range_raw={metrics.get('ret_ema_range_raw', float('nan')):.2f} "
                      f"adv_abs_mean={metrics.get('adv_abs_mean', float('nan')):.4f} "
                      f"imag_reward_mean={metrics.get('imag_reward_mean', float('nan')):.3f} "
                      f"imag_reward_min={metrics.get('imag_reward_min', float('nan')):.3f} "
                      f"action_abs_mean={metrics.get('action_abs_mean', float('nan')):.3f}", flush=True)
            if wandb is not None:
                wandb.log(metrics, step=global_step)
        # periodic evaluation + checkpoint
        if global_step > 0 and global_step % config['eval_every'] == 0:
            mean_return = evaluate(agent, eval_env, config['eval_episodes'])
            print(f"[step {global_step}] eval return = {mean_return:.2f} "
                  f"(episodes={config['eval_episodes']})", flush=True)
            if wandb is not None:
                wandb.log({'eval/mean_return': mean_return, 'eval/best_eval': max(best_eval, mean_return)},
                          step=global_step)
            # Always save 'latest' for crash-resume; save 'best' + numbered history on new best.
            agent.save(os.path.join(ckpt_dir, 'latest.pt'))
            if mean_return > best_eval:
                best_eval = mean_return
                agent.save(os.path.join(ckpt_dir, 'best.pt'))
                agent.save(os.path.join(ckpt_dir, f'best_step_{global_step}.pt'))
                print(f"[ckpt] new best {mean_return:.2f} -> best.pt + best_step_{global_step}.pt", flush=True)
            # Optional videos: one real-env rollout, one imagination rollout.
            if record_video:
                from dreamer.utils.video import record_env_video, record_imag_video, IMAG_VIDEO_HOOKS
                env_path = os.path.join(video_dir, f'env_step_{global_step}.mp4')
                imag_path = os.path.join(video_dir, f'imag_step_{global_step}.mp4')
                start_state = record_env_video(agent, video_env, env_path,
                                               seed=seed + global_step, env_name=config['env'])
                print(f"[video] wrote {env_path}", flush=True)
                if config['env'] in IMAG_VIDEO_HOOKS and start_state is not None:
                    record_imag_video(agent, video_env, imag_path, start_state,
                                      horizon=config.get('imag_video_horizon', 200),
                                      env_name=config['env'], seed=seed + global_step)
                    print(f"[video] wrote {imag_path}", flush=True)
                else:
                    print(f"[video] imagination video not implemented for {config['env']}, skipping",
                          flush=True)

    # final evaluation + save (same three-way save as periodic)
    mean_return = evaluate(agent, eval_env, config['eval_episodes'])
    print(f"[final] eval return = {mean_return:.2f}", flush=True)
    agent.save(os.path.join(ckpt_dir, 'latest.pt'))
    if mean_return > best_eval:
        best_eval = mean_return
        agent.save(os.path.join(ckpt_dir, 'best.pt'))
        agent.save(os.path.join(ckpt_dir, f'best_step_final.pt'))
        print(f"[ckpt] final is best {mean_return:.2f} -> best.pt + best_step_final.pt", flush=True)
    print(f"[final] best eval seen = {best_eval:.2f}", flush=True)
    train_env.close()
    eval_env.close()
    if video_env is not None:
        video_env.close()
    if wandb is not None:
        wandb.finish()


def main():
    """
    CLI entry point. Parses arguments, loads config, and starts training.

    Usage:
        python train.py --config configs/pendulum.yaml
        python train.py --config configs/halfcheetah.yaml --seed 42 --device cuda
    """
    parser = argparse.ArgumentParser(description="DreamerV3 training script")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to yaml config file (e.g., configs/pendulum.yaml)")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed override (overrides config['seed'])")
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                        help="Device override (overrides config['device'])")
    parser.add_argument('--record-video', action='store_true', dest='record_video',
                        help="At every eval, record one real-env episode and one imagination episode (mp4).")
    parser.add_argument('--total-steps', type=int, default=None, dest='total_steps',
                        help="Override total_steps (useful for smokes).")
    parser.add_argument('--eval-every', type=int, default=None, dest='eval_every',
                        help="Override eval_every (useful for smokes).")
    parser.add_argument('--eval-episodes', type=int, default=None, dest='eval_episodes',
                        help="Override eval_episodes.")
    parser.add_argument('--wandb', action='store_true',
                        help="Log metrics to Weights & Biases. Requires WANDB_API_KEY env var to be set.")
    parser.add_argument('--verbose', action='store_true',
                        help="Print extra per-step diagnostics (wm sub-losses, grad norms, policy entropy, etc.).")
    args = parser.parse_args()

    # build overrides dict from any non-None / non-False CLI args
    overrides = {}
    for k, v in vars(args).items():
        if k == 'config':
            continue
        if v is None:
            continue
        if k in ('record_video', 'wandb', 'verbose') and not v:
            continue
        overrides[k] = v

    config = load_config(args.config, overrides)

    # Tee stdout/stderr to a timestamped log file.
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = config.get('log_dir', 'logs')
    log_path = os.path.join(log_dir, f"{config['env']}_{ts}.log")
    log_file = tee_stdout(log_path)
    print(f"[log] tee to {log_path}", flush=True)

    try:
        train(config, run_ts=ts)
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
