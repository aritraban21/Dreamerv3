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
import yaml
import numpy as np
import torch
import gymnasium as gym

from dreamer.agent import DreamerV3
from dreamer.replay_buffer import ReplayBuffer
import dreamer.envs  # noqa: F401  (side-effect: registers custom envs like PointMass1D-v0)


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


def make_env(env_name: str, seed: int = 0) -> gym.Env:
    """
    Creates a gymnasium environment with the given seed.
    Wraps the action space to [-1, 1] using RescaleAction if the action space is not already bounded.

    Args:
        env_name: gymnasium environment ID (e.g., 'Pendulum-v1', 'HalfCheetah-v4')
        seed:     random seed for reproducibility
    Returns:
        env: wrapped gymnasium environment
    Notes:
        - For environments with action bounds other than [-1, 1],
          wrap with gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
        - Call env.reset(seed=seed) immediately after creation to set the seed
    """
    # create environment: gym.make(env_name)
    env = gym.make(env_name)
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

def evaluate(
    agent: DreamerV3,
    env: gym.Env,
    num_episodes: int = 5,
) -> float:
    """
    Evaluates the agent for a fixed number of episodes using greedy (mean) actions.
    No exploration noise — action = actor.mean().

    Args:
        agent:        trained DreamerV3 agent
        env:          gymnasium environment (can be the same env as training)
        num_episodes: number of complete episodes to run
    Returns:
        mean_return: average undiscounted episodic return across episodes
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
    return float(np.mean(returns))


def train(config: dict) -> None:
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

    grad_per_env = config['train_ratio'] / (config['batch_size'] * config['batch_length'])
    print(f"[startup] env={config['env']}  device={device}  seed={seed}", flush=True)
    print(f"[startup] total_steps={config['total_steps']}  batch={config['batch_size']}x{config['batch_length']}"
          f"  train_ratio={config['train_ratio']} -> {grad_per_env:.3f} grad/env", flush=True)

    train_env = make_env(config['env'], seed=seed)
    eval_env = make_env(config['env'], seed=seed)
    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    print(f"[startup] obs_dim={obs_dim}  action_dim={action_dim}", flush=True)

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
        # periodic evaluation + checkpoint
        if global_step > 0 and global_step % config['eval_every'] == 0:
            mean_return = evaluate(agent, eval_env, config['eval_episodes'])
            print(f"[step {global_step}] eval return = {mean_return:.2f}", flush=True)
            agent.save(config['save_path'])

    # final evaluation + save
    mean_return = evaluate(agent, eval_env, config['eval_episodes'])
    print(f"[final] eval return = {mean_return:.2f}", flush=True)
    agent.save(config['save_path'])
    train_env.close()
    eval_env.close()


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
    args = parser.parse_args()

    # build overrides dict from any non-None CLI args
    overrides = {k: v for k, v in vars(args).items()
                 if k != 'config' and v is not None}

    config = load_config(args.config, overrides)

    # create save directory if needed
    save_dir = os.path.dirname(config['save_path'])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    train(config)


if __name__ == "__main__":
    main()
