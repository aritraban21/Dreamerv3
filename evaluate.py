"""
evaluate.py — Standalone evaluation script for a saved DreamerV3 checkpoint.

Usage:
    python evaluate.py --config configs/pendulum.yaml \
                       --checkpoint checkpoints/Pendulum-v1/best.pt
    python evaluate.py --config configs/pendulum.yaml \
                       --checkpoint checkpoints/Pendulum-v1/best_step_50000.pt \
                       --num-episodes 50 --seed 7 --device cpu

Prints mean/std/min/max return over `--num-episodes` deterministic-actor episodes.
Reuses `evaluate_returns` from train.py so the rollout loop stays in one place.
"""

import argparse
import numpy as np
import torch

from dreamer.agent import DreamerV3
from train import load_config, make_env, evaluate_returns


def _resolve_overrides(args):
    overrides = {}
    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.device is not None:
        overrides['device'] = args.device
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Evaluate a DreamerV3 checkpoint")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to yaml config used at training time.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to .pt checkpoint (e.g., checkpoints/Pendulum-v1/best.pt).")
    parser.add_argument('--num-episodes', type=int, default=25, dest='num_episodes',
                        help="Number of eval episodes (default 25).")
    parser.add_argument('--seed', type=int, default=None,
                        help="Seed override.")
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                        help="Device override.")
    args = parser.parse_args()

    config = load_config(args.config, _resolve_overrides(args))
    seed = int(config.get('seed', 0))
    device = torch.device(config.get('device', 'cpu'))

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(config['env'], seed=seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DreamerV3(obs_dim, action_dim, config).to(device)
    agent.load(args.checkpoint)
    agent.eval()

    print(f"[eval] env={config['env']}  checkpoint={args.checkpoint}  "
          f"num_episodes={args.num_episodes}  seed={seed}  device={device}", flush=True)

    returns = evaluate_returns(agent, env, num_episodes=args.num_episodes)
    returns_np = np.asarray(returns, dtype=np.float64)

    for i, r in enumerate(returns):
        print(f"  ep {i+1:3d}: return = {r:.2f}", flush=True)

    print(
        f"[eval] mean={returns_np.mean():.2f}  std={returns_np.std():.2f}  "
        f"min={returns_np.min():.2f}  max={returns_np.max():.2f}  n={len(returns)}",
        flush=True,
    )

    env.close()


if __name__ == "__main__":
    main()
