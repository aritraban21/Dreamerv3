"""
evaluate.py — Standalone evaluation script for a saved DreamerV3 checkpoint.

Usage:
    python evaluate.py --checkpoint checkpoints/pendulum.pt --config configs/pendulum.yaml
    python evaluate.py --checkpoint checkpoints/halfcheetah.pt --config configs/halfcheetah.yaml --episodes 10 --render
"""

import argparse
import numpy as np
import torch
import gymnasium as gym

from dreamer.agent import DreamerV3
from train import load_config, make_env


def evaluate_checkpoint(
    checkpoint_path: str,
    config: dict,
    num_episodes: int = 10,
    render: bool = False,
) -> None:
    """
    Loads a saved checkpoint and evaluates the agent.
    Prints per-episode returns and the mean return.

    Args:
        checkpoint_path: path to .pt checkpoint file
        config:          hyperparameter dict (same config used during training)
        num_episodes:    number of evaluation episodes
        render:          if True, render the environment to screen
    """
    # determine device from config
    # create environment:
    #   if render: env = make_env(config['env'], seed=0); wrap with gym.wrappers.HumanRendering if needed
    #   else: env = make_env(config['env'], seed=0)
    # get obs_dim and action_dim from env.observation_space and env.action_space
    # build agent: agent = DreamerV3(obs_dim, action_dim, config)
    # load checkpoint: agent.load(checkpoint_path)
    # set agent to eval mode and move to device
    # collect episodes:
    #   returns = []
    #   for ep in range(num_episodes):
    #       obs, _ = env.reset(); state = None; ep_return = 0
    #       while True:
    #           action, state = agent.step(obs, state, training=False)
    #           obs, reward, terminated, truncated, _ = env.step(action)
    #           ep_return += reward
    #           if terminated or truncated: break
    #       returns.append(ep_return)
    #       print(f"Episode {ep+1}: return={ep_return:.2f}")
    # print(f"Mean return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    raise NotImplementedError


def main():
    """
    CLI entry point.
    """
    # create ArgumentParser with:
    #   --checkpoint: required, path to .pt file
    #   --config:     required, path to yaml config
    #   --episodes:   int, default=10
    #   --render:     store_true flag
    # parse args
    # config = load_config(args.config)
    # call evaluate_checkpoint(args.checkpoint, config, args.episodes, args.render)
    raise NotImplementedError


if __name__ == "__main__":
    main()
