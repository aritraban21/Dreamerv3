"""
actor.py — Policy network (Actor) for DreamerV3.

The actor maps RSSM state features to action distributions. It is trained purely
on imagined trajectories — it never directly observes the real environment.

For continuous actions (Pendulum, HalfCheetah): outputs a TruncatedNormal distribution.
The actor is updated via REINFORCE with normalized lambda-returns:

    L_actor = -E_tau [ R_norm_t * log pi(a_t | s_t) ]  -  eta * H[pi(s_t)]

where R_norm_t = R_t / max(1, S) and S = Perc_95(R) - Perc_5(R).

NOTE: DreamerV3 Eq. 6 actually normalizes the ADVANTAGE (R_t - v(s_t)), i.e. it subtracts
the critic value baseline before dividing by max(1, S). The formula above (raw returns, no
baseline) is a higher-variance simplification. The actor-loss wiring lives in
agent.update_actor_critic() — see its "PAPER BASELINE" note.

Key design choices:
  - Output layer initialized to zero for stable early training.
  - min_std prevents the policy from collapsing to near-deterministic too early.
  - Actions are in [-1, 1] — scale to environment range in train.py if needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.utils.distributions import TruncatedNormal


class Actor(nn.Module):
    """
    Continuous action policy network.

    Outputs mean and std of a TruncatedNormal([-1, 1]) distribution.
    During training: sample from the distribution for exploration.
    During evaluation: use the mean for deterministic action selection.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        min_std: float = 0.1,
    ):
        """
        Args:
            state_dim:  deter_dim + stoch_dim * stoch_classes
            action_dim: environment action dimensionality (1 for Pendulum, 6 for HalfCheetah)
            hidden_dim: hidden layer width
            num_layers: number of hidden layers
            min_std:    minimum action standard deviation (prevents deterministic collapse)
        """
        super().__init__()
        # store min_std as attribute
        self.min_std = min_std
        # build trunk MLP: Linear(state_dim, hidden_dim) → [RMSNorm→SiLU→Linear] x num_layers

        layers = [nn.Linear(state_dim, hidden_dim, bias=False)]
        for _ in range(num_layers):
            layers.append(nn.RMSNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.mlp = nn.Sequential(*layers)
        # build mean_head: Linear(hidden_dim, action_dim) — outputs raw mean logits
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # build std_head: Linear(hidden_dim, action_dim) — outputs raw standard deviation logits
        self.std_head = nn.Linear(hidden_dim, action_dim)
        # initialize mean_head and std_head weights to zero: nn.init.zeros_(...)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.std_head.weight)
        nn.init.zeros_(self.std_head.bias)

    def forward(self, state_feature: torch.Tensor) -> TruncatedNormal:
        """
        Computes the action distribution from state features.

        Args:
            state_feature: shape (B, state_dim) or (B, T, state_dim)
        Returns:
            TruncatedNormal distribution with:
                mean: shape (..., action_dim) — tanh-squashed to (-1, 1)
                std:  shape (..., action_dim) — softplus(std) + min_std
        Notes:
            Apply torch.tanh to the mean output so it lies in (-1, 1).
            std = F.softplus(std) + self.min_std ensures std is always positive.
        """
        # pass state_feature through trunk MLP
        mlp_output = self.mlp(state_feature)
        # compute mean: torch.tanh(self.mean_head(trunk_output))
        mean = torch.tanh(self.mean_head(mlp_output))
        # compute std: F.softplus(self.std_head(trunk_output)) + self.min_std
        std = self.std_head(mlp_output)
        std = F.softplus(std) + self.min_std

        # return TruncatedNormal(mean, std, low=-1.0, high=1.0)
        return TruncatedNormal(mean, std, low=-1.0, high=1.0)

    def get_action(self, state_feature: torch.Tensor, training: bool) -> torch.Tensor:
        """
        Returns an action given a state feature.

        Args:
            state_feature: shape (B, state_dim)
            training: if True, sample from distribution; if False, return mean
        Returns:
            action: shape (B, action_dim) — values in [-1, 1]
        """
        # compute the distribution via self.forward(state_feature)
        action_dist = self.forward(state_feature)
        # if training: return dist.sample()
        if training:
            return_value = action_dist.sample()
        # else: return dist.mean()
        else:
            return_value = action_dist.mean()
        return return_value
