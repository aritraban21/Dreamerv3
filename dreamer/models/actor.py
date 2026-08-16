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


class Actor(nn.Module):
    """
    Continuous action policy network.

    Outputs an unbounded Normal(tanh(mean), std) distribution — matches the
    default `dist='normal'` in NM512/dreamerv3-torch. tanh bounds the location
    of the distribution to [-1, 1]; samples themselves are unbounded (Normal,
    not TruncatedNormal) so log_prob remains numerically stable when the raw
    mean_head output grows during training. The env clips any out-of-range
    actions when they're taken.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        min_std: float = 0.1,
        max_std: float = 1.0,
    ):
        """
        Args:
            state_dim:  deter_dim + stoch_dim * stoch_classes
            action_dim: environment action dimensionality (1 for Pendulum, 6 for HalfCheetah)
            hidden_dim: hidden layer width
            num_layers: number of hidden layers
            min_std:    minimum action standard deviation (prevents deterministic collapse)
            max_std:    maximum action standard deviation (prevents unbounded exploration)
        """
        super().__init__()
        # store min_std / max_std as attributes
        self.min_std = min_std
        self.max_std = max_std
        # build trunk MLP: Linear(state_dim, hidden_dim) → [RMSNorm→SiLU→Linear] x num_layers

        layers = [nn.Linear(state_dim, hidden_dim, bias=True)]
        for _ in range(num_layers):
            layers.append(nn.LayerNorm(hidden_dim, eps=1e-3))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.mlp = nn.Sequential(*layers)
        # build mean_head / std_head
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)
        # A-04: non-zero small uniform init on output heads (ref: uniform_weight_init(outscale=1.0)).
        # Ref formula: limit = sqrt(1.0 * 3.0 / fan_in). Zero-init made canon_actor_mean start at 0 →
        # state-blind actor early. Uniform init gives a data-dependent starting mapping.
        import math
        limit_mean = math.sqrt(3.0 / hidden_dim)
        nn.init.uniform_(self.mean_head.weight, -limit_mean, limit_mean)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.uniform_(self.std_head.weight, -limit_mean, limit_mean)
        nn.init.zeros_(self.std_head.bias)

    def forward(self, state_feature: torch.Tensor) -> torch.distributions.Independent:
        """
        Computes the action distribution from state features.

        Args:
            state_feature: shape (B, state_dim) or (B, T, state_dim)
        Returns:
            Independent(Normal(tanh(mean), softplus(std)+min_std), 1)
            log_prob reduces over the last (action) dim.
        """
        mlp_output = self.mlp(state_feature)
        # bound mean to (-1, 1) via tanh (matches reference dist='normal')
        mean = torch.tanh(self.mean_head(mlp_output))
        # bound std to [min_std, max_std] via sigmoid (matches reference formula)
        raw_std = self.std_head(mlp_output)
        std = (self.max_std - self.min_std) * torch.sigmoid(raw_std + 2.0) + self.min_std
        base = torch.distributions.Normal(mean, std)
        # Independent(., 1) reinterprets last dim as event dim so log_prob sums it out
        return torch.distributions.Independent(base, 1)

    def get_action(self, state_feature: torch.Tensor, training: bool) -> torch.Tensor:
        """
        Returns an action given a state feature.

        Args:
            state_feature: shape (B, state_dim)
            training: if True, sample from distribution; if False, return mean
        Returns:
            action: shape (B, action_dim). May slightly exceed [-1, 1] — env clips.
        """
        action_dist = self.forward(state_feature)
        if training:
            # rsample uses reparameterization so gradients flow to actor params.
            sample = action_dist.rsample()
        else:
            sample = action_dist.mean
        # A-02: absmax=1.0 straight-through clip (ref ContDist.sample). Keeps imagined actions
        # numerically bounded even though the underlying Normal is unbounded.
        absmax = 1.0
        clipped = sample * (absmax / torch.clamp(torch.abs(sample), min=absmax)).detach()
        return clipped
