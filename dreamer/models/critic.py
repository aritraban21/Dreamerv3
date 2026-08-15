"""
critic.py — Value function (Critic) for DreamerV3.

The critic predicts the distribution of discounted lambda-returns from RSSM state features.
Like the reward predictor, it uses a TwoHot categorical distribution over symexp bins.

Two networks are maintained:
  - self.net:     the trainable critic (updated by gradient descent).
                  Also computes bootstrap values for lambda_return().
  - self.ema_net: the slow EMA copy (polyak averaging, no gradients).
                  Used ONLY as a regularization target — never for bootstrap.

Loss (DreamerV3 paper, p.6 & Table 4):
    L_critic = -E[critic(feature).log_prob(symlog(returns))].mean()
             + ema_reg * -E[critic(feat).log_prob(symlog(critic_ema(feat).mean().detach()))].mean()

Weight ema_reg = 1.0 (Table 4: "Critic EMA regularizer — 1"). This second term
is a distillation loss: the fast critic's distribution is trained toward the slow
critic's predicted mean (as a target), using the same cross-entropy form as the
main loss. This matches the official danijar/dreamerv3 code
(`slowreg * value.loss(sg(slowvalue.pred()))`) and stabilizes training without
introducing a bootstrap lag. This matches both the paper and
the official danijar/dreamerv3 code (`slowtar: False` by default in imag_loss
and repl_loss — bootstrap uses the current critic, EMA only regularizes).

NOT implemented here (scope cut): the critic REPLAY loss (beta_repval = 0.3,
Table 4), which applies the critic loss to replayed trajectories in addition to
imagined ones. `critic_repval_scale` in the config is currently unused as a result.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.utils.distributions import TwoHotDist
from dreamer.utils.math_utils import make_symexp_bins


class Critic(nn.Module):
    """
    Value function network with EMA target network.

    Predicts the distribution over lambda-returns using TwoHot categorical.
    The EMA copy (ema_net) provides a slow-moving regularization target for
    the fast critic. Bootstrap values in lambda_return() come from the fast
    critic itself (see agent.compute_lambda_returns()).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        num_bins: int = 255,
        bin_range: float = 20.0,
        ema_decay: float = 0.98,
    ):
        """
        Args:
            state_dim:  deter_dim + stoch_dim * stoch_classes
            hidden_dim: hidden layer width
            num_layers: number of hidden layers
            num_bins:   bins for TwoHot distribution (255 in paper)
            bin_range:  symlog-space range of bins (20.0 in paper)
            ema_decay:  polyak averaging decay for EMA target (0.98 in paper)
        """
        super().__init__()
        # store ema_decay as attribute
        self.ema_decay = ema_decay
        # build self.net as an MLP: Linear(state_dim, hidden_dim) → [RMSNorm→SiLU→Linear] x num_layers → Linear(hidden, num_bins)
        layers = [nn.Linear(state_dim, hidden_dim, bias=False)]
        for _ in range(num_layers):
            layers.append(nn.RMSNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        layers.append(nn.Linear(hidden_dim, num_bins))
        self.net = nn.Sequential(*layers)
        # initialize the output linear layer weights to zero: nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # build self.ema_net as a deep copy of self.net: copy.deepcopy(self.net)
        self.ema_net = copy.deepcopy(self.net)
        # set all ema_net parameters to not require gradients: p.requires_grad_(False) for all p
        for p in self.ema_net.parameters():
            p.requires_grad_(False)
        # register bins as a buffer: self.register_buffer('bins', make_symexp_bins(num_bins, bin_range))
        self.register_buffer('bins', make_symexp_bins(num_bins, bin_range))

    def forward(self, state_feature: torch.Tensor) -> TwoHotDist:
        """
        Computes value distribution from the trainable critic network.

        Args:
            state_feature: shape (B, state_dim) or (B, T, state_dim)
        Returns:
            TwoHotDist with logits shape (..., num_bins)

        Usage:
            dist = critic(feature)
            loss = -dist.log_prob(symlog(returns)).mean()
            value = dist.mean()   # in original (non-symlog) scale
        """
        # pass state_feature through self.net to get logits
        logits = self.net(state_feature)
        # return TwoHotDist(logits, self.bins)
        return TwoHotDist(logits, self.bins)

    @torch.no_grad()
    def forward_ema(self, state_feature: torch.Tensor) -> TwoHotDist:
        """
        Computes value distribution from the EMA copy (no gradients).
        Used ONLY as the regularization target — the fast critic's distribution
        is trained toward symlog(this.mean().detach()) via cross-entropy
        (distillation; see agent.update_actor_critic()).

        Do NOT use this for lambda-return bootstrap values. Per the DreamerV3
        paper (p.6) and the official code (`slowtar: False`), bootstrap uses
        the current critic (self.forward), not the EMA.

        Args:
            state_feature: shape (B, state_dim) or (B, T, state_dim)
        Returns:
            TwoHotDist (detached — no gradients)
        """
        # pass state_feature through self.ema_net to get logits
        logits = self.ema_net(state_feature)
        # return TwoHotDist(logits, self.bins)
        return TwoHotDist(logits, self.bins)

    @torch.no_grad()
    def update_ema(self) -> None:
        """
        Updates EMA network weights using polyak averaging.
        Call this AFTER each critic optimizer step.

        Formula: ema_param = decay * ema_param + (1 - decay) * param
        """
        # loop over zip(self.net.parameters(), self.ema_net.parameters())
        for param, ema_param in zip(self.net.parameters(), self.ema_net.parameters()):
            ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * param.data