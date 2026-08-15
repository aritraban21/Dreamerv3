"""
predictors.py — Reward and continue predictors for DreamerV3.

Both predictors take RSSM state features as input:
    RewardPredictor:   predicts reward distribution (TwoHot over 255 bins)
    ContinuePredictor: predicts episode continuation (Bernoulli)

Both use zero-initialized output layers to prevent large initial predictions
from destabilizing early training.
"""

import torch
import torch.nn as nn

from dreamer.utils.distributions import TwoHotDist
from dreamer.utils.math_utils import make_symexp_bins


class RewardPredictor(nn.Module):
    """
    Predicts the distribution over future rewards given a model state.

    Uses TwoHot categorical distribution over symexp-spaced bins instead of
    scalar regression. This decouples gradient magnitude from reward scale —
    crucial for environments with rewards ranging from -1 (Pendulum) to +10000 (Atari).

    Loss: -reward_pred(state_feature).log_prob(symlog(reward)).mean()
    Note: reward targets must be symlog-transformed before computing log_prob.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        num_bins: int = 255,
        bin_range: float = 20.0,
    ):
        """
        Args:
            state_dim:  deter_dim + stoch_dim * stoch_classes
            hidden_dim: hidden layer width
            num_layers: hidden layer depth
            num_bins:   number of TwoHot bins (255 in paper)
            bin_range:  symlog-space extent of bins (20.0 in paper)
        """
        super().__init__()
        # build MLP: Linear(state_dim, hidden_dim) → [RMSNorm→SiLU→Linear(hidden, hidden)] x num_layers → Linear(hidden, num_bins)
        layers = [nn.Linear(state_dim, hidden_dim, bias=False)]
        for _ in range(num_layers):
            layers.append(nn.RMSNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        layers.append(nn.Linear(hidden_dim, num_bins))
        self.mlp = nn.Sequential(*layers)
        # register bins as a buffer (not a parameter): self.register_buffer('bins', make_symexp_bins(num_bins, bin_range))
        self.register_buffer('bins', make_symexp_bins(num_bins, bin_range))
        # initialize the output layer weights to zero: nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, state_feature: torch.Tensor) -> TwoHotDist:
        """
        Args:
            state_feature: shape (B, state_dim) or (B, T, state_dim)
        Returns:
            TwoHotDist with logits shape (..., num_bins) and self.bins

        Usage:
            dist = reward_predictor(feature)
            loss = -dist.log_prob(symlog(reward)).mean()
            pred = dist.mean()   # in original reward scale (symexp applied internally)
        """
        # pass state_feature through the MLP layers to get logits of shape (..., num_bins)
        logits = self.mlp(state_feature)
        # return TwoHotDist(logits, self.bins)
        return TwoHotDist(logits, self.bins)


class ContinuePredictor(nn.Module):
    """
    Predicts whether the episode will continue at the next timestep.
    Binary classification: output is a Bernoulli distribution.

    Loss: -cont_pred(state_feature).log_prob(continues).mean()
    where continues = 1.0 for non-terminal steps, 0.0 at episode end.

    During imagination: use .probs (not .sample()) for soft continuation weights.
    This allows gradient flow through the continue mask.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
    ):
        """
        Args:
            state_dim:  deter_dim + stoch_dim * stoch_classes
            hidden_dim: hidden layer width
            num_layers: hidden layer depth
        """
        super().__init__()
        # build MLP: Linear(state_dim, hidden_dim) → [RMSNorm→SiLU→Linear] x num_layers → Linear(hidden, 1)
        layers = [nn.Linear(state_dim, hidden_dim, bias=False)]
        for _ in range(num_layers):
            layers.append(nn.RMSNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state_feature: torch.Tensor) -> torch.distributions.Bernoulli:
        """
        Args:
            state_feature: shape (B, state_dim) or (B, T, state_dim)
        Returns:
            Bernoulli distribution with logits; shape (...,)

        Usage:
            dist = cont_pred(feature)
            loss = -dist.log_prob(continues).mean()         # training loss
            soft_cont = dist.probs                          # for imagination masking
        """
        # pass state_feature through MLP to get a single logit: shape (..., 1)
        logits = self.mlp(state_feature)
        # squeeze the last dim to get shape (...,)
        logits = logits.squeeze(-1)
        # return torch.distributions.Bernoulli(logits=logits)
        return torch.distributions.Bernoulli(logits=logits)
