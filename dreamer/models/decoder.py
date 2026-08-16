"""
decoder.py — MLP decoder for reconstructing vector observations from model states.

The decoder is part of the world model prediction loss. It reconstructs observations
from the RSSM latent state (h_t, z_t) to ensure the representations are informative
(otherwise the model could encode nothing and still minimize the KL loss).

Reconstruction loss: MSE(decoder_output, symlog(obs_target))
Both decoder output and reconstruction target are in symlog space.

Architecture: Linear → RMSNorm → SiLU → ... → Linear (outputs in symlog space)
"""

import torch
import torch.nn as nn

from dreamer.utils.math_utils import symlog


class MLPDecoder(nn.Module):
    """
    Multi-layer perceptron decoder. Maps RSSM state features back to observation space.

    Note: the output is in symlog space. The reconstruction loss is:
        F.mse_loss(decoder(state_feature), symlog(obs_target))
    Not the raw observation — both sides are symlog-transformed.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
    ):
        """
        Args:
            state_dim: deter_dim + stoch_dim * stoch_classes
                       (concatenated h_t and flattened z_t)
            obs_dim:   raw observation dimensionality (3 or 17)
            hidden_dim: hidden layer width
            num_layers: number of hidden layers
        Notes:
            Mirrors the encoder architecture.
            Structure: Linear(state_dim→hidden_dim) → [RMSNorm→SiLU→Linear] x num_layers → Linear(hidden→obs_dim)
        """
        super().__init__()
        # build input projection: nn.Linear(state_dim, hidden_dim, bias=False)
        self.input = nn.Linear(state_dim, hidden_dim, bias=True)
        # build hidden layers as nn.ModuleList (same pattern as encoder)
        self.hidden = nn.ModuleList()
        for layer in range(num_layers):
            self.hidden.append(nn.LayerNorm(hidden_dim, eps=1e-3))
            self.hidden.append(nn.SiLU())
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        # build output projection: nn.Linear(hidden_dim, obs_dim, bias=False)
        self.output = nn.Linear(hidden_dim, obs_dim, bias=False)

    def forward(self, state_feature: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs observations (in symlog space) from latent state features.

        Args:
            state_feature: shape (B, state_dim) or (B, T, state_dim)
        Returns:
            obs_symlog_pred: shape (B, obs_dim) or (B, T, obs_dim)
                             Predictions are in symlog space.
                             Use MSE against symlog(obs_target) for the loss.
        """
        # pass through input projection
        x = self.input(state_feature)
        # loop through hidden blocks: apply norm, activation, linear
        for layer in self.hidden:
            x = layer(x)
        # pass through output projection (no final activation — output is symlog-scale)
        x = self.output(x)
        # return the prediction
        return x
