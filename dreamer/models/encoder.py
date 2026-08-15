"""
encoder.py — MLP encoder for vector (state-based) observations.

Maps raw environment observations to a fixed-size embedding vector for the RSSM.
Applies symlog to inputs first to compress large observation values.

For Pendulum-v1:  obs_dim=3  → embed_dim=256  (small config)
For HalfCheetah:  obs_dim=17 → embed_dim=1024 (full config)

Architecture: symlog → Linear → RMSNorm → SiLU → ... → Linear → embedding
"""

import torch
import torch.nn as nn

from dreamer.utils.math_utils import symlog


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder for vector observations.

    Input observations are symlog-transformed before the MLP to prevent large
    observation values (e.g., velocities, positions) from causing gradient issues.
    The output embedding is NOT activated — it is passed directly to the RSSM
    posterior network which computes z_t from cat(h_t, embedding).
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
    ):
        """
        Args:
            obs_dim:    dimensionality of raw observation (3 for Pendulum, 17 for HalfCheetah)
            embed_dim:  output embedding size (256 for Pendulum config, 1024 for full config)
            hidden_dim: hidden layer width (same as embed_dim typically)
            num_layers: number of hidden layers between input projection and output projection
        Notes:
            Use RMSNorm (torch.nn.RMSNorm) + SiLU activations as per the DreamerV3 paper.
            Structure: Linear(obs_dim→hidden_dim) → [RMSNorm→SiLU→Linear(hidden→hidden)] x num_layers → Linear(hidden→embed_dim)
        """
        super().__init__()
        # build input projection: nn.Linear(obs_dim, hidden_dim, bias=False)
        self.input = nn.Linear(obs_dim, hidden_dim, bias=False)
        self.hidden = nn.ModuleList()
        # build hidden layers as nn.ModuleList:
        for layer in range(num_layers):
            self.hidden.append(nn.RMSNorm(hidden_dim))
            self.hidden.append(nn.SiLU())
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        
        #   each block is: nn.RMSNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # build output projection: nn.Linear(hidden_dim, embed_dim, bias=False)
        self.output = nn.Linear(hidden_dim, embed_dim, bias=False)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encodes observations into embeddings.

        Args:
            obs: shape (B, obs_dim) or (B, T, obs_dim) — raw observations
        Returns:
            embed: shape (B, embed_dim) or (B, T, embed_dim) — encoded embeddings
        Notes:
            The forward pass should work for both 2D (B, obs_dim) and 3D (B, T, obs_dim) inputs
            because we process the sequence in world_model.encode() which handles the reshape.
        """
        # apply symlog transform to input: x = symlog(obs)
        x = symlog(obs)
        # pass through input projection
        x = self.input(x)
        # loop through hidden blocks: apply norm, activation, linear in each block
        for layer in self.hidden:
            x = layer(x)
        # pass through output projection (no activation after output)
        x = self.output(x)
        # return embedding
        return x
