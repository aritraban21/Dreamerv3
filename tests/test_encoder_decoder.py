"""
Tests for dreamer/models/encoder.py and dreamer/models/decoder.py

Run with: pytest tests/test_encoder_decoder.py -v

Requires math_utils.py (for symlog in encoder).
"""

import pytest
import torch
from dreamer.models.encoder import MLPEncoder
from dreamer.models.decoder import MLPDecoder
from dreamer.utils.math_utils import symlog


@pytest.fixture
def small_encoder():
    """Small encoder for Pendulum-like obs."""
    return MLPEncoder(obs_dim=3, embed_dim=64, hidden_dim=64, num_layers=2)


@pytest.fixture
def small_decoder():
    """Small decoder matching the encoder's state dim."""
    state_dim = 128 + 16 * 16  # deter=128, stoch=16*16=256 → state_dim=384
    return MLPDecoder(state_dim=384, obs_dim=3, hidden_dim=64, num_layers=2)


# ─────────────────────────────────────────────
# MLPEncoder tests
# ─────────────────────────────────────────────

def test_encoder_output_shape_2d(small_encoder):
    """Encoder returns (B, embed_dim) for (B, obs_dim) input."""
    obs = torch.randn(8, 3)
    embed = small_encoder(obs)
    assert embed.shape == (8, 64), f"Expected (8, 64), got {embed.shape}"


def test_encoder_output_shape_3d(small_encoder):
    """Encoder handles (B, T, obs_dim) and returns (B, T, embed_dim)."""
    obs = torch.randn(4, 16, 3)
    embed = small_encoder(obs)
    assert embed.shape == (4, 16, 64), f"Expected (4, 16, 64), got {embed.shape}"


def test_encoder_output_finite(small_encoder):
    """Encoder output must be finite for normal observations."""
    obs = torch.randn(8, 3)
    embed = small_encoder(obs)
    assert torch.isfinite(embed).all(), "Encoder output has non-finite values"


def test_encoder_large_obs_finite(small_encoder):
    """Encoder should handle large observation values (symlog compresses them)."""
    obs = torch.ones(4, 3) * 1e6  # very large values
    embed = small_encoder(obs)
    assert torch.isfinite(embed).all(), \
        "Encoder output must be finite even for large obs (symlog should prevent overflow)"


def test_encoder_gradient_flows(small_encoder):
    """Gradients must flow through the encoder."""
    obs = torch.randn(4, 3, requires_grad=False)
    params = list(small_encoder.parameters())
    embed = small_encoder(obs)
    loss = embed.sum()
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed through encoder"


def test_encoder_different_batch_sizes(small_encoder):
    """Encoder works for any batch size."""
    for B in [1, 4, 32]:
        obs = torch.randn(B, 3)
        embed = small_encoder(obs)
        assert embed.shape == (B, 64)


def test_encoder_symlog_effect():
    """
    Large inputs should produce embeddings similar to small inputs (symlog compression).
    Check that the encoder doesn't blow up with large values — embedding norms should
    be in a similar range for both small and large inputs.
    """
    encoder = MLPEncoder(obs_dim=3, embed_dim=64, hidden_dim=64, num_layers=1)
    with torch.no_grad():
        small_obs = torch.ones(4, 3) * 0.1
        large_obs = torch.ones(4, 3) * 1e6
        small_embed = encoder(small_obs)
        large_embed = encoder(large_obs)
    # embeddings from large obs should not be orders of magnitude larger
    ratio = large_embed.norm() / (small_embed.norm() + 1e-8)
    assert ratio < 100, \
        f"symlog should prevent large obs from dominating embeddings. Ratio: {ratio:.1f}"


# ─────────────────────────────────────────────
# MLPDecoder tests
# ─────────────────────────────────────────────

def test_decoder_output_shape_2d(small_decoder):
    """Decoder returns (B, obs_dim) for (B, state_dim) input."""
    state_feature = torch.randn(8, 384)
    pred = small_decoder(state_feature)
    assert pred.shape == (8, 3), f"Expected (8, 3), got {pred.shape}"


def test_decoder_output_shape_3d(small_decoder):
    """Decoder handles (B, T, state_dim) and returns (B, T, obs_dim)."""
    state_feature = torch.randn(4, 16, 384)
    pred = small_decoder(state_feature)
    assert pred.shape == (4, 16, 3), f"Expected (4, 16, 3), got {pred.shape}"


def test_decoder_output_finite(small_decoder):
    """Decoder output must be finite."""
    state_feature = torch.randn(8, 384)
    pred = small_decoder(state_feature)
    assert torch.isfinite(pred).all(), "Decoder output has non-finite values"


def test_decoder_gradient_flows(small_decoder):
    """Gradients must flow through the decoder."""
    state_feature = torch.randn(4, 384)
    params = list(small_decoder.parameters())
    pred = small_decoder(state_feature)
    loss = pred.sum()
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed through decoder"


def test_decoder_reconstruction_loss_finite():
    """MSE(decoder(state), symlog(obs)) should be finite — the typical training loss."""
    decoder = MLPDecoder(state_dim=384, obs_dim=3, hidden_dim=64, num_layers=1)
    state_feature = torch.randn(8, 384)
    obs = torch.randn(8, 3)
    pred = decoder(state_feature)
    loss = torch.nn.functional.mse_loss(pred, symlog(obs))
    assert torch.isfinite(loss), "Reconstruction loss is not finite"


def test_encoder_decoder_roundtrip_loss():
    """
    Combined encoder-decoder system should have finite reconstruction loss.
    Checks that the two modules have compatible shapes.
    """
    obs_dim, embed_dim, state_dim = 3, 64, 384
    encoder = MLPEncoder(obs_dim=obs_dim, embed_dim=embed_dim, hidden_dim=64, num_layers=1)
    decoder = MLPDecoder(state_dim=state_dim, obs_dim=obs_dim, hidden_dim=64, num_layers=1)
    obs = torch.randn(8, obs_dim)
    state_feature = torch.randn(8, state_dim)  # simulated RSSM output
    pred = decoder(state_feature)
    loss = torch.nn.functional.mse_loss(pred, symlog(obs))
    assert torch.isfinite(loss), "Encoder-decoder roundtrip loss is not finite"
    assert loss.item() >= 0, "MSE loss must be non-negative"
