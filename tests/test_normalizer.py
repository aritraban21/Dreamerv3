"""
Tests for dreamer/utils/normalizer.py

Run with: pytest tests/test_normalizer.py -v
"""

import pytest
import torch
import numpy as np
from dreamer.utils.normalizer import ReturnNormalizer


def test_normalizer_initial_scale():
    """Before any updates, scale should default to min_scale (1.0)."""
    norm = ReturnNormalizer(min_scale=1.0)
    assert norm.scale == pytest.approx(1.0), \
        "Initial scale before any update should be min_scale"


def test_normalizer_update_and_scale():
    """After updating with returns spread over [-10, 10], scale should increase above 1."""
    norm = ReturnNormalizer(decay=0.0, min_scale=1.0)  # decay=0: instant update
    returns = torch.linspace(-10, 10, 100)
    norm.update(returns)
    # range from 5th to 95th percentile of linspace(-10,10,100) is approximately [-9.1, 9.1]
    assert norm.scale > 1.0, f"Scale should increase with spread returns. Got {norm.scale}"


def test_normalizer_clamps_to_min():
    """When returns have near-zero spread, scale should be clamped to min_scale."""
    norm = ReturnNormalizer(decay=0.0, min_scale=1.0)
    # all returns equal to 5 → inter-percentile range = 0
    returns = torch.ones(100) * 5.0
    norm.update(returns)
    assert norm.scale == pytest.approx(1.0), \
        f"Scale should be clamped to min_scale for constant returns. Got {norm.scale}"


def test_normalizer_normalize_output_shape():
    """normalize() returns the same shape as input."""
    norm = ReturnNormalizer()
    returns = torch.randn(4, 16)
    norm.update(returns)
    normalized = norm.normalize(returns)
    assert normalized.shape == returns.shape


def test_normalizer_normalize_scales_correctly():
    """normalize() divides by scale — verifiable when decay=0 and returns have known spread."""
    norm = ReturnNormalizer(decay=0.0, lower_pct=0.0, upper_pct=100.0, min_scale=1.0)
    returns = torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0])
    norm.update(returns)
    # range = max - min = 16.0 - 0.0 = 16.0
    expected_scale = max(1.0, 16.0)
    normalized = norm.normalize(returns)
    expected = returns / expected_scale
    assert torch.allclose(normalized, expected, atol=1e-4), \
        f"normalize() output doesn't match expected. Scale: {norm.scale}"


def test_normalizer_ema_smoothing():
    """EMA smoothing: multiple updates converge toward true range, not instant jump."""
    norm = ReturnNormalizer(decay=0.9, min_scale=1.0)
    # first update: wide returns
    norm.update(torch.linspace(-100, 100, 100))
    scale_after_1 = norm.scale
    # second update: zero returns
    norm.update(torch.zeros(100))
    scale_after_2 = norm.scale
    # scale should decrease but not to 1 immediately (EMA smoothing)
    assert scale_after_2 < scale_after_1, "Scale should decrease after narrower returns"
    assert scale_after_2 > 1.0, "EMA should not instantly collapse to min_scale"


def test_normalizer_2d_input():
    """update() handles (B, H) shaped returns (the actual training input format)."""
    norm = ReturnNormalizer(decay=0.0, min_scale=1.0)
    returns = torch.randn(16, 64) * 5  # B=16, H=64
    norm.update(returns)  # should not raise
    normalized = norm.normalize(returns)
    assert normalized.shape == (16, 64)


def test_normalizer_normalize_is_tensor():
    """normalize() should return a torch.Tensor, not a float."""
    norm = ReturnNormalizer()
    returns = torch.randn(8, 16)
    norm.update(returns)
    result = norm.normalize(returns)
    assert isinstance(result, torch.Tensor), \
        f"normalize() should return a tensor, got {type(result)}"
