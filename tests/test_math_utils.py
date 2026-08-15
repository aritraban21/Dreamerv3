"""
Tests for dreamer/utils/math_utils.py

Run with: pytest tests/test_math_utils.py -v

Implement ALL functions in math_utils.py before running these tests.
These are the foundation — if these pass, the rest of the codebase can be built on them.
"""

import pytest
import torch
import numpy as np
from dreamer.utils.math_utils import symlog, symexp, make_symexp_bins, twohot_encode, lambda_return


# ─────────────────────────────────────────────
# symlog tests
# ─────────────────────────────────────────────

def test_symlog_zero():
    """symlog(0) must be exactly 0."""
    x = torch.tensor(0.0)
    assert symlog(x).item() == pytest.approx(0.0)


def test_symlog_one():
    """symlog(1) = sign(1) * ln(|1| + 1) = ln(2) ≈ 0.693."""
    x = torch.tensor(1.0)
    assert symlog(x).item() == pytest.approx(np.log(2), abs=1e-5)


def test_symlog_negative_symmetry():
    """symlog(-x) == -symlog(x) for all x (odd function)."""
    x = torch.tensor([0.5, 1.0, 5.0, 100.0, 1e6])
    assert torch.allclose(symlog(-x), -symlog(x), atol=1e-6)


def test_symlog_large_values_finite():
    """symlog(1e6) should be finite and approximately ln(1e6+1) ≈ 13.8."""
    x = torch.tensor(1e6)
    result = symlog(x)
    assert torch.isfinite(result), "symlog(1e6) should be finite"
    assert result.item() < 20.0, "symlog should compress large values"
    assert result.item() > 10.0, "symlog(1e6) should be positive and meaningful"


def test_symlog_shape_preserved():
    """Output shape must match input shape."""
    x = torch.randn(4, 8, 3)
    assert symlog(x).shape == x.shape


def test_symlog_gradient_exists():
    """symlog should be differentiable (gradient should exist)."""
    x = torch.tensor(2.0, requires_grad=True)
    y = symlog(x)
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad)


# ─────────────────────────────────────────────
# symexp tests
# ─────────────────────────────────────────────

def test_symexp_zero():
    """symexp(0) must be exactly 0."""
    x = torch.tensor(0.0)
    assert symexp(x).item() == pytest.approx(0.0)


def test_symexp_inverse_of_symlog():
    """symexp(symlog(x)) ≈ x for a wide range of values."""
    x = torch.tensor([-1e4, -100.0, -1.5, 0.0, 1.5, 100.0, 1e4])
    reconstructed = symexp(symlog(x))
    assert torch.allclose(reconstructed, x, rtol=1e-4, atol=1e-4), \
        f"symexp(symlog(x)) != x: max error = {(reconstructed - x).abs().max().item()}"


def test_symlog_inverse_of_symexp():
    """symlog(symexp(y)) ≈ y for values in symlog space."""
    y = torch.tensor([-15.0, -5.0, 0.0, 5.0, 15.0])
    reconstructed = symlog(symexp(y))
    assert torch.allclose(reconstructed, y, rtol=1e-5, atol=1e-5)


def test_symexp_negative_symmetry():
    """symexp is also an odd function: symexp(-y) == -symexp(y)."""
    y = torch.tensor([0.5, 2.0, 10.0])
    assert torch.allclose(symexp(-y), -symexp(y), atol=1e-6)


# ─────────────────────────────────────────────
# make_symexp_bins tests
# ─────────────────────────────────────────────

def test_bins_shape():
    """make_symexp_bins(255, 20.0) must return shape (255,)."""
    bins = make_symexp_bins(255, 20.0)
    assert bins.shape == (255,), f"Expected (255,), got {bins.shape}"


def test_bins_center_near_zero():
    """The center bin (index 127 of 255) should be near 0 in symlog space."""
    bins = make_symexp_bins(255, 20.0)
    assert abs(bins[127].item()) < 0.2, \
        f"Center bin should be near 0, got {bins[127].item()}"


def test_bins_monotonically_increasing():
    """Bins should be strictly monotonically increasing."""
    bins = make_symexp_bins(255, 20.0)
    diffs = bins[1:] - bins[:-1]
    assert (diffs > 0).all(), "Bins must be monotonically increasing"


def test_bins_endpoints():
    """First and last bins should be -bin_range and +bin_range."""
    bins = make_symexp_bins(255, 20.0)
    assert bins[0].item() == pytest.approx(-20.0, abs=1e-5)
    assert bins[-1].item() == pytest.approx(20.0, abs=1e-5)


def test_bins_custom_count():
    """Works with different num_bins."""
    bins = make_symexp_bins(11, 5.0)
    assert bins.shape == (11,)
    assert bins[0].item() == pytest.approx(-5.0, abs=1e-5)
    assert bins[-1].item() == pytest.approx(5.0, abs=1e-5)


# ─────────────────────────────────────────────
# twohot_encode tests
# ─────────────────────────────────────────────

@pytest.fixture
def bins():
    return make_symexp_bins(255, 20.0)


def test_twohot_output_shape(bins):
    """twohot_encode(x, bins) must return shape (*x.shape, num_bins)."""
    x = torch.zeros(4, 8)
    encoded = twohot_encode(x, bins)
    assert encoded.shape == (4, 8, 255), f"Expected (4, 8, 255), got {encoded.shape}"


def test_twohot_sums_to_one(bins):
    """Each two-hot vector must sum to 1.0."""
    x = torch.randn(16, 64)  # arbitrary values
    encoded = twohot_encode(x, bins)
    sums = encoded.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"Two-hot vectors must sum to 1. Max deviation: {(sums - 1).abs().max().item()}"


def test_twohot_at_most_two_nonzero(bins):
    """Each two-hot vector should have at most 2 nonzero entries."""
    x = torch.tensor([0.0, 1.0, -1.0, 5.0])
    encoded = twohot_encode(x, bins)
    for i in range(len(x)):
        nonzero_count = (encoded[i] > 0).sum().item()
        assert nonzero_count <= 2, \
            f"Expected at most 2 nonzero bins, got {nonzero_count} for x={x[i].item()}"


def test_twohot_at_bin_center(bins):
    """If x exactly equals a bin center, only that bin should have weight 1.0."""
    # pick the center bin
    center_idx = 127
    x = bins[center_idx].unsqueeze(0)
    encoded = twohot_encode(x, bins)
    # the center bin should have weight ~1.0
    assert encoded[0, center_idx].item() == pytest.approx(1.0, abs=1e-4), \
        f"Expected weight 1.0 at center bin, got {encoded[0, center_idx].item()}"


def test_twohot_between_bins(bins):
    """Value between bins[10] and bins[11] should have nonzero weight only at those two."""
    midpoint = (bins[10] + bins[11]) / 2
    x = midpoint.unsqueeze(0)
    encoded = twohot_encode(x, bins)
    nonzero_indices = encoded[0].nonzero().squeeze()
    assert 10 in nonzero_indices or nonzero_indices.item() in [10, 11], \
        "Nonzero weights should be at adjacent bins only"


def test_twohot_nonnegative(bins):
    """All two-hot weights must be non-negative."""
    x = torch.randn(32)
    encoded = twohot_encode(x, bins)
    assert (encoded >= 0).all(), "Two-hot weights must be non-negative"


# ─────────────────────────────────────────────
# lambda_return tests
# ─────────────────────────────────────────────

def test_lambda_return_shape():
    """lambda_return must output shape (B, H)."""
    B, H = 4, 16
    rewards = torch.randn(B, H)
    values = torch.randn(B, H + 1)
    continues = torch.ones(B, H)
    returns = lambda_return(rewards, values, continues, lambda_=0.95, discount=0.997)
    assert returns.shape == (B, H), f"Expected ({B}, {H}), got {returns.shape}"


def test_lambda_return_terminal():
    """With continues=0 (all terminal), R_t = r_t (no discounted future)."""
    B, H = 2, 5
    rewards = torch.ones(B, H) * 2.0
    values = torch.ones(B, H + 1) * 10.0  # high value, but should be ignored
    continues = torch.zeros(B, H)           # all terminal
    returns = lambda_return(rewards, values, continues, lambda_=0.95, discount=0.99)
    # With all continues=0: R_t = r_t + discount * 0 * (...) = r_t = 2.0
    assert torch.allclose(returns, torch.ones(B, H) * 2.0, atol=1e-5), \
        f"With terminal=True, returns should equal rewards. Got {returns}"


def test_lambda_return_no_discount_lambda0():
    """With lambda=0 and no terminal, R_t = r_t + discount * V(s_{t+1})."""
    B, H = 1, 3
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.0, 0.5, 1.0, 5.0]])  # V(s_0..s_H)
    continues = torch.ones(B, H)
    discount = 0.9
    returns = lambda_return(rewards, values, continues, lambda_=0.0, discount=discount)
    # With lambda=0: R_t = r_t + discount * c_t * V(s_{t+1})
    expected_R2 = 3.0 + discount * 1.0 * 5.0    # R_2 = r_2 + gamma * V(s_3)
    expected_R1 = 2.0 + discount * 1.0 * 1.0    # R_1 = r_1 + gamma * V(s_2) [not bootstrap]
    expected_R0 = 1.0 + discount * 1.0 * 0.5    # R_0 = r_0 + gamma * V(s_1)
    expected = torch.tensor([[expected_R0, expected_R1, expected_R2]])
    assert torch.allclose(returns, expected, atol=1e-4), \
        f"lambda=0 returns mismatch. Expected {expected}, got {returns}"


def test_lambda_return_finite():
    """lambda_return should produce finite values for random inputs."""
    B, H = 8, 16
    rewards = torch.randn(B, H)
    values = torch.randn(B, H + 1)
    continues = torch.rand(B, H)
    returns = lambda_return(rewards, values, continues)
    assert torch.isfinite(returns).all(), "lambda_return produced non-finite values"


def test_lambda_return_no_gradient_leak():
    """lambda_return output should be a valid tensor (can compute gradients from it)."""
    B, H = 2, 4
    rewards = torch.randn(B, H, requires_grad=False)
    values = torch.randn(B, H + 1, requires_grad=True)
    continues = torch.ones(B, H)
    returns = lambda_return(rewards, values, continues)
    loss = returns.mean()
    loss.backward()
    assert values.grad is not None, "Gradient should flow to values"
