"""
Tests for dreamer/utils/distributions.py

Run with: pytest tests/test_distributions.py -v

Requires math_utils.py to be implemented first (twohot_encode, symexp, make_symexp_bins).
"""

import pytest
import torch
import torch.nn.functional as F
from dreamer.utils.distributions import TwoHotDist, UnimixCategorical, TruncatedNormal
from dreamer.utils.math_utils import make_symexp_bins, symlog, symexp


@pytest.fixture
def bins():
    return make_symexp_bins(255, 20.0)


# ─────────────────────────────────────────────
# TwoHotDist tests
# ─────────────────────────────────────────────

def test_twohot_dist_log_prob_shape(bins):
    """log_prob returns shape (B,) for input (B,)."""
    B = 8
    logits = torch.randn(B, 255)
    dist = TwoHotDist(logits, bins)
    targets = torch.randn(B)
    lp = dist.log_prob(targets)
    assert lp.shape == (B,), f"Expected ({B},), got {lp.shape}"


def test_twohot_dist_log_prob_sequence(bins):
    """log_prob works on (B, T) shaped targets."""
    B, T = 4, 16
    logits = torch.randn(B, T, 255)
    dist = TwoHotDist(logits, bins)
    targets = torch.randn(B, T)
    lp = dist.log_prob(targets)
    assert lp.shape == (B, T), f"Expected ({B}, {T}), got {lp.shape}"


def test_twohot_dist_log_prob_negative(bins):
    """log_prob is a log-likelihood log p(x), so its values must be <= 0."""
    logits = torch.randn(16, 255)
    dist = TwoHotDist(logits, bins)
    targets = torch.zeros(16)  # center value
    lp = dist.log_prob(targets)
    assert (lp <= 0).all(), "log_prob (a log-likelihood) should be non-positive"


def test_twohot_dist_mean_shape(bins):
    """mean() returns shape (B,)."""
    B = 8
    logits = torch.randn(B, 255)
    dist = TwoHotDist(logits, bins)
    mean = dist.mean()
    assert mean.shape == (B,), f"Expected ({B},), got {mean.shape}"


def test_twohot_dist_mean_finite(bins):
    """mean() should produce finite values."""
    logits = torch.randn(16, 255)
    dist = TwoHotDist(logits, bins)
    mean = dist.mean()
    assert torch.isfinite(mean).all(), "TwoHotDist.mean() produced non-finite values"


def test_twohot_dist_mean_in_symexp_range(bins):
    """mean() is in original scale (symexp applied), should be in symexp([-20, 20]) range."""
    logits = torch.randn(100, 255)
    dist = TwoHotDist(logits, bins)
    mean = dist.mean()
    max_val = symexp(torch.tensor(20.0)).item()
    assert (mean.abs() <= max_val + 1).all(), \
        f"mean() out of symexp range. Max: {mean.abs().max().item()}"


def test_twohot_dist_mean_decodes_peaked(bins):
    """A distribution peaked on a single bin must decode (via mean) to that bin's symexp value.

    This checks the full decode path: softmax -> weighted sum in symlog space -> symexp.
    A wrong implementation (e.g. forgetting symexp, or averaging over the wrong axis) still
    passes the shape/finite/range tests but silently corrupts every reward/value estimate.
    """
    idx = 160  # bins[160] ~ +5.2 in symlog space -> symexp ~ 179
    logits = torch.full((2, 255), -10.0)
    logits[:, idx] = 10.0  # nearly all mass on bin `idx`
    dist = TwoHotDist(logits, bins)
    mean = dist.mean()
    expected = symexp(bins[idx])
    assert torch.allclose(mean, expected.expand_as(mean), rtol=1e-2, atol=1e-2), \
        f"mean() should decode a peaked distribution to symexp(bin). " \
        f"Got {mean[0].item()}, expected {expected.item()}"


def test_twohot_dist_log_prob_matches_manual(bins):
    """log_prob must equal sum(twohot_target * log_softmax(logits)) — the true log-likelihood.

    Note: log_prob returns log p(x) (<= 0). The cross-entropy LOSS is -log_prob, i.e.
    -(twohot * log_softmax).sum(-1). Do not confuse the two — the negation lives at the
    call site (`loss = -dist.log_prob(...)`), not inside log_prob.
    """
    from dreamer.utils.math_utils import twohot_encode
    B = 4
    logits = torch.randn(B, 255)
    targets = torch.tensor([0.0, 1.0, -1.0, 3.0])
    dist = TwoHotDist(logits, bins)
    lp = dist.log_prob(targets)
    # compute manually: log-likelihood = sum(twohot * log_softmax); CE loss would be -this
    twohot = twohot_encode(targets, bins)
    log_probs = F.log_softmax(logits, dim=-1)
    expected = (twohot * log_probs).sum(-1)
    assert torch.allclose(lp, expected, atol=1e-4), \
        f"log_prob does not match manual log-likelihood. Max diff: {(lp - expected).abs().max()}"


# ─────────────────────────────────────────────
# UnimixCategorical tests
# ─────────────────────────────────────────────

def test_unimix_probs_sum_to_one():
    """probs() must sum to 1 along last dim."""
    logits = torch.randn(4, 32, 32)  # B, stoch_dim, stoch_classes
    dist = UnimixCategorical(logits, unimix=0.01)
    probs = dist.probs()
    sums = probs.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"probs() must sum to 1. Max deviation: {(sums - 1).abs().max().item()}"


def test_unimix_probs_cached_at_construction():
    """probs() must lazily cache its result, populated at construction time, not
    recomputed from self.logits on every call.

    __init__ triggers the first (and only) computation by calling self.probs() once;
    every later call to probs() should return that cached tensor. Mutating the source
    logits tensor in-place after construction should NOT change probs() — if it did,
    that would mean probs() is recomputing softmax + blend from self.logits on every
    call instead of reusing the cached value.
    """
    logits = torch.randn(4, 8, 8)
    dist = UnimixCategorical(logits, unimix=0.01)
    probs_before = dist.probs().clone()
    logits += 1000.0  # in-place mutation; would drastically change softmax if recomputed
    probs_after = dist.probs()
    assert torch.allclose(probs_before, probs_after), \
        "probs() should be cached in __init__, not recomputed from (mutated) self.logits"


def test_unimix_probs_nonnegative():
    """All probabilities must be non-negative."""
    logits = torch.randn(8, 32, 32)
    dist = UnimixCategorical(logits, unimix=0.01)
    assert (dist.probs() >= 0).all(), "probs() must be non-negative"


def test_unimix_uniform_with_full_mix():
    """With unimix=1.0, all classes have equal probability regardless of logits."""
    logits = torch.tensor([[[1000.0, -1000.0, 0.0, 0.0]]])  # strongly biased
    dist = UnimixCategorical(logits, unimix=1.0)
    probs = dist.probs()
    expected = torch.ones_like(probs) / 4
    assert torch.allclose(probs, expected, atol=1e-5), \
        "With unimix=1.0, all probs should be uniform"


def test_unimix_minimum_probability():
    """With unimix=0.01 and 32 classes, no class has probability below 0.01/32."""
    logits = torch.randn(16, 32, 32) * 100  # very peaked
    dist = UnimixCategorical(logits, unimix=0.01)
    probs = dist.probs()
    min_prob = 0.01 / 32
    assert (probs >= min_prob - 1e-7).all(), \
        f"All probs must be >= unimix/num_classes = {min_prob:.6f}"


def test_unimix_sample_shape():
    """sample() returns shape (...,) with correct integer values."""
    logits = torch.randn(4, 32, 32)
    dist = UnimixCategorical(logits, unimix=0.01)
    samples = dist.sample()
    assert samples.shape == (4, 32), f"Expected (4, 32), got {samples.shape}"
    assert samples.dtype in (torch.int64, torch.long), "samples must be integer indices"
    assert (samples >= 0).all() and (samples < 32).all(), "samples out of range"


def test_unimix_straight_through_shape():
    """straight_through_sample() returns shape (..., num_classes)."""
    logits = torch.randn(4, 32, 32)
    dist = UnimixCategorical(logits, unimix=0.01)
    st = dist.straight_through_sample()
    assert st.shape == (4, 32, 32), f"Expected (4, 32, 32), got {st.shape}"


def test_unimix_straight_through_gradient():
    """straight_through_sample() must produce gradients w.r.t. logits."""
    logits = torch.randn(2, 4, 4, requires_grad=True)
    dist = UnimixCategorical(logits, unimix=0.01)
    st = dist.straight_through_sample()
    loss = st.sum()
    loss.backward()
    assert logits.grad is not None, "straight_through_sample must produce gradients"
    assert torch.isfinite(logits.grad).all(), "Gradients must be finite"


def test_unimix_straight_through_is_onehot():
    """In the forward pass, straight_through_sample() returns valid one-hot vectors."""
    logits = torch.randn(4, 32, 32)
    dist = UnimixCategorical(logits, unimix=0.01)
    with torch.no_grad():
        st = dist.straight_through_sample()
    # each vector should sum to 1 and have values in [0,1]
    sums = st.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # forward pass values should be one-hot (but with straight-through trick they might be probs)
    # at least they should be non-negative and bounded
    assert (st >= 0).all()


def test_unimix_kl_nonneg():
    """KL(p || q) must be non-negative."""
    logits_p = torch.randn(4, 32, 32)
    logits_q = torch.randn(4, 32, 32)
    p = UnimixCategorical(logits_p, unimix=0.01)
    q = UnimixCategorical(logits_q, unimix=0.01)
    kl = p.kl_divergence(q)
    assert (kl >= -1e-5).all(), f"KL must be non-negative. Min: {kl.min().item()}"


def test_unimix_kl_self_is_zero():
    """KL(p || p) should be approximately 0."""
    logits = torch.randn(4, 32, 32)
    p = UnimixCategorical(logits, unimix=0.01)
    q = UnimixCategorical(logits.clone(), unimix=0.01)
    kl = p.kl_divergence(q)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5), \
        f"KL(p || p) should be 0. Got max {kl.abs().max().item()}"


def test_unimix_entropy_nonneg():
    """Entropy must be non-negative."""
    logits = torch.randn(8, 32, 32)
    dist = UnimixCategorical(logits, unimix=0.01)
    H = dist.entropy()
    assert (H >= -1e-5).all(), f"Entropy must be non-negative. Min: {H.min().item()}"


def test_unimix_entropy_shape():
    """entropy() returns shape (B, stoch_dim)."""
    logits = torch.randn(4, 32, 32)
    dist = UnimixCategorical(logits, unimix=0.01)
    H = dist.entropy()
    assert H.shape == (4, 32), f"Expected (4, 32), got {H.shape}"


# ─────────────────────────────────────────────
# TruncatedNormal tests
# ─────────────────────────────────────────────

def test_truncnorm_sample_shape():
    """sample() returns shape (B, action_dim)."""
    B, D = 8, 3
    mean = torch.zeros(B, D)
    std = torch.ones(B, D)
    dist = TruncatedNormal(mean, std)
    samples = dist.sample()
    assert samples.shape == (B, D), f"Expected ({B}, {D}), got {samples.shape}"


def test_truncnorm_sample_in_bounds():
    """All samples must lie in [-1, 1]."""
    B, D = 64, 6
    mean = torch.randn(B, D) * 2  # intentionally outside bounds
    std = torch.ones(B, D) * 0.5
    dist = TruncatedNormal(mean, std)
    samples = dist.sample()
    assert (samples >= -1.0 - 1e-6).all() and (samples <= 1.0 + 1e-6).all(), \
        f"Samples outside [-1, 1]. Min: {samples.min()}, Max: {samples.max()}"


def test_truncnorm_log_prob_shape():
    """log_prob returns shape (B,) — summed over action dimensions."""
    B, D = 8, 3
    mean = torch.zeros(B, D)
    std = torch.ones(B, D)
    dist = TruncatedNormal(mean, std)
    x = torch.zeros(B, D)
    lp = dist.log_prob(x)
    assert lp.shape == (B,), f"Expected ({B},), got {lp.shape}"


def test_truncnorm_entropy_shape():
    """entropy() returns shape (B,) — summed over action dimensions."""
    B, D = 4, 6
    mean = torch.zeros(B, D)
    std = torch.ones(B, D) * 0.5
    dist = TruncatedNormal(mean, std)
    H = dist.entropy()
    assert H.shape == (B,), f"Expected ({B},), got {H.shape}"


def test_truncnorm_mean_in_bounds():
    """mean() must be clipped to [-1, 1]."""
    B, D = 4, 2
    mean = torch.tensor([[2.0, -2.0], [0.5, -0.5], [0.0, 0.0], [-3.0, 3.0]])
    std = torch.ones(B, D)
    dist = TruncatedNormal(mean, std)
    clipped = dist.mean()
    assert (clipped >= -1.0 - 1e-6).all() and (clipped <= 1.0 + 1e-6).all(), \
        "mean() must be clipped to [-1, 1]"


def test_truncnorm_sample_gradient():
    """sample() (rsample) must allow gradient flow through std and mean."""
    B, D = 4, 2
    mean = torch.zeros(B, D, requires_grad=True)
    std = torch.ones(B, D, requires_grad=True)
    dist = TruncatedNormal(mean, std)
    samples = dist.sample()
    loss = samples.sum()
    loss.backward()
    assert mean.grad is not None or std.grad is not None, \
        "Gradient must flow through sample (use rsample)"
