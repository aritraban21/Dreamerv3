"""
math_utils.py — Core mathematical primitives for DreamerV3.

Implement these FIRST. Every other module depends on symlog/symexp/twohot/lambda_return.
Run tests/test_math_utils.py after implementing each function.

Key insight: DreamerV3 uses symlog to handle rewards/observations at wildly different
scales (Atari scores ~10000, control rewards ~1, Minecraft ~0). The symlog transform
maps large values to a manageable range while keeping gradients stable.
"""

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric logarithm transform. Compresses large values while preserving sign.
    Used to normalize: encoder inputs, decoder targets, reward targets, value targets.

    Formula: sign(x) * ln(|x| + 1)
    Property: symlog(0) = 0, symlog(-x) = -symlog(x), approximately identity near 0.

    Args:
        x: Tensor of any shape and dtype.
    Returns:
        Tensor of same shape. Finite for all real inputs.
    """
    # extract the sign of x (returns -1, 0, or 1 element-wise)
    x = torch.sign(x) * torch.log(torch.abs(x) + 1.0)
    # compute ln(|x| + 1) — the +1 ensures ln(0) = 0 instead of -inf
    # multiply sign by the log magnitude and return
    return x


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of symlog. Expands compressed values back to original scale.
    Use this to decode TwoHot bin predictions back into real reward/value estimates.

    Formula: sign(x) * (exp(|x|) - 1)
    Property: symexp(symlog(x)) == x for all finite x.

    Args:
        x: Tensor of any shape (typically in symlog space).
    Returns:
        Tensor of same shape in original (uncompressed) scale.
    """
    # extract the sign of x
    # compute exp(|x|) - 1 — the -1 ensures symexp(0) = 0
    # multiply sign by the exponential magnitude and return
    x = torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)
    return x


def make_symexp_bins(num_bins: int = 255, bin_range: float = 20.0) -> torch.Tensor:
    """
    Creates bin centers for TwoHot distributions. Bins are uniformly spaced in
    symlog space, meaning they are exponentially spaced in original value space.
    This covers a huge dynamic range symmetrically around 0.

    Example: with num_bins=255, bin_range=20.0:
        - Bins in symlog space: linspace(-20, 20, 255)
        - Bin 127 (center) is 0 in both spaces
        - Bins in original space: symexp(-20) ≈ -485M  to  symexp(20) ≈ 485M

    Args:
        num_bins:  Number of bins (255 in the paper).
        bin_range: Max absolute bin value in symlog space (20.0 in the paper).
    Returns:
        bins: shape (num_bins,) — bin centers in symlog space (NOT symexp'd yet).
              Pass these directly to TwoHotDist. Call symexp(bins) to get original-space centers.
    """
    # create num_bins evenly spaced values from -bin_range to +bin_range
    bins = torch.linspace(-bin_range, bin_range, num_bins)
    return bins


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Encodes scalar targets as two-hot vectors for use as cross-entropy targets.
    A two-hot vector has nonzero weight at the two bins nearest to x, proportional
    to proximity — a generalization of one-hot to continuous values.

    Why: Instead of MSE(scalar_pred, scalar_target), we minimize CE(dist_pred, twohot_target).
    This decouples gradient magnitude from value scale (1e6 reward doesn't cause 1e6 gradient).

    Args:
        x:    shape (...,) — scalar targets in symlog space (apply symlog before calling)
        bins: shape (num_bins,) — bin centers from make_symexp_bins() (in symlog space)
    Returns:
        shape (..., num_bins) — two-hot encoded vectors, each summing to 1.0.
    Notes:
        - Values outside [bins[0], bins[-1]] get all weight on the nearest edge bin.
        - Only two adjacent positions are nonzero in each output vector.
    """
    # clamp x to the range [bins[0], bins[-1]] to handle out-of-range values
    x = torch.clamp(x, bins[0], bins[-1])
    # find the index of the bin just below x: lower_idx = searchsorted(bins, x) - 1
    lower_idx = torch.searchsorted(bins,x,right=True) - 1
    # clamp lower_idx to [0, num_bins-2] so upper_idx = lower_idx + 1 is always valid
    lower_idx = torch.clamp(lower_idx, 0, bins.shape[0] - 2)
    # compute upper_idx = lower_idx + 1
    upper_idx = lower_idx + 1
    # compute the weight on the upper bin: upper_weight = (x - bins[lower_idx]) / (bins[upper_idx] - bins[lower_idx])
    upper_weight = (x - bins[lower_idx]) / (bins[upper_idx] - bins[lower_idx])
    # lower_weight = 1 - upper_weight
    lower_weight = 1.0 - upper_weight
    # create a zeros tensor of shape (*x.shape, num_bins)
    result = torch.zeros(*x.shape, bins.shape[0], device=x.device, dtype=x.dtype)
    # scatter lower_weight at lower_idx positions along the last dimension
    result.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
    # scatter upper_weight at upper_idx positions along the last dimension
    result.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
    # return the result
    return result


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    lambda_: float = 0.95,
    discount: float = 0.997,
) -> torch.Tensor:
    """
    Computes TD(lambda) bootstrapped returns over an imagined trajectory.
    These returns are used as regression targets for the critic and advantage
    estimates for the actor.

    Recurrence (computed backward from t=H-1 to t=0):
        R_H     = V(s_H)                                           [pure bootstrap]
        R_t     = r_t + discount * c_t * [(1-lambda_) * V(s_{t+1}) + lambda_ * R_{t+1}]

    Expanding: R_t mixes immediate rewards with discounted future value, with lambda_
    controlling how much to rely on bootstrapped values vs. multi-step returns.

    Args:
        rewards:   shape (B, H)    — imagined rewards r_t, t = 0 .. H-1
        values:    shape (B, H+1)  — critic estimates V(s_t), t = 0 .. H (incl. bootstrap)
        continues: shape (B, H)    — continuation flags c_t (0 at terminal, 1 otherwise)
        lambda_:   float — mixing coefficient (0.95 in paper; 0=TD(0), 1=MC returns)
        discount:  float — per-step discount gamma (0.997 in paper)
    Returns:
        returns: shape (B, H) — lambda-return R_t^lambda for each step t = 0..H-1
    Notes:
        - Compute this BACKWARD (H-1 down to 0) to reuse R_{t+1} in R_t.
        - Start the backward loop with R_next = values[:, -1] (the bootstrap value V(s_H)).
        - At each step: R_t = r_t + discount * c_t * ((1 - lambda_) * V_{t+1} + lambda_ * R_{t+1})
        - Collect all R_t into a list, reverse it, and stack into shape (B, H).
    """
    # initialize R_next = values[:, -1] (bootstrap from final critic estimate)
    R_next = values[:, -1]
    # create an empty list to collect R_t values
    R_t = torch.zeros_like(rewards)
    # loop backward: for t in range(H-1, -1, -1):
    for t in range(rewards.shape[1]-1, -1,-1):
        
        R_t[:,t] = rewards[:,t] + discount * continues[:,t] *((1-lambda_)*values[:,t+1] +lambda_*R_next)
        R_next = R_t[:,t]

    return R_t