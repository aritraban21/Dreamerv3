"""
normalizer.py — Percentile-based EMA return normalizer for DreamerV3.

DreamerV3 normalizes actor targets by the inter-percentile range of recent returns
rather than their standard deviation. This is robust to outliers and to near-zero
variance (e.g., sparse reward environments early in training where std ≈ 0).

Key behavior:
  - Under sparse rewards: denominator stays at min_scale (1.0), so returns pass through
    unchanged — the agent explores rather than amplifying tiny noise.
  - Under dense rewards: denominator tracks the return spread, keeping the normalized
    returns roughly in [-1, 1], enabling a fixed entropy coefficient eta to work universally.
"""

import torch
import numpy as np


class ReturnNormalizer:
    """
    EMA-smoothed inter-percentile range normalizer.

    Maintains a running estimate of (P_upper - P_lower) of recent returns.
    Normalizes by max(min_scale, ema_range) to prevent division by near-zero values.

    Usage in agent.py:
        normalizer.update(returns)                     # update statistics
        returns_norm = normalizer.normalize(returns)   # normalize for actor loss
    """

    def __init__(
        self,
        decay: float = 0.99,
        lower_pct: float = 5.0,
        upper_pct: float = 95.0,
        min_scale: float = 1.0,
    ):
        """
        Args:
            decay:     EMA decay for smoothing the percentile range (0.99 in paper)
            lower_pct: lower percentile (5th in paper)
            upper_pct: upper percentile (95th in paper)
            min_scale: minimum denominator — prevents amplifying near-zero returns (1.0 in paper)
        """
        # store decay, lower_pct, upper_pct, min_scale as instance attributes
        self.decay = decay
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.min_scale = min_scale
        # initialize self._ema_range = 1.0 (will be updated on first call to update())
        self._ema_range = 1.0
        # initialize self._initialized = False (track whether we've seen any data)
        self._initialized = False

    def update(self, returns: torch.Tensor) -> None:
        """
        Updates the EMA of the inter-percentile range from a new batch of returns.
        Should be called once per training iteration, before normalize().

        Args:
            returns: shape (B, H) or (N,) — a batch of return values (raw, not normalized)
        """
        # flatten returns to 1D numpy array for percentile computation
        returns_flat = torch.flatten(returns)
        # compute the lower and upper percentiles using np.percentile
        low = torch.quantile(returns_flat, self.lower_pct / 100.0)
        high = torch.quantile(returns_flat, self.upper_pct / 100.0)
        # compute current_range = upper_percentile - lower_percentile
        current_range = high - low
        # if not yet initialized: set self._ema_range = current_range; set _initialized = True
        if not self._initialized:
            self._ema_range = current_range
            self._initialized = True
        # else: update EMA: self._ema_range = decay * self._ema_range + (1 - decay) * current_range
        else:
            self._ema_range = self.decay * self._ema_range + (1 - self.decay) * current_range

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Normalizes returns by the EMA inter-percentile range.

        Formula: normalized = returns / max(min_scale, ema_range)

        Args:
            returns: shape (...,) — raw return values
        Returns:
            shape (...,) — normalized returns
        Notes:
            The max(min_scale, ...) clamp ensures we never divide by near-zero.
        """
        # compute effective scale = max(self.min_scale, self._ema_range)
        # return returns / scale (keep as tensor operation)
        # scale = torch.max(torch.tensor(self.min_scale, device=returns.device), torch.tensor(self._ema_range, device=returns.device))
        scale = max(self.min_scale, self._ema_range)
        return returns / scale

    @property
    def scale(self) -> float:
        """Current normalization denominator (after min_scale clamp). For logging."""
        # return max(self.min_scale, self._ema_range)
        return max(self.min_scale, float(self._ema_range))

    @property
    def ema_range_raw(self) -> float:
        """Raw EMA inter-percentile range BEFORE the min_scale clamp. Diagnostic only:
        lets us tell a genuinely growing return spread (this climbs) apart from a
        clamp/stuck-EMA artifact (this would sit at/below min_scale)."""
        return float(self._ema_range)