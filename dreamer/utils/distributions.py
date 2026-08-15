"""
distributions.py — Custom probability distributions for DreamerV3.

Three distributions are needed:
  1. TwoHotDist    — categorical over symexp bins; used for reward/value prediction
  2. UnimixCategorical — categorical mixed with uniform; used for RSSM latent z_t
  3. TruncatedNormal   — bounded normal for continuous actions

Implement in this order. TwoHotDist depends on math_utils; UnimixCategorical
contains straight_through_sample which is critical for gradient flow through z_t.
"""

import torch
import torch.nn.functional as F

from dreamer.utils.math_utils import twohot_encode, symexp


class TwoHotDist:
    """
    Categorical distribution over exponentially-spaced bins for scalar regression.

    Why: Predicting scalars (rewards, values) via cross-entropy on a two-hot target
    is more stable than MSE, because the gradient magnitude is bounded by the bin
    weights rather than the raw prediction error (which can be huge for large rewards).

    Usage pattern:
        bins = make_symexp_bins(255, 20.0)                    # create once, cache
        logits = reward_predictor(state_feature)              # shape (..., 255)
        dist = TwoHotDist(logits, bins)
        loss = -dist.log_prob(symlog(reward)).mean()          # cross-entropy loss
        pred = dist.mean()                                    # decode to original scale
    """

    def __init__(self, logits: torch.Tensor, bins: torch.Tensor):
        """
        Args:
            logits: shape (..., num_bins) — raw network output (pre-softmax)
            bins:   shape (num_bins,)     — symlog-space bin centers from make_symexp_bins()
        """
        # store logits and bins as instance attributes
        self.logits = logits
        self.bins = bins

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log-likelihood of target x under the two-hot categorical. Returns log p(x),
        which is <= 0. The training loss is the cross-entropy CE = -log_prob, so every
        call site does `loss = -dist.log_prob(...)` (see predictors.py / critic.py).

        IMPORTANT: this returns a log-probability (negative), NOT the CE loss. Do not
        add a leading minus here — the minus belongs at the call site. (A previous
        version of this docstring returned -(...) which flipped the sign of every loss.)

        Steps:
            1. Encode x into a two-hot target: target = twohot_encode(x, self.bins)
            2. Compute log-softmax of logits along last dim: log_probs = log_softmax(logits)
            3. Return (target * log_probs).sum(-1)     [log p = sum(target * log(p)); CE = -this]

        Args:
            x: shape (...,) — scalar targets in symlog space (apply symlog(reward) before calling)
        Returns:
            shape (...,) — log-likelihood per element (<= 0). Negate and mean to get the CE loss.
        """
        # encode x into two-hot target vector using twohot_encode(x, self.bins)
        x = twohot_encode(x, self.bins)
        # compute log-softmax of self.logits along the last dimension
        log_probs = F.log_softmax(self.logits, dim=-1)
        # compute log-likelihood: (target * log_probs).sum(dim=-1)   [NO leading minus]
        log_likelihood = (x*log_probs).sum(dim=-1)
        # return the result (the caller negates it to get the cross-entropy loss)
        return log_likelihood

    def mean(self) -> torch.Tensor:
        """
        Expected value of the distribution, decoded back to original (non-symlog) scale.

        Steps:
            1. probs = softmax(logits, dim=-1)         shape (..., num_bins)
            2. value_symlog = (probs * bins).sum(-1)   weighted average in symlog space
            3. return symexp(value_symlog)             decode to original scale

        Returns:
            shape (...,) — predicted scalar in original (not symlog) space
        """
        # compute softmax probabilities from self.logits
        probs = F.softmax(self.logits, dim=-1)
        # compute weighted sum of bin centers: (probs * self.bins).sum(-1)
        value_symlog = (probs * self.bins).sum(-1)
        # apply symexp to convert from symlog space to original scale
        value = symexp(value_symlog)
        # return the result
        return value


class UnimixCategorical:
    """
    Categorical distribution mixed with a uniform distribution.
    Used for the stochastic RSSM latent z_t (shape: B, stoch_dim, stoch_classes).

    Why the uniform mix: prevents any category from having exactly zero probability,
    which would make the KL divergence between posterior and prior infinite or unstable.

    p_mixed(k) = (1 - unimix) * softmax(logits)[k]  +  unimix / num_classes

    The key method is straight_through_sample(), which allows gradients to flow
    through the discrete sampling step (required for actor-critic training).

    Caching: probs() lazily computes and caches the mixed distribution on first call;
    __init__ triggers that first call immediately so the cache is populated at
    construction time. Since logits/unimix are fixed after construction, every other
    method (sample, straight_through_sample, log_prob, entropy, kl_divergence) reads
    the same cached tensor instead of recomputing softmax + blend on each call.
    """

    def __init__(self, logits: torch.Tensor, unimix: float = 0.01):
        """
        Args:
            logits:  shape (..., num_classes) — raw network output
            unimix:  float in [0, 1] — uniform mixture fraction (0.01 in paper)
        Notes:
            Calls self.probs() once here purely to trigger and cache the computation
            immediately at construction time. The actual softmax + uniform-blend logic
            lives in probs() itself (single source of truth) — see its docstring.
        """
        # store logits and unimix as instance attributes
        self.logits = logits
        self.unimix = unimix
        # initialize the cache slot (probs() checks this to decide whether to compute)
        self._probs = None
        # call self.probs() once to compute and cache the mixed probs now, rather than
        # waiting for the first external call to probs()/sample()/log_prob()/etc.
        self.probs()

    def probs(self) -> torch.Tensor:
        """
        Returns the mixed probability distribution.

        Formula: p = (1 - unimix) * softmax(logits) + unimix / num_classes

        This is a lazily-cached property: the first call (triggered by __init__)
        computes the mixed probs and stores them in self._probs; every subsequent
        call (from sample()/straight_through_sample()/log_prob()/entropy()/
        kl_divergence()) just returns that cached tensor instead of recomputing
        softmax + blend. Since logits/unimix never change after construction, the
        mixed probs are constant for the lifetime of the instance — caching avoids
        redundant work. This matters because a new UnimixCategorical is constructed
        at every RSSM step (observe()/imagine()), and straight_through_sample() alone
        would otherwise trigger two recomputations per step (once directly, once via
        self.sample()).

        Returns:
            shape (..., num_classes) — mixed probabilities, summing to 1 along last dim
        """
        # if not yet cached (self._probs is None): compute and cache it now
        #   compute softmax over self.logits along the last dimension
        #   compute uniform distribution: ones_like(softmax_probs) / num_classes
        #   blend: (1 - self.unimix) * softmax_probs + self.unimix * uniform_probs
        #   store the blended probabilities in self._probs
        # return self._probs (freshly computed on first call, cached on every call after)
        if self._probs is None:
            num_classes = self.logits.shape[-1]
            softmax_probs = F.softmax(self.logits, dim=-1)
            uniform_probs = torch.ones_like(softmax_probs) / num_classes
            self._probs = (1 - self.unimix) * softmax_probs + self.unimix * uniform_probs
        return self._probs

    def sample(self) -> torch.Tensor:
        """
        Samples an integer class index from the mixed distribution. Not differentiable.

        Returns:
            shape (...,) — integer class indices, values in [0, num_classes-1]
        """
        # use torch.distributions.Categorical(probs=...).sample() to draw a sample
        indices = torch.distributions.Categorical(probs=self._probs).sample()
        # return the sampled indices
        return indices

    def straight_through_sample(self) -> torch.Tensor:
        """
        Samples a one-hot vector with straight-through gradient estimator.

        Forward pass:  one-hot of the sampled class (discontinuous)
        Backward pass: gradient flows as if we used probs directly (continuous)

        This is the standard trick for differentiating through discrete samples.
        Trick: sample_onehot + probs - probs.detach()
            - In forward: probs.detach() cancels probs, leaving sample_onehot
            - In backward: gradient flows through probs (the only non-detached term)

        Returns:
            shape (..., num_classes) — one-hot in forward, probs-gradient in backward
        Notes:
            This is called inside RSSM.observe() and RSSM.imagine() at every step.
            A bug here silently kills all gradient flow to the actor without an error.
        """
 
        # sample class indices via self.sample() (internally reuses the same cached probs)
        indices = self.sample()
        # convert sampled indices to one-hot: F.one_hot(indices, num_classes).float()
        one_hot = F.one_hot(indices, num_classes=self._probs.shape[-1]).float()
        # apply straight-through: one_hot + probs - probs.detach()
        result = one_hot + self._probs - self._probs.detach()
        # return the result (float tensor, same shape as probs)
        return result

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log probability of integer class indices x.

        Args:
            x: shape (...,) — integer class indices
        Returns:
            shape (...,) — log p(x) under the mixed distribution
        """
        # get mixed probs via self.probs() (cached — no recomputation)
        # gather the probabilities at the given indices
        probs_gathered = self._probs.gather(-1,index=x.unsqueeze(-1)).squeeze(-1)
        # return log of gathered probabilities (add small epsilon for numerical safety)
        return torch.log(probs_gathered + 1e-8)

    def entropy(self) -> torch.Tensor:
        """
        Entropy of the mixed distribution: H[p] = -sum_k p(k) * log(p(k))

        Returns:
            shape (...,) — entropy per distribution in the batch
        """
        # get mixed probs via self.probs() (cached — no recomputation)
        # compute log of probs (add small epsilon to avoid log(0))
        log_probs = torch.log(self._probs + 1e-8)
        # entropy = -(probs * log_probs).sum(dim=-1)
        entropy = -(self._probs*log_probs).sum(dim=-1)
        # return entropy
        return entropy

    def kl_divergence(self, other: "UnimixCategorical") -> torch.Tensor:
        """
        KL(self || other) — used for world model KL losses.

        KL(p || q) = sum_k p(k) * (log p(k) - log q(k))

        Used in two ways (see world_model.py compute_kl_loss):
          - Dynamics loss:        KL[sg(posterior) || prior]    (trains prior toward posterior)
          - Representation loss:  KL[posterior || sg(prior)]    (trains encoder toward prior)
        where sg() means stop-gradient (.detach()).

        Args:
            other: another UnimixCategorical with matching shape
        Returns:
            shape (...,) — KL per element, summed over the num_classes dimension
        Notes:
            The paper also sums over the stoch_dim dimension (32 independent categoricals).
            That sum happens in compute_kl_loss, not here.
        """
        # get mixed probs for self and other via self.probs() / other.probs() (both cached)
        # compute log probs for self (add epsilon for safety)
        log_probs_self = torch.log(self._probs + 1e-8)
        # compute log probs for other (add epsilon for safety)
        log_probs_other = torch.log(other._probs + 1e-8)
        # KL = (self_probs * (log_self - log_other)).sum(dim=-1)
        kl = (self._probs * (log_probs_self - log_probs_other)).sum(dim=-1)
        # return KL (should be non-negative)
        return kl


class TruncatedNormal:
    """
    Normal distribution truncated to [low, high] for bounded continuous actions.

    Used by the Actor for environments with bounded action spaces (standard in control).
    Actions are clipped during sampling; log_prob accounts for the truncation normalization.

    For simplicity, a common implementation is to:
    1. Sample from the base Normal, then clamp to [low, high].
    2. Use the base Normal log_prob (ignoring the truncation normalizer) — an approximation
       that works well in practice when most probability mass is inside the bounds.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
    ):
        """
        Args:
            mean: shape (B, action_dim) — predicted mean actions
            std:  shape (B, action_dim) — predicted standard deviations (must be > 0)
            low:  lower bound (default -1.0)
            high: upper bound (default +1.0)
        """
        # store mean, std, low, high as instance attributes
        self._mean = mean
        self._std = std
        self._low = low
        self._high = high
        # create a torch.distributions.Normal(mean, std) as self.base_dist
        self.base_dist = torch.distributions.Normal(mean, std)

    def sample(self) -> torch.Tensor:
        """
        Samples actions from the normal distribution, clamped to [low, high].

        Returns:
            shape (B, action_dim) — sampled actions in [low, high]
        """
        # sample from self.base_dist using rsample() (reparameterized, keeps gradients)
        sample = self.base_dist.rsample()
        # clamp sample to [self._low, self._high]
        sample = torch.clamp(sample, self._low, self._high)
        # return clamped sample
        return sample

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log probability of action x. Sums over action_dim.
        Uses the base Normal log_prob (approximation — ignores truncation constant).

        Args:
            x: shape (B, action_dim)
        Returns:
            shape (B,) — summed log probability per batch element
        """
        # compute log_prob from self.base_dist for each action dimension
        # clamp x to [low, high] before computing log_prob to avoid -inf at boundaries
        x = torch.clamp(x, self._low, self._high)
        log_prob = self.base_dist.log_prob(x)
        # sum over the action_dim dimension (dim=-1) and return
        return log_prob.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """
        Approximate entropy of the truncated normal: uses the base Normal entropy.
        Sum over action_dim.

        Returns:
            shape (B,) — entropy per batch element
        """
        # compute entropy of self.base_dist (closed-form for Normal: 0.5 * ln(2*pi*e*sigma^2))
        #entropy = 0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)) * self._std ** 2)
        entropy = self.base_dist.entropy()
        # sum over action_dim (dim=-1) and return
        return entropy.sum(dim=-1)

    def mean(self) -> torch.Tensor:
        """
        Returns the mean action, clamped to [low, high]. Used at evaluation time.

        Returns:
            shape (B, action_dim)
        """
        # clamp self._mean to [self._low, self._high] and return
        return torch.clamp(self._mean, self._low, self._high)
