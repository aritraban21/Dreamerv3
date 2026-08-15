"""
optim.py — Part 2 optimizer: LaProp + Adaptive Gradient Clipping (AGC).

These are stubs for Part 2. In Part 1, use torch.optim.Adam with global norm clipping.
Only implement these after Part 1 training is working on Pendulum-v1.

LaProp (Ziyin et al., 2020): A variant of Adam that applies the learning rate BEFORE
computing the variance estimate, rather than after. This decouples the effective step size
from gradient variance, providing more stable training on heterogeneous losses.

AGC (Brock et al., 2021): Per-parameter gradient clipping that clips each parameter's
gradient norm to be at most agc_lambda times its parameter norm. More principled than
global norm clipping because it adapts to each parameter's natural scale.
"""

import torch
import torch.optim as optim


class LaProp(optim.Optimizer):
    """
    LaProp optimizer. Drop-in replacement for Adam in Part 2.

    Key difference from Adam:
        Adam:   param -= lr * m_t / (sqrt(v_t) + eps)
                where v_t estimates E[g^2]

        LaProp: param -= m_t / (sqrt(v_t) + eps)
                where v_t estimates E[(lr * g)^2] / lr^2
                i.e., the LR is applied INSIDE the variance estimate

    This prevents the LR from being implicitly scaled by gradient variance,
    leading to more consistent effective step sizes across parameters.

    Reference: https://arxiv.org/abs/2002.04839
    """

    def __init__(
        self,
        params,
        lr: float = 4e-5,
        betas: tuple = (0.9, 0.999),
        epsilon: float = 1e-20,
    ):
        """
        Args:
            params:  iterable of parameters or parameter groups
            lr:      learning rate (4e-5 in DreamerV3, Table 4)
            betas:   (beta1, beta2) — EMA coefficients for first and second moment
            epsilon: numerical stability constant. NOTE: DreamerV3 Table 4 uses eps = 1e-20,
                     NOT Adam's typical 1e-8. This very small value is characteristic of LaProp
                     (the LR is applied inside the variance estimate, so a tiny eps is stable).
        """
        # create defaults dict with keys: lr, betas, epsilon
        # call super().__init__(params, defaults)
        raise NotImplementedError

    def step(self, closure=None):
        """
        Performs one LaProp optimization step.

        For each parameter p with gradient g:
            1. m_t = beta1 * m_{t-1} + (1 - beta1) * g            [first moment]
            2. step_t = lr * g                                      [scaled gradient]
            3. v_t = beta2 * v_{t-1} + (1 - beta2) * step_t^2     [second moment of SCALED grad]
            4. m_hat = m_t / (1 - beta1^t)                         [bias correction]
            5. v_hat = v_t / (1 - beta2^t)                         [bias correction]
            6. p -= m_hat / (sqrt(v_hat) + epsilon)                [parameter update]

        Args:
            closure: optional callable that re-evaluates the model and returns loss
        Returns:
            loss: computed loss if closure provided, else None
        """
        # optionally evaluate closure if provided
        # loop over all parameter groups
        #     extract lr, beta1, beta2, epsilon from group
        #     loop over each parameter p in group
        #         skip if p.grad is None
        #         get or initialize state for p: step count, exp_avg (m), exp_avg_sq (v)
        #         increment step count
        #         compute scaled gradient: step_g = lr * p.grad.data
        #         update first moment: m = beta1 * m + (1 - beta1) * p.grad.data
        #         update second moment of scaled grad: v = beta2 * v + (1 - beta2) * step_g^2
        #         compute bias-corrected first moment: m_hat = m / (1 - beta1^step)
        #         compute bias-corrected second moment: v_hat = v / (1 - beta2^step)
        #         update parameter: p.data -= m_hat / (sqrt(v_hat) + epsilon)
        # return loss
        raise NotImplementedError


def adaptive_gradient_clipping(
    parameters,
    lambda_agc: float = 0.3,
    eps: float = 1e-3,
) -> None:
    """
    Adaptive Gradient Clipping (AGC). Clips each parameter's gradient so that
    ||g_i|| <= lambda_agc * ||w_i||. More principled than global norm clipping
    because it respects each parameter's natural scale.

    Formula: g_i = g_i * min(1, lambda_agc * ||w_i|| / max(||g_i||, eps))

    Call this AFTER loss.backward() and BEFORE optimizer.step().

    Args:
        parameters:  iterable of nn.Parameter (e.g., list(model.parameters()))
        lambda_agc:  clipping threshold (0.3 in DreamerV3 paper)
        eps:         minimum gradient norm to avoid division by zero (1e-3)
    Notes:
        - Skip 1D parameters (biases, batch norm scales) — they don't benefit from AGC.
        - Use Frobenius norm (torch.norm) for both parameter and gradient norms.
        - This clips IN PLACE — no return value needed.
    """
    # convert parameters to a list (to allow iteration)
    # loop over each parameter p
    #     skip if p.grad is None (no gradient computed)
    #     skip if p.ndim == 1 (bias terms and 1D params excluded from AGC)
    #     compute param_norm = ||p.data|| (Frobenius norm)
    #     compute grad_norm  = ||p.grad.data|| (Frobenius norm)
    #     compute max_norm = lambda_agc * param_norm (the allowed gradient magnitude)
    #     compute clip_coef = max_norm / max(grad_norm, eps)
    #     if clip_coef < 1: scale gradient in-place: p.grad.data *= clip_coef
    raise NotImplementedError
