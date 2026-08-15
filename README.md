# DreamerV3 — From Scratch

Implementation of [DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (Hafner et al., 2023) in PyTorch. This is a structured coding assignment — function stubs with guided comments are provided, and you implement the bodies.

---

## Overview

DreamerV3 trains a **world model** from raw observations, then trains a **policy (actor)** and **value function (critic)** purely in imagination — never needing to simulate the environment for behavior learning. A single fixed set of hyperparameters works across 150+ environments without tuning.

```
Real environment
      │
      ▼
  Encoder (MLP)
      │ embed_t
      ▼
  RSSM (GRU + Posterior/Prior)
      │ state = (h_t, z_t)
      ├─────────────────────────────────────────┐
      ▼                                         ▼
  Decoder          Reward/Continue Predictors   │
  (Reconstruction  (Prediction loss)            │
   loss)                                        │
                                                ▼
                                     Imagination Rollout (Prior only)
                                                │
                                      Actor ← [REINFORCE on normalized returns]
                                      Critic ← [TwoHot cross-entropy]
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For HalfCheetah-v4 (Part 2), also install MuJoCo
pip install mujoco
```

---

## Assignment: Implementation Guide

Follow this order — each step depends on the previous ones.

### Step 1: `dreamer/utils/math_utils.py`

Implement the four mathematical primitives. Test immediately:

```bash
pytest tests/test_math_utils.py -v
```

Key functions:
- `symlog(x)` — `sign(x) * ln(|x| + 1)`
- `symexp(x)` — inverse of symlog: `sign(x) * (exp(|x|) - 1)`
- `make_symexp_bins(255, 20.0)` — uniform bins in symlog space
- `twohot_encode(x, bins)` — two-hot vector for scalar targets
- `lambda_return(r, v, c, λ, γ)` — TD(λ) returns via backward recurrence

---

### Step 2: `dreamer/utils/distributions.py`

Implement the three distribution classes. Test:

```bash
pytest tests/test_distributions.py -v
```

Critical: `UnimixCategorical.straight_through_sample()` — a bug here silently kills all gradients to the actor.

---

### Step 3: `dreamer/utils/normalizer.py`

Implement `ReturnNormalizer` (EMA of inter-percentile range). Test:

```bash
pytest tests/test_normalizer.py -v
```

---

### Step 4: `dreamer/models/encoder.py` and `dreamer/models/decoder.py`

Implement MLP encoder (with symlog input) and decoder (outputs in symlog space). Test:

```bash
pytest tests/test_encoder_decoder.py -v
```

Architecture: `Linear → [RMSNorm → SiLU → Linear] × num_layers → Linear`

---

### Step 5: `dreamer/models/rssm.py` ← Most Important

Implement `BlockGRU` and `RSSM`. This is the hardest file. Test frequently:

```bash
pytest tests/test_rssm.py -v
```

Implementation order within the file:
1. `BlockGRU.forward()` — single GRU step
2. `initial_state()` — zeros
3. `get_state_feature()` — cat(h, z.flatten())
4. `observe()` — one posterior step
5. `imagine()` — one prior step (no observation)
6. `observe_sequence()` — loop observe() over T, handle is_first resets
7. `imagine_sequence()` — loop imagine() for H steps using actor

---

### Step 6: `dreamer/models/predictors.py`, `actor.py`, `critic.py`

Implement reward/continue predictors, actor (TruncatedNormal), and critic (TwoHot + EMA). Test:

```bash
pytest tests/test_predictors.py tests/test_actor_critic.py -v
```

**Zero-initialize output layers** in RewardPredictor, Critic, and Actor — prevents large initial predictions.

---

### Step 7: `dreamer/replay_buffer.py`

Implement the circular replay buffer with contiguous sequence sampling. Test:

```bash
pytest tests/test_replay_buffer.py -v
```

---

### Step 8: `dreamer/world_model.py`

Wire together encoder + RSSM + decoder + predictors. Implement the KL loss with free bits and KL balancing. Test:

```bash
pytest tests/test_world_model.py -v
```

**Common bug**: getting the `sg()` (stop-gradient / `.detach()`) wrong in `compute_kl_loss()`. See the docstring.

---

### Step 9: `dreamer/agent.py`

Implement the top-level agent that ties everything together. Test:

```bash
pytest tests/test_agent.py -v
```

---

### Step 10: `train.py` — Part 1

Implement the training loop, `make_env()`, `collect_random_episodes()`, and `evaluate()`. Run:

```bash
python train.py --config configs/pendulum.yaml
```

**Expected**: reward improves from ~−1400 (random) to ~−150 within 100K steps.

---

### Step 11: `dreamer/utils/optim.py` — Part 2

Implement `LaProp` and `adaptive_gradient_clipping()`. Update `agent.py` to use them. Run:

```bash
python train.py --config configs/halfcheetah.yaml
```

**Expected**: reward reaches ~4000–6000 within 1M steps.

---

## Key Paper Equations

### symlog / symexp (Eq. 1)
```
symlog(x) = sign(x) · ln(|x| + 1)
symexp(x) = sign(x) · (exp(|x|) − 1)
```

### Two-Hot Encoding
Value y is split between adjacent bins k, k+1:
```
target[k]   = (bins[k+1] − y) / (bins[k+1] − bins[k])
target[k+1] = (y − bins[k])   / (bins[k+1] − bins[k])
```
All other bins are 0. Loss = cross-entropy(predict, target).

### KL Loss (free bits + fixed β weights)
```
L_dyn = max(1, KL[sg(posterior) ∥ prior])    # trains prior
L_rep = max(1, KL[posterior ∥ sg(prior)])    # trains encoder
L_KL  = 0.5 · L_dyn + 0.1 · L_rep            # β_dyn = 0.5, β_rep = 0.1
```
`sg()` = `.detach()`. Getting this wrong causes representation collapse.
Note: DreamerV3 uses fixed β weights with 1-nat free bits, *not* DreamerV2's 0.8/0.2 KL balancing.

### λ-Return (backward recurrence)
```
R_H = V(s_H)
R_t = r_t + γ · c_t · [(1−λ) · V(s_{t+1}) + λ · R_{t+1}]
```

### Return Normalization (Eq. 7)
```
S = max(1, EMA(Perc₉₅(R) − Perc₅(R)))
R_norm = R / S
```

### Actor Loss (REINFORCE + entropy)
```
L_actor = −E[R_norm · log π(a|s)] − η · H[π(s)]    η = 3×10⁻⁴
```
> ⚠️ Paper fidelity: DreamerV3 Eq. 6 normalizes the **advantage** `(Rλ − v(s))`, not the raw
> return — i.e. it subtracts the critic value baseline before dividing by `max(1, S)`. The form
> above omits the baseline (higher variance but still unbiased). See `agent.update_actor_critic()`.

---

## Common Pitfalls

| Bug | Symptom | Fix |
|---|---|---|
| Missing `sg()` in KL loss | Reward and value loss decrease, but KL collapses to 0 and world model learns nothing useful | Check `.detach()` placement in `compute_kl_loss()` |
| Missing `straight_through_sample` | Actor loss becomes NaN or stays constant; no learning | Ensure `UnimixCategorical.straight_through_sample()` is called, not `sample()` |
| Forgetting to `.detach()` critic bootstrap values | Critic loss oscillates; critic learns to shrink its own targets | Wrap the `critic(features).mean()` call for bootstrap in `torch.no_grad()` (or detach the result). Per DreamerV3 (p.6) and the official code, bootstrap uses the **current** critic; the EMA copy is used **only** as a regularization target in the critic loss — see `critic.py` and `agent.compute_lambda_returns()`. |
| Raw rewards in TwoHot loss | Exploding loss when rewards are large (e.g., Atari) | Target must be `symlog(reward)`, not raw reward |
| Forgetting `is_first` reset in RSSM | Training diverges or shows artifact periodicity | Reset h and z for elements where `is_first=True` before `observe()` |
| Detaching features in `imagine_sequence` | Actor gradients are zero, no policy learning | Do NOT call `.detach()` inside `imagine_sequence` |

---

## Expected Learning Curves

### Part 1: Pendulum-v1
- Random policy: ~−1400 episodic return
- After 50K steps: ~−500 to −800
- After 100K steps: ~−200 to −150 (near-optimal)

### Part 2: HalfCheetah-v4
- Random policy: ~−300 to 0
- After 300K steps: ~1000 to 2000
- After 1M steps: ~4000 to 6000

---

## File Overview

```
dreamer/
├── configs/
│   ├── default.yaml        # All hyperparameters (paper values)
│   ├── pendulum.yaml       # Smaller model for Part 1 (fast iteration)
│   └── halfcheetah.yaml    # Full model for Part 2
├── dreamer/
│   ├── utils/
│   │   ├── math_utils.py   # symlog, symexp, twohot_encode, lambda_return  ← implement first
│   │   ├── distributions.py # TwoHotDist, UnimixCategorical, TruncatedNormal
│   │   ├── normalizer.py   # ReturnNormalizer (percentile EMA)
│   │   └── optim.py        # LaProp, AGC  ← Part 2 only
│   ├── models/
│   │   ├── encoder.py      # MLPEncoder
│   │   ├── decoder.py      # MLPDecoder
│   │   ├── rssm.py         # BlockGRU + RSSM  ← most important
│   │   ├── predictors.py   # RewardPredictor, ContinuePredictor
│   │   ├── actor.py        # Actor (TruncatedNormal policy)
│   │   └── critic.py       # Critic (TwoHot + EMA target)
│   ├── world_model.py      # Integrates all model components
│   ├── replay_buffer.py    # Circular replay with sequence sampling
│   └── agent.py            # DreamerV3: training coordination
├── train.py                # Training loop (Part 1 & 2)
├── evaluate.py             # Standalone evaluation
└── tests/                  # Unit and integration tests (fully implemented)
    ├── test_math_utils.py
    ├── test_distributions.py
    ├── test_normalizer.py
    ├── test_encoder_decoder.py
    ├── test_rssm.py
    ├── test_predictors.py
    ├── test_actor_critic.py
    ├── test_replay_buffer.py
    ├── test_world_model.py
    └── test_agent.py
```

---

## References

1. Hafner et al. (2023). **Mastering Diverse Domains through World Models**. arXiv:2301.04104
2. Hafner et al. (2021). **Mastering Atari with Discrete World Models** (DreamerV2). ICLR 2021.
3. Hafner et al. (2020). **Dream to Control: Learning Behaviors by Latent Imagination** (DreamerV1). ICLR 2020.
4. Brock et al. (2021). **High-Performance Large-Scale Image Recognition Without Normalization** (AGC). ICML 2021.
5. Ziyin et al. (2020). **LaProp: Separating Momentum and Adaptivity in Adam**. arXiv:2002.04839
