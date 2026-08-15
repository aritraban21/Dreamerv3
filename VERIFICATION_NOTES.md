# DreamerV3 Scaffold — Verification Against the Paper

I read the actual DreamerV3 paper in the repo (`2301.04104v2.pdf` — this is the Nature version; the other two PDFs are V1 and V2). No leftover generation artifacts exist. I cross-checked every wrapper/definition and config value against the paper's equations (§"Critic/Actor learning", Eqs. 2–7) and **Table 4** (the master hyperparameter list). One note first: I installed `pypdf` into the `.venv` to read the PDF — it's not in `pyproject.toml`, so `uv sync` may remove it later; harmless, remove with `uv pip uninstall pypdf` if you want.

---

## 1. Definite bug — two tests contradict each other (blocks you no matter what you write)

`tests/test_distributions.py` has two mutually exclusive expectations for `TwoHotDist.log_prob`, which I verified numerically:

- `test_twohot_dist_log_prob_negative` asserts `log_prob(x) <= 0` (a real log-probability).
- `test_twohot_dist_loss_is_cross_entropy` asserts `log_prob(x) == -(twohot * log_softmax).sum(-1)`, i.e. **positive** cross-entropy.

One is `+CE`, the other is `−CE`. My check: the docstring impl passes the CE test but fails the "negative" test; the true-log-prob impl does the reverse. **They can't both pass.**

The correct convention (given every call site does `loss = -dist.log_prob(...)` in `predictors.py`, `critic.py`, `world_model.py`) is: `log_prob` returns the **true log-likelihood** `(twohot * log_softmax).sum(-1)` (negative), so `-log_prob` is the CE loss.

**Fixes needed (all scaffold, not your function body):**
- `distributions.py` `log_prob` docstring/pseudocode: drop the leading minus — it currently says `Return -(target * log_probs).sum(-1)` and `Lower is better`, which mislabels a log-prob as a loss.
- `test_twohot_dist_loss_is_cross_entropy`: `expected` has a sign error — should be `(twohot * log_probs).sum(-1)` (or compare against `-lp`).

---

## 2. Config values that don't match the paper's Table 4

| Param | Paper (Table 4 / Eq. 2) | Your config | Note |
|---|---|---|---|
| **Dynamics loss β_dyn** | **1.0** (Eq. 2 line 257 *and* Table 4) | `lambda_dyn: 0.5` | See below |
| **Learning rate** | **4×10⁻⁵** | `lr: 0.0001` (1e-4) | 2.5× high |
| **LaProp ε** | **10⁻²⁰** | `epsilon=1e-8` (+ docstring falsely says "1e-8 in paper") | `optim.py` |
| Replay capacity | 5×10⁶ | `1000000` | minor |
| Imagination horizon | Table 4 says **15**, main-text §Critic says **16** | `16` | paper contradicts itself; 16 is fine |

**On β_dyn — I owe you a correction.** Last turn I "aligned" the KL weights and kept `lambda_dyn = 0.5`. The repo paper actually specifies **β_dyn = 1.0** in *both* Eq. 2 and Table 4. (The widely-used official DreamerV3 GitHub code uses 0.5, which is likely where 0.5 came from — this is a real, known paper-vs-code discrepancy.) To match the paper in your repo, it should be **1.0**. Your call which to follow; I shouldn't have left it implying paper-fidelity at 0.5.

---

## 3. Algorithmic deviations from the paper (these live in functions *you* implement — flagging, not editing)

**a) λ-return bootstrap uses the wrong critic.** ✅ RESOLVED. Scaffold now matches the paper (p.6): bootstrap uses the current critic (detached), EMA is regularizer-only. Updated in `critic.py`, `agent.compute_lambda_returns()`, and README pitfalls table. Also confirmed against the official danijar/dreamerv3 code (`slowtar: False` default in both `imag_loss` and `repl_loss`; `slowreg: 1.0` uses the slow net purely as regularization target).

**b) Actor loss omits the value baseline.** Paper Eq. 6 uses the **advantage**: `sg((Rλ_t − v_ψ(s_t)) / max(1, S)) · log π + η·H`. Your `agent.py` guidance and the README use `returns_norm · log π` — i.e. `Rλ / max(1,S)` with **no `− v(s_t)` baseline**. Unbiased but higher-variance; the paper subtracts the value baseline.

**c) Critic replay loss not implemented.** Paper applies the critic loss to replayed trajectories too (`β_repval = 0.3`, p.6). Your scaffold only does the imagined critic loss. Reasonable scope cut — but `critic_repval_scale: 0.3` is a dead/mislabeled config key as a result (and it's distinct from the EMA-regularizer weight, which Table 4 lists as `1`).

---

## What matches the paper correctly (verified)

symlog/symexp (Eq. 9–10), symexp-twohot for reward & critic + symlog-MSE for the decoder, 1% unimix, 1-nat free bits clipped **per-term**, REINFORCE for both action types (Eq. 6), return norm `S = EMA(Per95 − Per5, 0.99)` with limit `L=1`, discount 0.997, λ=0.95, η=3e-4, critic EMA decay 0.98, zero-init reward/critic output heads, RMSNorm+SiLU, batch 16×64, entropy sign. All correct.

---

**None of these are new import-level breakers** — only finding #1 will actually fail a test run (and it fails regardless of your code). Findings #2–3 are fidelity issues.

Want me to apply a specific subset? My suggestion:
- **Fix now (pure scaffold, unambiguous):** #1 (the test contradiction + docstring), and the LaProp-ε docstring falsehood.
- **Your decision (config values):** β_dyn → 1.0, LR → 4e-5, replay capacity.
- **Confirm before I touch (changes what you'll implement):** the EMA-bootstrap guidance (#3a) and the advantage baseline (#3b) — these rewrite the intended algorithm in *your* functions, so I don't want to change the guidance out from under you without a yes.

Tell me which buckets to apply.
