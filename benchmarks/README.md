# Benchmarks — reference DreamerV3 on Pendulum-v1

This folder holds an **independent, unmodified-core** copy of the standard
[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) implementation,
adapted only enough to run **Pendulum-v1 with state-based observations** using
**hyperparameters matched to our implementation**. The purpose is a clean A/B test:

> Does the standard library also destabilize/diverge on Pendulum-v1 under our
> hyperparameters, or does it learn a stable swing-up-and-hold policy?

If the reference **also** diverges → our hyperparameters (e.g. the return-scale
runaway) are the cause. If the reference **converges stably** → the divergence is
a bug in *our* reimplementation, not the algorithm or the task.

## What was added on top of the upstream copy

Everything else is upstream. The only additions/edits:

| File | Change |
|---|---|
| `dreamerv3-torch/envs/pendulum.py` | **New.** Pendulum-v1 in the repo's dict-obs / 4-tuple interface (mirrors `envs/pointmass.py`). obs = `[cosθ, sinθ, θ̇]`, dummy `image` key, `theta` for rendering. |
| `dreamerv3-torch/pendulum_video.py` | **New.** Parity videos: one real-env eval episode + one imagined rollout from the same start state, both drawn from `theta` with matplotlib (no GL/xvfb). |
| `dreamerv3-torch/configs.yaml` | Added a `pendulum:` config block (hyperparameters matched to `configs/pendulum.yaml` + `configs/default.yaml`) and three `render_video`/`video_dir`/`imag_video_horizon` defaults. |
| `dreamerv3-torch/dreamer.py` | Added a `pendulum` suite branch in `make_env`, and a video hook in the eval loop (gated by `--render_video True`). |
| `dreamerv3-torch/requirements-pendulum.txt` | **New.** Lean deps (no mujoco/dm_control/crafter) for the state-based run. |

Heavy image-env modules (`envs/atari.py`, `crafter`, `dmlab`, `minecraft`, …) and
the `.git`/`.venv` of the upstream clone were **not** copied — they are unused for
Pendulum and would only bloat the repo.

## Hyperparameter match

The reference's *own defaults* already equal ours for: `dyn_stoch=dyn_discrete=32`,
actor entropy `3e-4`, actor/critic lr `3e-5`, `model_lr=1e-4`, `grad_clip=1000`,
`dyn_scale=0.5`, `rep_scale=0.1`, `kl_free=1.0`, `discount=0.997`,
`discount_lambda=0.95`, `batch=16×64`, `train_ratio=512`, `pretrain=100`,
`unimix=0.01`. The `pendulum:` block only overrides the model **sizes**
(`dyn_deter=4096`, `dyn_hidden=1024`, `units=1024`), `imag_horizon=16`, and the env
settings (`time_limit=200`, `prefill=1000`, `eval_episode_num=25`, `eval_every=5000`,
`steps=200000`).

## Run it

See [`COLAB.ipynb`](./COLAB.ipynb) for a ready-to-run Colab notebook (clone → install
→ smoke → full run → plot `eval_return` → show videos).

Local quick smoke (CPU, tiny model, seconds):

```bash
cd benchmarks/dreamerv3-torch
pip install -r requirements-pendulum.txt   # plus a torch build for your platform
SDL_VIDEODRIVER=dummy MPLBACKEND=Agg python -u dreamer.py --configs pendulum \
  --logdir ./logdir/smoke --device cpu --steps 130 --prefill 60 --pretrain 5 \
  --eval_every 120 --eval_episode_num 1 --batch_size 4 --batch_length 16 \
  --dyn_deter 64 --dyn_hidden 64 --units 64 --dyn_stoch 8 --dyn_discrete 8 \
  --imag_horizon 5 --render_video True --imag_video_horizon 8
```

Eval returns are appended to `<logdir>/metrics.jsonl` as `eval_return` — grep/plot
those to compare against our runs. Videos land in `<video_dir>` (default
`videos/pendulum/`) as `env_step_{N}.mp4` and `imag_step_{N}.mp4`.
