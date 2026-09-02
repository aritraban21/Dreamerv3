"""probes.py — lightweight jsonl tensor-stats logger (mirror of ours)."""
from __future__ import annotations
import json, os
import torch

_STATE = {"file": None, "step": 0, "enabled": False}


def set_probe_file(path):
    if _STATE["file"] is not None:
        _STATE["file"].close()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _STATE["file"] = open(path, "w", buffering=1)
    _STATE["step"] = 0
    _STATE["enabled"] = True


def set_step(step):
    _STATE["step"] = int(step)


def probe(name, x):
    if not _STATE["enabled"] or _STATE["file"] is None:
        return
    if isinstance(x, (int, float)):
        t = torch.tensor([float(x)])
    elif isinstance(x, torch.Tensor):
        t = x.detach().to(torch.float32).flatten()
    else:
        # numpy array or other array-like
        t = torch.as_tensor(x).to(torch.float32).flatten()
    def _f(v):
        v = float(v)
        if v != v: return None
        if v == float("inf") or v == float("-inf"): return "inf" if v > 0 else "-inf"
        return v
    rec = {
        "step": _STATE["step"],
        "name": name,
        "shape": list(getattr(x, "shape", [])),
        "mean": _f(t.mean().item()) if t.numel() > 0 else None,
        "std":  _f(t.std().item())  if t.numel() > 1 else 0.0,
        "min":  _f(t.min().item())  if t.numel() > 0 else None,
        "max":  _f(t.max().item())  if t.numel() > 0 else None,
        "first": [_f(v) for v in t[:8].tolist()],
    }
    _STATE["file"].write(json.dumps(rec) + "\n")
