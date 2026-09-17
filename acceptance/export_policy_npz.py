#!/usr/bin/env python3
"""Export an rsl_rl checkpoint's actor weights (+obs normalizer) to .npz
for the MuJoCo rollout. Same logic as the pipeline's export_policy_npz.
Usage: python export_policy_npz.py <model.pt> [more.pt ...]"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

for arg in sys.argv[1:]:
    p = Path(arg)
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    pairs = {}
    for k, v in sd.items():
        m = re.match(r"actor\.(?:[a-zA-Z_]+\.)?(\d+)\.(weight|bias)$", k)
        if m:
            pairs.setdefault(int(m.group(1)), {})[m.group(2)] = v.detach().float().numpy()
    idx = sorted(i for i, d in pairs.items() if "weight" in d)
    out = {}
    for n, i in enumerate(idx):
        out[f"l{n}_w"] = pairs[i]["weight"]
        if "bias" in pairs[i]:
            out[f"l{n}_b"] = pairs[i]["bias"]
    mean = std = None
    for cand_m, cand_s in (("actor_obs_normalizer._mean", "actor_obs_normalizer._std"),
                           ("actor_obs_normalizer.mean", "actor_obs_normalizer.std")):
        if cand_m in sd and cand_s in sd:
            mean = sd[cand_m].detach().float().numpy().reshape(-1)
            std = sd[cand_s].detach().float().numpy().reshape(-1)
            break
    if mean is None:
        raise SystemExit(f"{p}: no normalizer keys: {[k for k in sd if 'normal' in k]}")
    out["mean"], out["std"] = mean, std
    dst = p.with_suffix(".policy.npz")
    np.savez(str(dst), **out)
    print(f"[EXPORT] {dst.name} iter={ck.get('iter')} obs={mean.shape}")
