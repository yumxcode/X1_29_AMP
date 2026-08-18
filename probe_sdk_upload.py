#!/usr/bin/env python3
"""SDK .pt registration probe (runs INSIDE a Gradmotion container).

Purpose: determine empirically which filesystem locations the platform SDK
scans/registers as model checkpoints, and whether:
  - registration has a per-task count quota (v16-v21b: exactly 5 gvhmr junk
    files registered every run, while real 16MB checkpoints never registered)
  - the registration window closes early (late files never registered)
  - large files (16MB) are skipped
  - .gitignore'd paths (logs/) are skipped

Writes 6 early marker .pt at t=0 across candidate locations, waits 8 min,
writes 3 late markers + one 16MB dummy at t=8min, waits 8 min, exits.
After the task completes, `gm task model list` reveals the rules.

Marker files are tiny torch tensors with unique names encoding location.
"""
import functools
import sys
import time
from pathlib import Path

import torch

print = functools.partial(print, flush=True)

REPO_ROOT = Path(__file__).resolve().parent
if REPO_ROOT.name != "X1_29_AMP":
    REPO_ROOT = REPO_ROOT.parent
WS_ROOT = REPO_ROOT.parent  # /workspace/isaaclab

EARLY = [
    ("loc1_repo_root", REPO_ROOT / "probe_loc1_repo_root.pt"),
    ("loc2_model_upload", REPO_ROOT / "model_upload" / "probe_loc2_mu.pt"),
    ("loc3_ckpt_dir", REPO_ROOT / "ckpt_reg" / "probe_loc3_cr.pt"),
    ("loc4_ws_root_outside", WS_ROOT / "probe_loc4_ws.pt"),
    ("loc5_x1_upload_outside", WS_ROOT / "x1_upload" / "probe_loc5_xu.pt"),
    ("loc6_logs_gitignored", REPO_ROOT / "logs" / "probe_loc6_logs.pt"),
]
LATE = [
    ("late1_repo_root", REPO_ROOT / "probe_late1_repo_root.pt"),
    ("late2_model_upload", REPO_ROOT / "model_upload" / "probe_late2_mu.pt"),
    ("late3_x1_upload", WS_ROOT / "x1_upload" / "probe_late3_xu.pt"),
]


def write_marker(path: Path, big: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if big:
        payload = {"model_state_dict": {"w": torch.zeros(4_000_000)}}
    else:
        payload = {"probe": path.stem}
    torch.save(payload, path)
    print(f"[PROBE] wrote {path} ({path.stat().st_size} bytes)")


def main():
    print("=" * 60)
    print("SDK .pt registration probe")
    print(f"REPO_ROOT={REPO_ROOT}  WS_ROOT={WS_ROOT}")
    print("=" * 60)
    for name, p in EARLY:
        write_marker(p)
    big = REPO_ROOT / "model_upload" / "probe_big16mb.pt"
    write_marker(big, big=True)  # loc7: same place as real ckpts, 16MB size
    print("[PROBE] early markers done; sleeping 480s ...")
    time.sleep(480)
    for name, p in LATE:
        write_marker(p)
    print("[PROBE] late markers done; sleeping 480s for SDK scan ...")
    time.sleep(480)
    print("[PROBE] done — check `gm task model list` for which markers registered")


if __name__ == "__main__":
    main()
