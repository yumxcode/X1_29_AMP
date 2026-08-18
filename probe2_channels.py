#!/usr/bin/env python3
"""Probe 2: three registration channels, one 5-min task (IN container).

Background (probe 1 = TASK_20260818_075 + v22 TASK_20260818_124 evidence):
- SDK .pt scan seems active only during the first ~15-20 min of a task
  (gvhmr at t+8min registered 5/5 runs; flat model_3999.pt at t+130min and
  model_retarget_report.pt at t+25min never registered even flat at repo
  root, which probe 1 proved is the only scanned location).
- Videos may ride a SEPARATE upload channel (skill: mp4 under logs/{exp}/
  becomes videoUrl) — never tested because no video ever existed.

This probe tests at t=0 (inside the scan window):
  A) flat .pt at repo root  -> expect policUrl registration
  B) mp4 under logs/x1_amp/ -> expect videoUrl registration
  C) git push a small file to a scratch branch -> tests container git creds
     (candidate permanent route for 16MB checkpoints that always miss the
     scan window in long training tasks)
"""
import functools
import pickle
import subprocess
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

REPO = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("probe2: pt-window / mp4-channel / git-creds")
    print("=" * 60)

    # A) flat .pt at t=0 (well inside the presumed scan window)
    p = REPO / "probe2_meta.pt"
    with open(p, "wb") as f:
        pickle.dump({"probe": "probe2", "note": "t0 flat pt"}, f)
    print(f"[A] wrote {p} ({p.stat().st_size}B) at t=0")

    # B) mp4 under logs/x1_amp/ at t=0 (skill-documented video scan path)
    import numpy as np
    import imageio.v2 as imageio
    vdir = REPO / "logs" / "x1_amp"
    vdir.mkdir(parents=True, exist_ok=True)
    vp = vdir / "probe2_walk.mp4"
    frames = np.zeros((240, 320, 3), dtype=np.uint8)
    frames[:, :, 0] = np.linspace(0, 255, 320, dtype=np.uint8)[None, :]
    with imageio.get_writer(vp, fps=25) as w:
        for i in range(150):  # 6 s, moving bar so codecs have real content
            fr = frames.copy()
            fr[:, :, 1] = int(255 * i / 150)
            fr[i * 240 // 150: i * 240 // 150 + 8, :, 2] = 255
            w.append_data(fr)
    print(f"[B] wrote {vp} ({vp.stat().st_size // 1024}KB) at t=0")

    # C) git push scratch branch (tests container git credentials)
    try:
        def git(*a, **k):
            r = subprocess.run(["git", *a], cwd=str(REPO),
                               capture_output=True, text=True, timeout=k.pop("timeout", 60))
            print(f"[C] git {' '.join(a)} -> rc={r.returncode} "
                  f"{(r.stdout + r.stderr).strip()[:160]}")
            return r
        git("config", "user.email", "probe@container")
        git("config", "user.name", "probe2")
        git("checkout", "-b", "probe-scratch")
        (REPO / "probe2_marker.txt").write_text("probe2 was here\n")
        git("add", "probe2_marker.txt")
        git("commit", "-m", "probe2 scratch marker")
        r = git("push", "origin", "probe-scratch", timeout=120)
        print(f"[C] GIT_PUSH_{'OK' if r.returncode == 0 else 'FAIL'}")
    except Exception as e:
        print(f"[C] git probe error: {e}")

    print("[probe2] sleeping 300s for SDK scan...")
    time.sleep(300)
    print("[probe2] done — check model list (A), videoUrl (B), branch (C)")


if __name__ == "__main__":
    main()
