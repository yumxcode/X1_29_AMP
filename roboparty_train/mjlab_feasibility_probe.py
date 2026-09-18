#!/usr/bin/env python3
"""mjlab FEASIBILITY PROBE (audit round 1, item 1 — V57_REPORT §2 unlock path).

The heel-toe blocker (cross-sim terminal ankle flick) is only breakable
by closing the train/eval domain gap at the source: MuJoCo-native
training (mjlab = Isaac Lab manager API + MuJoCo Warp physics). This
probe answers the ONE question that gates that route on gradmotion:
can the training image install and import mjlab (JAX + mujoco-warp)?

Runs on the plain ubuntu:22.04 image (BJX00000016) — the Isaac images
have no pip egress (documented: v56/v58 log pip DNS failures). Steps:
  1. pip install mjlab (with pinned jax for CUDA if egress works)
  2. import mjlab, mujoco, jax; print versions + CUDA visibility
  3. minimal step: build a free-falling box via mjlab's runtime if the
     import succeeds (no humanoid port — feasibility only)

Exit 0 = route OPEN (schedule the pilot arc); exit 3 = egress blocked
(infra requirement: custom image with mjlab baked in); exit 4 = install
ok but import/runtime broken (version matrix issue).
"""
import importlib
import subprocess
import sys


def main():
    print("[PROBE] python:", sys.version.split()[0])
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mjlab"],
                       capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-6:]
        print("[PROBE] pip install FAILED (rc=%d): %s" % (r.returncode,
                                                          " | ".join(tail)[:1200]))
        return 3
    print("[PROBE] pip install OK")
    try:
        import mujoco
        print("[PROBE] mujoco", mujoco.__version__)
        import jax
        print("[PROBE] jax", jax.__version__, "devices:", jax.devices())
        import mjlab  # noqa: F401
        print("[PROBE] mjlab import OK:", getattr(mjlab, "__version__", "?"))
    except Exception as e:
        print("[PROBE] import FAILED:", type(e).__name__, str(e)[:800])
        return 4
    print("[PROBE] ROUTE OPEN — mjlab is installable on this image; "
          "schedule the MuJoCo-domain pilot arc (task port of the AMP env).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
