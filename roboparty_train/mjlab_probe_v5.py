#!/usr/bin/env python3
"""mjlab feasibility probe v5 (audit r3 item 3, final ordering attempt).

v3: mjlab ALONE installs fast (~5min) but no jax (exit 4).
v4: jax FIRST (653s OK) then mjlab-on-top — pip re-resolves against the
    installed jax (downloads conflicting jaxlib/jax versions) and the pod
    died at its ~30min budget mid-install.
v5: MJLAB FIRST (proven fast), THEN jax (standalone install also proven,
    653s) — total ~16min < pod budget, no resolver fight expected.

Exit contract unchanged: 0=route open / 3=egress / 4=version matrix /
6=pod budget insufficient (a NEW definitive code for this failure mode,
distinguishing infra from version issues).
"""
import subprocess
import sys
import time

HB = "/tmp/mjlab_probe5_heartbeat.txt"


def hb(stage, msg=""):
    with open(HB, "a") as f:
        f.write(f"{time.time():.1f} stage={stage} {msg}\n")
    print(f"[PROBE5] stage={stage} {msg}", flush=True)


MIRROR = ["-i", "https://mirrors.aliyun.com/pypi/simple/",
          "--trusted-host", "mirrors.aliyun.com"]


def pip(label, pkgs, timeout):
    t0 = time.time()
    hb(2, f"installing {label}: {pkgs}")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--timeout", "60",
             "--no-cache-dir"] + MIRROR + pkgs,
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        hb(2, f"{label} TIMEOUT after {timeout}s")
        return False
    dt = time.time() - t0
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-4:]
        hb(2, f"{label} FAILED rc={r.returncode} ({dt:.0f}s): {' | '.join(tail)[:400]}")
        return False
    hb(2, f"{label} OK ({dt:.0f}s)")
    return True


def main():
    hb(0, f"python={sys.version.split()[0]} t0")
    # stage 2a: MJLAB FIRST (fast, proven in v3)
    if not pip("mjlab", ["mjlab"], 700):
        hb(9, "mjlab failed -> exit 4")
        return 4
    # stage 2b: jax on top (standalone-proven 653s)
    if not pip("jax", ["jax"], 900):
        hb(9, "jax after mjlab failed -> exit 4")
        return 4
    # stage 3: imports
    try:
        import jax
        hb(3, f"jax {jax.__version__} devices={jax.devices()}")
        import mujoco
        hb(3, f"mujoco {mujoco.__version__}")
        import mjlab  # noqa: F401
        hb(3, "mjlab import OK")
    except Exception as e:
        hb(3, f"IMPORT FAILED: {type(e).__name__}: {str(e)[:600]}")
        return 4
    import pkgutil
    import mjlab as _m
    hb(4, f"submodules: {[m.name for m in pkgutil.iter_modules(_m.__path__)][:12]}")
    hb(9, "ROUTE OPEN — jax+mjlab co-installable on gradmotion; MuJoCo-domain pilot schedulable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
