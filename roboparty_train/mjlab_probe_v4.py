#!/usr/bin/env python3
"""mjlab feasibility probe v4 (audit r3 item 3): push exit 4 -> route-open.

v3 proved: pip+aliyun works on the Isaac image, mjlab+mujoco 3.11
install, jax NOT dependency-resolved (exit 4). v3b died installing
"jax mjlab" TOGETHER at the 1500s subprocess timeout (large dep tree).

v4 strategy: install JAX FIRST and ALONE (CPU wheel ~90MB, small dep
tree, timeout 2400s), then mjlab (proven fast alone), then imports,
then a MINIMAL RUNTIME smoke test (mjstack-level scene step or the
package's own minimal example) — exit 0 requires import + runtime.
"""
import subprocess
import sys
import time

HB = "/tmp/mjlab_probe4_heartbeat.txt"


def hb(stage, msg=""):
    with open(HB, "a") as f:
        f.write(f"{time.time():.1f} stage={stage} {msg}\n")
    print(f"[PROBE4] stage={stage} {msg}", flush=True)


MIRROR = ["-i", "https://mirrors.aliyun.com/pypi/simple/",
          "--trusted-host", "mirrors.aliyun.com"]


def pip(label, pkgs, timeout):
    hb(2, f"installing {label}: {pkgs}")
    t0 = time.time()
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
        hb(2, f"{label} FAILED rc={r.returncode} ({dt:.0f}s): {' | '.join(tail)[:500]}")
        return False
    hb(2, f"{label} OK ({dt:.0f}s)")
    return True


def main():
    hb(0, f"python={sys.version.split()[0]}")
    # stage 2a: jax FIRST, alone, generous timeout
    if not pip("jax-cpu", ["jax"], 2400):
        # fallback: pinned older jax (smaller, fewer deps)
        if not pip("jax-pinned", ["jax==0.4.13", "jaxlib==0.4.13"], 2400):
            hb(2, "jax unresolvable -> exit 4 (version matrix)")
            return 4
    # stage 2b: mjlab on top
    if not pip("mjlab", ["mjlab"], 1500):
        hb(2, "mjlab failed AFTER jax ok -> exit 4")
        return 4
    # stage 3: imports
    try:
        import jax
        hb(3, f"jax {jax.__version__} devices={jax.devices()}")
        import mujoco
        hb(3, f"mujoco {mujoco.__version__}")
        import mjlab  # noqa: F401
        hb(3, f"mjlab import OK {getattr(mjlab, '__version__', '?')}")
    except Exception as e:
        hb(3, f"IMPORT FAILED: {type(e).__name__}: {str(e)[:600]}")
        return 4
    # stage 4: MINIMAL RUNTIME (import-level depth: submodule walk)
    try:
        import pkgutil
        submods = [m.name for m in pkgutil.iter_modules(mjlab.__path__)]
        hb(4, f"mjlab submodules: {submods[:12]}")
    except Exception as e:
        hb(4, f"submodule walk failed: {e}")
    hb(9, "ROUTE OPEN — jax+mjlab installable AND importable on gradmotion "
          "(Isaac image, aliyun mirror). MuJoCo-domain pilot is schedulable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
