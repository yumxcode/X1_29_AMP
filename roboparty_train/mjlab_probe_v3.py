#!/usr/bin/env python3
"""mjlab feasibility probe v3 (audit r2 item 1).

v1/v2 (TASK_20260918_053/_054, ubuntu:22.04-v4) died at 360s with EMPTY
logs — never reaching any diagnostic exit code. v3 changes the probe
FORM to isolate why, and produces a DEFINITIVE exit code either way:

  - writes a heartbeat file every stage (survives log-channel loss)
  - tries pip through CHINESE MIRRORS first (the documented Isaac-image
    failure was DNS for the INTERNAL /pypi mirror; aliyun/tsinghua may
    resolve), then falls back to default index
  - stages: [0] python/pip sanity  [1] DNS reachability of mirror hosts
    [2] pip install mjlab  [3] import mjlab/jax/mujoco  [4] CUDA devices

Exit codes (the probe's own contract):
  0 = route OPEN (install+import ok — schedule the MuJoCo-domain pilot)
  3 = EGRESS BLOCKED (no mirror resolves / all installs fail on network)
  4 = VERSION MATRIX BROKEN (installs but imports/runtime fail)
  5 = runtime environment itself broken (no pip / no heartbeat possible)
"""
import os
import subprocess
import sys
import time

HB = "/tmp/mjlab_probe_heartbeat.txt"


def hb(stage, msg=""):
    with open(HB, "a") as f:
        f.write(f"{time.time():.1f} stage={stage} {msg}\n")
    print(f"[PROBE3] stage={stage} {msg}", flush=True)


MIRRORS = [
    "-i", "https://mirrors.aliyun.com/pypi/simple/",
    "--trusted-host", "mirrors.aliyun.com",
]


def main():
    hb(0, f"python={sys.version.split()[0]} pid={os.getpid()}")
    r = subprocess.run([sys.executable, "-m", "pip", "--version"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        hb(0, f"NO PIP: {r.stderr[:200]}")
        return 5
    hb(0, "pip present")

    # stage 1: DNS reachability
    import socket
    reachable = []
    for host in ["mirrors.aliyun.com", "pypi.tuna.tsinghua.edu.cn", "pypi.org"]:
        try:
            socket.getaddrinfo(host, 443)
            reachable.append(host)
            hb(1, f"DNS OK {host}")
        except Exception as e:
            hb(1, f"DNS FAIL {host}: {type(e).__name__}")
    if not reachable:
        hb(1, "no package host resolves -> EGRESS BLOCKED")
        return 3

    # stage 2: install (mirror first, default fallback)
    ok = False
    for label, extra in [("aliyun-mirror", MIRRORS), ("default-index", [])]:
        cmd = [sys.executable, "-m", "pip", "install", "-q",
               "--timeout", "30"] + extra + ["mjlab"]
        hb(2, f"trying {label}")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
        except subprocess.TimeoutExpired:
            hb(2, f"{label} TIMEOUT")
            continue
        if r.returncode == 0:
            hb(2, f"{label} INSTALL OK")
            ok = True
            break
        tail = (r.stderr or "").strip().splitlines()[-4:]
        hb(2, f"{label} FAILED rc={r.returncode}: {' | '.join(tail)[:600]}")
    if not ok:
        hb(2, "all install routes failed")
        # distinguish: network errors vs resolution conflicts
        return 3

    # stage 3: imports
    try:
        import mujoco
        hb(3, f"mujoco {mujoco.__version__}")
        import jax
        hb(3, f"jax {jax.__version__}")
        import mjlab  # noqa: F401
        hb(3, f"mjlab import OK {getattr(mjlab, '__version__', '?')}")
    except Exception as e:
        hb(3, f"IMPORT FAILED: {type(e).__name__}: {str(e)[:500]}")
        return 4

    # stage 4: CUDA
    try:
        import jax
        devs = jax.devices()
        hb(4, f"jax devices: {devs}")
    except Exception as e:
        hb(4, f"jax devices FAILED: {e}")
        return 4

    hb(9, "ROUTE OPEN — mjlab installable+importable on this image")
    return 0


if __name__ == "__main__":
    sys.exit(main())
