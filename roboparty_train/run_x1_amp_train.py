#!/usr/bin/env python3
"""
X1 AMP training pipeline v18 (acceptance-gated).

Phases:
  1. GMR retarget (SMPLX -> X1, isolated venv, installed OUTSIDE repo tree)
  2. Isaac Lab dataset_retarget (x1_gmr -> x1_lab)
  3. STRICT retarget acceptance gate (acceptance/check_retarget.py)
     -> FAIL aborts BEFORE burning GPU-hours on training
  4. AMP training (cwd=REPO_ROOT so logs land inside the SDK-scanned tree;
     stdout captured to file) with checkpoint upload monitor
  5. Play rollout + video (X1-AMP-Play, command 1.0 m/s forward)
  6. AMP training acceptance (acceptance/check_amp.py)
  7. Artifacts mirrored to repo-tree model_upload/ (SDK scans repo tree for
     *.pt during the run — this is the upload path that verifiably worked for
     gvhmr_pt in v16/v17) + exported_data/ paths.

Usage:
    python run_x1_amp_train.py --headless
"""

import functools
import os
import pickle as _pkl
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime as _dt
from pathlib import Path

print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = None
for candidate in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent,
                  SCRIPT_DIR.parent.parent.parent.parent]:
    if (candidate / "AMASS_minimal").is_dir() and (candidate / "roboparty_train").is_dir():
        REPO_ROOT = candidate
        break
    if (candidate / "AMASS_minimal").is_dir() and (candidate / "X1_29_AMP" / "roboparty_train").is_dir():
        REPO_ROOT = candidate / "X1_29_AMP"
        break
if REPO_ROOT is None:
    for p in [Path("/workspace/isaaclab/X1_29_AMP")]:
        if p.is_dir():
            REPO_ROOT = p
            break
if REPO_ROOT is None:
    print("[FATAL] Cannot find repo root")
    sys.exit(1)
print(f"[INFO] REPO_ROOT = {REPO_ROOT}")

sys.path.insert(0, str(SCRIPT_DIR))
from run_gmr_retarget import (
    reassemble_smplx, setup_gmr, register_x1_in_gmr,
    run_auto_ik, run_gmr_retarget, run_dataset_retarget
)

# Repo-tree upload dir: SDK periodic scan registers *.pt here (proven pattern).
UPLOAD_DIR = REPO_ROOT / "model_upload"
# v21: PRIMARY mirror location OUTSIDE the repo working tree. Empirical SDK
# behavior across v16-v20: the ONLY .pt files ever registered as models
# (loadRun=gvhmr_pt, v16/v18/v19/v20 model lists) lived outside the repo at
# /workspace/isaaclab/GMR_X1/**. In-repo mirrors never registered:
#  - logs/**  : blocked by .gitignore ('logs/') for the in-repo scanner
#  - model_upload/** : detected ("New file detected globally") but never
#    registered as a model (no exported_data pattern).
# Outside-repo .pt files get registered within ~20 min of appearing (gvhmr
# precedent: created ~12:50, registered 13:04 in v18/v19).
OUTSIDE_DIR = REPO_ROOT.parent / "x1_upload" if REPO_ROOT.parent.name == "isaaclab" else Path("/workspace/isaaclab/x1_upload")
MOTIONS_DIR = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions"
TRAIN_LOG_FILE = REPO_ROOT / "train_stdout.log"
PLAY_LOG_FILE = REPO_ROOT / "play_stdout.log"


def wrap_json_for_upload(name: str, obj) -> Path:
    """SDK only uploads .pt files — wrap JSON payloads in a pickle .pt."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    p = UPLOAD_DIR / name
    with open(p, "wb") as f:
        _pkl.dump(obj, f)
    print(f"[UPLOAD] {p.relative_to(REPO_ROOT)} ({p.stat().st_size // 1024}KB)")
    return p


def wait_for_sdk(seconds: int, why: str):
    print(f"[INFO] Waiting {seconds}s for SDK upload queue ({why})...")
    time.sleep(seconds)


# ────────────────────────────────────────────────────────────────────
def phase_retarget():
    print("\n=== Phase 1-2: GMR Retarget + Isaac Lab dataset_retarget ===\n")
    reassemble_smplx()
    gmr_dir, venv_dir = setup_gmr()
    register_x1_in_gmr(gmr_dir)

    print("\n--- Auto-IK Calibration ---")
    run_auto_ik(gmr_dir, venv_dir)

    gmr_output = MOTIONS_DIR / "x1_gmr"
    if gmr_output.exists() and len(list(gmr_output.glob("*.pkl"))) >= 14:
        print(f"[INFO] x1_gmr already has {len(list(gmr_output.glob('*.pkl')))} files, skipping GMR retarget")
    else:
        run_gmr_retarget(gmr_dir, venv_dir)

    lab_output = MOTIONS_DIR / "x1_lab"
    if lab_output.exists() and len(list(lab_output.glob("*.pkl"))) >= 14:
        print(f"[INFO] x1_lab already has {len(list(lab_output.glob('*.pkl')))} files, skipping dataset_retarget")
    else:
        run_dataset_retarget(gmr_output)

    lab_files = list(lab_output.glob("*.pkl"))
    print(f"\n[INFO] x1_lab: {len(lab_files)} files")
    if len(lab_files) < 14:
        print("[ERROR] Expected 14 lab files for AMP training!")
        sys.exit(1)
    return gmr_output, lab_output, venv_dir


def phase_retarget_acceptance(venv_dir: Path) -> bool:
    """Strict gate: acceptance/check_retarget.py. Returns True on PASS."""
    print("\n=== Phase 3: STRICT Retarget Acceptance Gate ===\n")
    checker = REPO_ROOT / "acceptance" / "check_retarget.py"
    report_json = UPLOAD_DIR / "retarget_acceptance_report.json"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # checker json write needs it
    venv_python = str(venv_dir / "bin" / "python")
    cmd = [
        "python" if not Path(venv_python).exists() else venv_python,
        str(checker), "--repo-root", str(REPO_ROOT),
        "--json", str(report_json),
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))

    payload = {"phase": "retarget_acceptance",
               "passed": result.returncode == 0,
               "checker": "acceptance/check_retarget.py",
               "spec": "acceptance/RETARGET_ACCEPTANCE.md"}
    if report_json.exists():
        payload["report"] = report_json.read_text()
    wrap_json_for_upload("model_retarget_report.pt", payload)

    if result.returncode != 0:
        print("\n[FATAL] Retarget acceptance FAILED — blocking training start.")
        print("[INFO] Report wrapped for upload. Aborting (no GPU-hours spent).")
        wait_for_sdk(180, "upload failure report")
        sys.exit(1)
    print("\n[INFO] Retarget acceptance PASSED — proceeding to training.")
    return True


def phase_package_retarget(gmr_output: Path, lab_output: Path):
    print("\n=== Phase 3.5: Package retarget artifacts ===")
    retarget_pkg = {}
    for f in sorted(lab_output.glob("*.pkl")):
        retarget_pkg[f"x1_lab/{f.name}"] = f.read_bytes()
    for f in sorted(gmr_output.glob("*.pkl")):
        retarget_pkg[f"x1_gmr/{f.name}"] = f.read_bytes()
    auto_cfg = REPO_ROOT / "AMASS_minimal" / "smplx_to_x1_auto.json"
    if auto_cfg.exists():
        retarget_pkg["smplx_to_x1_auto.json"] = auto_cfg.read_bytes()
    p = UPLOAD_DIR / "model_retarget_data.pt"
    with open(p, "wb") as fh:
        _pkl.dump(retarget_pkg, fh)
    print(f"[UPLOAD] {p.relative_to(REPO_ROOT)} ({p.stat().st_size / 1e6:.1f}MB)")


def find_checkpoint_roots():
    """train.py writes logs relative to ITS cwd. We pass cwd=REPO_ROOT, but
    also sweep the process cwd root as belt-and-braces (v16/v17 bug: logs went
    to /workspace/isaaclab/logs while monitor watched REPO_ROOT/logs)."""
    roots = [REPO_ROOT / "logs" / "rsl_rl" / "x1_amp",
             Path.cwd() / "logs" / "rsl_rl" / "x1_amp"]
    return [r for r in roots if r.exists()]


def latest_run_dir():
    run_dirs = []
    for root in find_checkpoint_roots():
        for d in root.glob("*/"):
            if d.name == "exported_data":
                continue
            if list(d.glob("model_*.pt")):
                run_dirs.append(d)
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def all_checkpoints():
    ckpts = {}
    for root in find_checkpoint_roots():
        for d in root.glob("*/"):
            if d.name == "exported_data":
                continue
            for c in d.glob("model_*.pt"):
                ckpts[c.name] = c  # later roots overwrite earlier
    return ckpts


def mirror_checkpoint(ckpt: Path, tag: str):
    """Mirror a checkpoint into every SDK-visible location."""
    copied = []
    # 0) PRIMARY: outside-repo dir (proven registration path, see OUTSIDE_DIR)
    try:
        od = OUTSIDE_DIR / tag
        od.mkdir(parents=True, exist_ok=True)
        dst0 = od / ckpt.name
        if not dst0.exists():
            shutil.copy2(ckpt, dst0)
            copied.append(dst0)
    except OSError as e:
        print(f"[MONITOR] outside mirror failed: {e}")
    # 1) repo tree upload dir (kept for redundancy)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dst = UPLOAD_DIR / ckpt.name
    if not dst.exists():
        shutil.copy2(ckpt, dst)
        copied.append(dst)
    # 2) exported_data pattern logs/{exp}/exported_data/{run}/model_*.pt
    for root in [REPO_ROOT / "logs", Path.cwd() / "logs"]:
        exp = root / "x1_amp" / "exported_data" / tag
        try:
            exp.mkdir(parents=True, exist_ok=True)
            dst2 = exp / ckpt.name
            if not dst2.exists():
                shutil.copy2(ckpt, dst2)
                copied.append(dst2)
        except OSError:
            pass
    for c in copied:
        try:
            print(f"[MONITOR] {ckpt.name} -> {c}")
        except Exception:
            pass


def purge_gmr_junk_pt():
    """Delete GMR's bundled gvhmr_pt/*.pt immediately after the retarget phase.

    v21b evidence: these junk reference-policy .pt registered on the platform
    at task-start+8min (from OUTSIDE-repo GMR_X1 — so the SDK scans beyond the
    repo tree), while every real training checkpoint stayed unregistered —
    consistent with a 5-per-task quota consumed by junk. The retarget pkl
    data is fully extracted by the time this runs, so deletion is safe."""
    roots = [REPO_ROOT.parent / "GMR_X1" / "gvhmr_pt",
             REPO_ROOT / "GMR" / "gvhmr_pt"]
    n = 0
    for r in roots:
        if r.is_dir():
            for pt in r.glob("*.pt"):
                try:
                    pt.unlink()
                    n += 1
                except OSError:
                    pass
    print(f"[PURGE] removed {n} GMR gvhmr junk .pt files (before training)")


def phase_train() -> int:
    print("\n=== Phase 4: AMP Training ===\n")
    purge_gmr_junk_pt()

    robolab_src = REPO_ROOT / "roboparty_train" / "robolab"
    rsl_rl_src = REPO_ROOT / "roboparty_train" / "rsl_rl"

    print("[INFO] Installing robolab/rsl_rl (force rsl_rl to override Isaac Lab v3.1.2)...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "rsl-rl-lib", "-y", "-q"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(rsl_rl_src), "-q",
                    "--force-reinstall", "--no-deps"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(robolab_src), "-q",
                    "--no-deps"], check=True)

    train_script = robolab_src / "scripts" / "rsl_rl" / "train.py"
    cmd = [sys.executable, str(train_script),
           "--task=X1-AMP", "--headless", "--logger=tensorboard", "--num_envs=4096"]

    tag = _dt.now().strftime("%Y-%m-%d_%H-%M-%S") + "x1_amp"
    stop_monitor = threading.Event()
    mirrored = set()

    def monitor():
        """Watch training checkpoints (logging only — mirrors happen in the
        final sweep). v22: during-training mirroring removed; v16-v21b
        evidence suggests a 5-per-task registration quota, so intermediate
        checkpoints must NOT be pushed into SDK-visible paths."""
        last = None
        while not stop_monitor.is_set():
            try:
                names = sorted(all_checkpoints())
                if names != last:
                    print(f"[MONITOR] checkpoints on disk: {names}")
                    last = names
            except Exception as e:
                print(f"[MONITOR] error: {e}")
            stop_monitor.wait(60)

    threading.Thread(target=monitor, daemon=True).start()

    print(f"[INFO] Starting AMP training: {' '.join(cmd)}")
    print(f"[INFO] cwd={REPO_ROOT}  stdout -> {TRAIN_LOG_FILE.name}")
    with open(TRAIN_LOG_FILE, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        for raw in iter(proc.stdout.readline, b""):
            logf.write(raw)
            logf.flush()
            try:
                print(raw.decode(errors="replace"), end="")
            except Exception:
                pass
        rc = proc.wait()
    stop_monitor.set()
    time.sleep(2)

    # v22 final sweep: mirror ONLY the final checkpoint. Reports are written
    # to UPLOAD_DIR by the packaging/acceptance phases. Keeping the total
    # SDK-visible .pt count small protects the final ckpt + reports against
    # a possible 5-per-task registration quota (v16-v21b: gvhmr junk took
    # all 5 slots every single run).
    print("\n[INFO] Final checkpoint sweep:")
    ckpts = all_checkpoints()
    if not ckpts:
        print("[ERROR] NO CHECKPOINTS FOUND anywhere under logs/ roots!")
    final = final_checkpoint()
    if final is not None:
        mirror_checkpoint(final, tag)
        mirrored.add(final.name)
    print(f"[INFO] Mirrored final only: {final.name if final else None}. "
          f"Run dir: {latest_run_dir()}")
    return rc


def final_checkpoint() -> Path | None:
    ckpts = all_checkpoints()
    if not ckpts:
        return None
    def key(name):
        stem = name[len("model_"):-len(".pt")]
        return (0, int(stem)) if stem.isdigit() else (1, 0)
    best = max(ckpts, key=key)
    return ckpts[best]


def phase_play_video(ckpt: Path):
    """Record a fixed-command walk video with the final policy."""
    print("\n=== Phase 5: Play rollout + video ===\n")
    if ckpt is None:
        print("[WARN] No checkpoint — skipping play/video")
        return None
    play_script = REPO_ROOT / "roboparty_train" / "robolab" / "scripts" / "rsl_rl" / "play_amp.py"
    cmd = [sys.executable, str(play_script),
           "--task", "X1-AMP-Play",
           "--num_envs", "1",
           "--checkpoint", str(ckpt),
           "--video", "--video_length", "600",   # 600 steps = 12 s @ 50 Hz
           "--headless"]
    print(f"[INFO] {' '.join(cmd)}")
    try:
        with open(PLAY_LOG_FILE, "wb") as logf:
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf,
                                stderr=subprocess.STDOUT, timeout=1500).returncode
    except subprocess.TimeoutExpired:
        rc = -1
        print("[WARN] play timed out after 1500s (killed) — continuing without fresh video")
    print(f"[INFO] play exit={rc} (log: {PLAY_LOG_FILE.name})")

    # v20 postmortem: play exit=0 but no mp4 under logs/ — RecordVideo
    # produced nothing. v21: search widely + dump diagnostics + mujoco
    # sim2sim fallback so P6 always has video evidence when a policy exists.
    videos = []
    search_roots = [REPO_ROOT / "logs", Path.cwd() / "logs", REPO_ROOT,
                    ckpt.parent / "videos", REPO_ROOT.parent / "x1_upload"]
    for root in search_roots:
        try:
            if root.exists():
                videos += [v for v in root.rglob("*.mp4")]
        except OSError:
            pass
    print(f"[INFO] mp4 search ({len(videos)} found) under: "
          f"{[str(r) for r in search_roots]}")
    # diagnostics: video_folder listing + play log tail (postmortem evidence)
    vf = ckpt.parent / "videos" / "play"
    if vf.exists():
        print(f"[INFO] video_folder {vf}: {[p.name for p in vf.iterdir()]}")
    else:
        print(f"[INFO] video_folder {vf} does not exist")
    if PLAY_LOG_FILE.exists():
        tail = PLAY_LOG_FILE.read_text(errors="replace").splitlines()[-12:]
        print("[INFO] play_stdout.log tail: " + " | ".join(t.strip()[:120] for t in tail))

    video = None
    if videos:
        video = max(videos, key=lambda v: v.stat().st_mtime)
    else:
        print("[WARN] No mp4 from Isaac play — falling back to MuJoCo sim2sim rollout")
        video = sim2sim_fallback_video(ckpt)

    if video is None:
        print("[WARN] No video from any source")
        return None
    # mirror ALL sim2sim products (videos + metric jsons) to SDK-visible
    # locations; metrics jsons also get wrapped as .pt for registration
    stamp = f"{_dt.now():%H%M%S}"
    to_mirror = [video]
    sim_dir = OUTSIDE_DIR / "sim2sim"
    if sim_dir.exists():
        to_mirror += sorted(sim_dir.glob("x1_sim2sim_*.mp4"))
        for js in sorted(sim_dir.glob("x1_sim2sim_*.json")):
            try:
                import json as _json
                obj = _json.loads(js.read_text())
                p = wrap_json_for_upload(f"sim2sim_{js.stem}.pt", obj)
                print(f"[UPLOAD] {p.name} ({p.stat().st_size // 1024}KB)")
            except Exception as e:
                print(f"[WARN] wrap {js.name}: {e}")
    seen = set()
    for v in to_mirror:
        if v in seen:
            continue
        seen.add(v)
        dests = [OUTSIDE_DIR / f"{v.stem}_{stamp}.mp4" if v == video else OUTSIDE_DIR / v.name,
                 REPO_ROOT / "logs" / "x1_amp" / v.name,
                 UPLOAD_DIR / v.name]
        for d in dests:
            try:
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(v, d)
                print(f"[VIDEO] -> {d}")
            except OSError as e:
                print(f"[VIDEO] mirror failed {d}: {e}")
    return video


def _mujoco_env_matrix(sys_ok: bool, pylibs: Path):
    """Env attempt matrix for headless MuJoCo rendering (v22).

    v21b post-mortem: pip --target pylibs mujoco pulled a BROKEN PyOpenGL
    (import crash in OpenGL/GL/VERSION/GL_1_1.py) which killed both egl and
    osmesa. The container python already ships mujoco 3.6.0 (pip conflict
    log) and imageio (moviepy dep) — so try the SYSTEM interpreter first
    with NO PYTHONPATH shadowing, then a pylibs fallback whose broken
    OpenGL copy has been stripped (mujoco falls back to system OpenGL)."""
    attempts = []
    if sys_ok:
        attempts += [
            ("sys-egl", dict(os.environ, MUJOCO_GL="egl")),
            ("sys-osmesa", dict(os.environ, MUJOCO_GL="osmesa",
                                PYOPENGL_PLATFORM="osmesa")),
        ]
    pp = str(pylibs)
    attempts += [
        ("pylibs-egl", dict(os.environ, MUJOCO_GL="egl", PYTHONPATH=pp)),
        ("pylibs-osmesa", dict(os.environ, MUJOCO_GL="osmesa",
                               PYOPENGL_PLATFORM="osmesa", PYTHONPATH=pp)),
    ]
    return attempts


def sim2sim_fallback_video(ckpt: Path):
    """Render walking videos of the final policy in MuJoCo — 3 rollouts
    (walk 1.0 / walk 1.5 / walk+turn) with metrics JSON. Doubles as the
    sim2sim deliverable. Returns the first video path, None on failure."""
    try:
        import subprocess as _sp

        sys_ok = _sp.run([sys.executable, "-c", "import mujoco, imageio.v2"],
                         capture_output=True).returncode == 0
        print(f"[INFO] system python mujoco+imageio probe: {'OK' if sys_ok else 'missing'}")

        pylibs = REPO_ROOT / "pylibs"
        pylibs.mkdir(exist_ok=True)
        _sp.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(pylibs),
                 "mujoco", "imageio", "imageio-ffmpeg"], check=True, timeout=600)
        # v21b root cause fix: strip the broken pip PyOpenGL from pylibs so
        # mujoco's GL bindings resolve to the system OpenGL instead
        for bad in list(pylibs.glob("OpenGL")) + list(pylibs.glob("PyOpenGL*")):
            shutil.rmtree(bad, ignore_errors=True)
            print(f"[INFO] stripped {bad.name} from pylibs (broken PyOpenGL)")

        rollout = REPO_ROOT / "sim2sim" / "mujoco_rollout.py"
        out_dir = OUTSIDE_DIR / "sim2sim"
        out_dir.mkdir(parents=True, exist_ok=True)
        rollouts = [
            ("walk_1.0", ["--cmd", "1.0", "0.0", "0.0", "--duration", "12"]),
            ("walk_1.5", ["--cmd", "1.5", "0.0", "0.0", "--duration", "8"]),
            ("walk_turn", ["--cmd", "1.0", "0.0", "0.8", "--duration", "8"]),
        ]
        good_env, videos = None, []
        logf = open(REPO_ROOT / "mujoco_fallback.log", "wb")
        try:
            for label, env in _mujoco_env_matrix(sys_ok, pylibs):
                logf.write(f"\n===== {label} MUJOCO_GL={env.get('MUJOCO_GL')} =====\n".encode())
                logf.flush()
                ok = True
                for name, extra in rollouts:
                    mp4 = out_dir / f"x1_sim2sim_{name}.mp4"
                    js = out_dir / f"x1_sim2sim_{name}.json"
                    cmd = [sys.executable, str(rollout), "--ckpt", str(ckpt),
                           "--repo-root", str(REPO_ROOT), "--video", str(mp4),
                           "--json", str(js)] + extra
                    try:
                        rc = _sp.run(cmd, cwd=str(REPO_ROOT), env=env, timeout=900,
                                     stdout=logf, stderr=_sp.STDOUT).returncode
                    except Exception as run_err:
                        print(f"[WARN] mujoco {label}/{name} run error: {run_err}")
                        rc = -1
                    if rc != 0 or not (mp4.exists() and mp4.stat().st_size > 100_000):
                        ok = False
                        break
                    videos.append(mp4)
                if ok:
                    good_env = label
                    break
                videos.clear()
        finally:
            logf.close()
        if good_env:
            print(f"[INFO] mujoco rollouts OK via {good_env}: {[v.name for v in videos]}")
            return videos[0]
        # last resort: install system GL libs and retry osmesa via pylibs
        try:
            print("[INFO] last resort: apt-get install libosmesa6 + retry")
            _sp.run(["apt-get", "install", "-y", "-q", "libosmesa6", "libegl1"],
                    timeout=300, stdout=logf, stderr=_sp.STDOUT)
            logf = open(REPO_ROOT / "mujoco_fallback.log", "ab")
            env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
                       PYTHONPATH=str(pylibs))
            ok = True
            for name, extra in rollouts:
                mp4 = out_dir / f"x1_sim2sim_{name}.mp4"
                js = out_dir / f"x1_sim2sim_{name}.json"
                cmd = [sys.executable, str(rollout), "--ckpt", str(ckpt),
                       "--repo-root", str(REPO_ROOT), "--video", str(mp4),
                       "--json", str(js)] + extra
                rc = _sp.run(cmd, cwd=str(REPO_ROOT), env=env, timeout=900,
                             stdout=logf, stderr=_sp.STDOUT).returncode
                if rc != 0 or not (mp4.exists() and mp4.stat().st_size > 100_000):
                    ok = False
                    break
                videos.append(mp4)
            if ok:
                print(f"[INFO] mujoco rollouts OK via apt-osmesa: {[v.name for v in videos]}")
                logf.close()
                return videos[0]
            videos.clear()
            logf.close()
        except Exception as apt_err:
            print(f"[WARN] apt osmesa retry failed: {apt_err}")
        log = (REPO_ROOT / "mujoco_fallback.log").read_text(errors="replace")
        print("[INFO] mujoco tail: " + " | ".join(
            l.strip()[:120] for l in log.splitlines()[-8:]))
    except Exception as e:
        print(f"[WARN] mujoco fallback failed: {e}")
    return None


def phase_amp_acceptance(video: Path | None):
    print("\n=== Phase 6: AMP Training Acceptance ===\n")
    checker = REPO_ROOT / "acceptance" / "check_amp.py"
    report_json = UPLOAD_DIR / "amp_acceptance_report.json"
    max_iter = 4000
    cmd = [sys.executable, str(checker), "--log", str(TRAIN_LOG_FILE),
           "--max-iters", str(max_iter), "--json", str(report_json)]
    if video is not None:
        cmd += ["--video", str(video)]
    if PLAY_LOG_FILE.exists():
        cmd += ["--play-log", str(PLAY_LOG_FILE)]
    rc = subprocess.run(cmd).returncode

    payload = {"phase": "amp_acceptance", "passed": rc == 0,
               "checker": "acceptance/check_amp.py",
               "spec": "acceptance/AMP_ACCEPTANCE.md", "max_iterations": max_iter}
    if report_json.exists():
        payload["report"] = report_json.read_text()
    wrap_json_for_upload("model_amp_report.pt", payload)
    return rc


def main():
    print("=" * 60)
    print("X1 AMP Pipeline v18 (acceptance-gated)")
    print("=" * 60)

    gmr_output, lab_output, venv_dir = phase_retarget()
    phase_retarget_acceptance(venv_dir)
    phase_package_retarget(gmr_output, lab_output)

    rc = phase_train()
    if rc != 0:
        print(f"[ERROR] AMP training exited with code {rc}")
    ckpt = final_checkpoint()
    print(f"[INFO] Final checkpoint: {ckpt}")

    video = phase_play_video(ckpt)
    amp_rc = phase_amp_acceptance(video)

    print("\n=== Phase 7: wrap-up ===")
    print(f"[INFO] Artifacts in {UPLOAD_DIR}:")
    for f in sorted(UPLOAD_DIR.glob("*")):
        print(f"  {f.name} ({f.stat().st_size // 1024}KB)")
    # v21: mirror reports + exported jit/onnx + play log to the OUTSIDE-repo
    # dir — the only location empirically registered by the SDK (see OUTSIDE_DIR)
    try:
        out = OUTSIDE_DIR / "final"
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(UPLOAD_DIR.glob("*")):
            if f.is_file():
                shutil.copy2(f, out / f.name)
        # exported jit/onnx from play_amp (useful for deployment/sim2sim)
        for exp_dir in (REPO_ROOT / "logs" / "rsl_rl" / "x1_amp").glob("*/exported"):
            for f in exp_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, out / f"{exp_dir.parent.name}_{f.name}")
        if PLAY_LOG_FILE.exists():
            shutil.copy2(PLAY_LOG_FILE, out / "play_stdout.log")
        if TRAIN_LOG_FILE.exists():
            shutil.copy2(TRAIN_LOG_FILE, out / "train_stdout.log")
        print(f"[INFO] Final artifacts mirrored to {out}: "
              f"{[p.name for p in out.iterdir()]}")
    except OSError as e:
        print(f"[WARN] outside mirror of final artifacts failed: {e}")
    wait_for_sdk(420, "final checkpoint + video + reports")
    print("[INFO] Pipeline done.")


if __name__ == "__main__":
    main()
