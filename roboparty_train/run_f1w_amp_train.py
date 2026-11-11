#!/usr/bin/env python3
"""F1W-AMP RUN training pipeline v1 (weilai F1 29DOF, acceptance-gated).

F1w = weilai F1 (SYS-SPEC-0002-V1.1, 35.033 kg mass-merged, ankle
241.9/218.4 Nm = 3x X1). Dataset f1w_lab_v1 (54 clips) retargeted LOCALLY
(AMASS run sources CMU 09/35 sprint + ACCAD transitions + CMU 08_x jog +
v72 walk anchors -> GMR X1 chain v30/v31/fix_root_xy_run ->
retarget_f1w_from_x1lab target-shrink projection; gates: check_retarget
strict A-G PASS on the X1 pair + check_f1w_retarget R1/R2/R3 on the F1w
library). This pod pipeline starts at training:

  1. Dataset gate: f1w_lab_v1 (52 clips, v7: 82_12 pair excluded) + f1w assets
  2. AMP training: F1W-AMP @ 50 Hz, 4096 envs, PYTHONPATH shadowing
  3. Checkpoint monitor: SDK-visible mirrors (.bin stealth rename on resume)
  4. Play rollout + video (F1W-AMP-Play @ F1W_PLAY_VEL, default 1.6 m/s)

Env vars: F1W_ITERS (fresh budget, default 6000), F1W_RESUME_CKPT,
F1W_FINE_TUNE_ITERS, F1W_PLAY_VEL; X1_* recipe levers via f1w_env.txt
(startScript cannot pass env vars to gm-run).

Usage (startScript): gm-run X1_29_AMP/roboparty_train/run_f1w_amp_train.py
"""

import functools
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = None
for candidate in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent]:
    if (candidate / "roboparty_train").is_dir():
        REPO_ROOT = candidate
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

EXP = "f1w_amp"
MOTIONS = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions" / os.environ.get("F1W_MOTION_DIR", "f1w_lab_v3")
ROBOT = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "robots" / "f1w"
UPLOAD_DIR = REPO_ROOT / "model_upload"
OUTSIDE_DIR = REPO_ROOT.parent / "f1w_upload" if REPO_ROOT.parent.name == "isaaclab" else Path("/workspace/isaaclab/f1w_upload")
TRAIN_LOG = REPO_ROOT / "train_stdout.log"
PLAY_LOG = REPO_ROOT / "play_stdout.log"
TAG = time.strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------- gates
def phase_dataset_gate() -> bool:
    print("\n=== Phase 1: dataset + assets gate ===")
    pkls = sorted(MOTIONS.glob("*.pkl"))
    expected = int(os.environ.get("F1W_N_CLIPS", "4"))
    ok = len(pkls) == expected
    print(f"[{'OK' if ok else 'FAIL'}] {MOTIONS.name} clips: {len(pkls)}/{expected}")
    for p in pkls[:3]:
        try:
            c = pickle.load(open(p, "rb"))
            shape_ok = (len(c["dof_pos"].shape) == 2 and c["dof_pos"].shape[1] == 29
                        and c["key_body_pos"].shape[1] == 10 and 30 <= c["fps"] <= 250)
            print(f"  {p.name}: fps={c['fps']} T={len(c['dof_pos'])} "
                  f"dof={c['dof_pos'].shape[1]} kb={c['key_body_pos'].shape[1]} "
                  f"[{'OK' if shape_ok else 'FAIL'}]")
            ok = ok and shape_ok
        except Exception as e:
            print(f"  {p.name}: LOAD FAIL {e}")
            ok = False
    # weights <-> file bijection (TASK_20261110_107 crashed at env build:
    # 12 stale weight keys pointed at nonexistent clips)
    import re as _re
    cfg_src = (REPO_ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks" /
               "manager_based" / "amp" / "f1w_amp_env_cfg.py").read_text()
    m_w = _re.search(r"motion_data_weights\s*=\s*\{(.*?)\}", cfg_src, _re.S)
    keys = set(_re.findall(r'"([^"]+)":', m_w.group(1))) if m_w else set()
    names = {p_.stem for p_ in pkls}
    stale = keys - names
    unweighted = names - keys
    ok = ok and not stale and not unweighted
    print(f"[{'OK' if not stale and not unweighted else 'FAIL'}] weights<->files "
          f"bijection: {len(keys)} keys / {len(names)} files; "
          f"stale={sorted(stale)[:4]} unweighted={sorted(unweighted)[:4]}")

    urdf = ROBOT / "urdf" / "f1w.urdf"
    print(f"[{'OK' if urdf.exists() else 'FAIL'}] F1 urdf: {urdf.exists()}")
    ok = ok and urdf.exists()
    n_mesh = len(list((ROBOT / "meshes").glob("*.STL"))) + len(list((ROBOT / "meshes").glob("*.stl")))
    print(f"[{'OK' if n_mesh >= 60 else 'FAIL'}] F1 meshes: {n_mesh}")
    ok = ok and n_mesh >= 60
    # env import check (cheap: file presence; real import needs isaac)
    for f in ["roboparty_train/robolab/robolab/assets/robots/f1w.py",
              "roboparty_train/robolab/robolab/tasks/manager_based/amp/f1w_amp_env_cfg.py",
              "roboparty_train/robolab/robolab/tasks/manager_based/amp/agents/f1w_amp_agent_cfg.py"]:
        p = REPO_ROOT / f
        print(f"[{'OK' if p.exists() else 'FAIL'}] {Path(f).name}")
        ok = ok and p.exists()
    if not ok:
        print("[FATAL] dataset gate FAIL — aborting BEFORE burning GPU-hours")
    return ok


# ------------------------------------------------------------- checkpoints
def find_checkpoint_roots():
    roots = [REPO_ROOT / "logs" / "rsl_rl" / EXP,
             Path.cwd() / "logs" / "rsl_rl" / EXP]
    return [r for r in roots if r.exists()]


def all_checkpoints():
    ckpts = {}
    for root in find_checkpoint_roots():
        for d in root.glob("*/"):
            if d.name == "exported_data":
                continue
            for c in d.glob("model_*.pt"):
                ckpts[c.name] = c
    return ckpts


def wait_for_sdk(seconds: int, why: str):
    """Post-mirror wait: the SDK scans globally every ~16 min and the FINAL
    checkpoint appears right before task exit — v1d (TASK_20261010_133) lost
    model_4000..5498 to exactly this (detected 19:10:13, pod died before
    upload; the mid-run 091 uploads only succeeded because its run was long
    enough to span scan cycles)."""
    print(f"[WAIT] {why}: sleeping {seconds}s for the SDK upload cycle...")
    time.sleep(seconds)


def mirror_checkpoint(ckpt: Path, tag: str):
    """Mirror the checkpoint to every SDK-visible location.

    v1e forensics (TASK_20261010_178): the SDK 'Found PT file' uploader only
    ever queued logs/rsl_rl/f1_amp/_resume_src/model_3999.pt — the run-dir /
    checkpoints/ / model_upload/ / repo-flat / outside-tree mirrors were all
    merely 'detected globally', never uploaded. The skill-documented PROVEN
    channel (8-round probe matrix) is output/{unique-name}.pt at the TOP
    level of the task output dir — added here as the PRIMARY."""
    copied = []
    isaaclab = REPO_ROOT.parent if REPO_ROOT.parent.name == "isaaclab" else Path("/workspace/isaaclab")
    for dst in [isaaclab / "output" / ckpt.name,
                REPO_ROOT / "output" / ckpt.name,
                REPO_ROOT / "roboparty_train" / "checkpoints" / ckpt.name,
                UPLOAD_DIR / ckpt.name,
                REPO_ROOT / ckpt.name,
                REPO_ROOT / "logs" / "rsl_rl" / EXP / "exported_data" / tag / ckpt.name,
                OUTSIDE_DIR / tag / ckpt.name]:
        try:
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ckpt, dst)
                copied.append(dst)
        except OSError as e:
            print(f"[MONITOR] mirror failed {dst}: {e}")
    for c in copied:
        print(f"[MONITOR] {ckpt.name} -> {c}")


def final_checkpoint():
    cks = all_checkpoints()

    def num(p):
        s = p.stem[len("model_"):]
        return int(s) if s.isdigit() else -1
    if not cks:
        return None
    return max(cks.values(), key=lambda p: num(p))


# ----------------------------------------------------------------- train
def phase_train() -> int:
    print("\n=== Phase 2: AMP training (F1W-AMP @ 50 Hz) ===")
    robolab_src = REPO_ROOT / "roboparty_train" / "robolab"
    rsl_rl_src = REPO_ROOT / "roboparty_train" / "rsl_rl"
    parts = [str(rsl_rl_src), str(robolab_src)]
    old = os.environ.get("PYTHONPATH", "")
    if old:
        parts.append(old)
    os.environ["PYTHONPATH"] = ":".join(parts)
    print(f"[INFO] PYTHONPATH shadowing (no pip): {os.environ['PYTHONPATH']}")

    train_script = robolab_src / "scripts" / "rsl_rl" / "train.py"
    num_envs = os.environ.get("F1W_NUM_ENVS", "4096")
    cmd = [sys.executable, str(train_script),
           "--task=F1W-AMP", "--headless", "--logger=tensorboard", "--num_envs=" + str(num_envs)]
    iters = os.environ.get("F1_ITERS", "")
    if iters and not os.environ.get("F1_RESUME_CKPT", "").strip():
        cmd += ["--max_iterations", iters]

    resume_ckpt = "" if os.environ.get("F1W_FRESH", "0") == "1" else os.environ.get("F1_RESUME_CKPT", "").strip()
    if os.environ.get("F1W_FRESH", "0") == "1":
        print("[FRESH] F1W_FRESH=1: resume detection DISABLED (checkpoints/ dir "
              "permanently holds X1-era + phase artifacts; TASK_20261112_018 "
              "resumed a stale 288-dim X1 ckpt and crashed on this exact gap)")
    if not resume_ckpt and os.environ.get("F1W_FRESH", "0") != "1":
        import re as _re
        def _num(p):
            # platform-mounted names: model_1500_20261110185133A467.pt
            m = _re.match(r"model_(\d+)", p.stem)
            return int(m.group(1)) if m else -1
        cands = sorted((p for p in list(REPO_ROOT.glob("model_*.pt"))
                        + list((REPO_ROOT / "roboparty_train" / "checkpoints").glob("model_*.pt"))
                        if _num(p) >= 0), key=_num)
        # v2n/v2n2 lesson (TASK_20261111_274/298): the platform
        # checkPointFilePath mount NEVER materialized (600s poll exhausted)
        # — phase-B resumes now ship the checkpoint IN THE REPO
        # (roboparty_train/checkpoints/, established 60MB-artifact practice)
        # so the clone delivers it atomically. F1W_RESUME_WAIT_SEC remains
        # as a safety net for platform mounts.
        _wait_s = int(os.environ.get("F1W_RESUME_WAIT_SEC", "0"))
        _polled = 0
        while not cands and _wait_s > 0 and _polled < _wait_s:
            print(f"[RESUME-WAIT] no mounted checkpoint yet; polling "
                  f"({_polled}/{_wait_s}s)...")
            time.sleep(30)
            _polled += 30
            cands = sorted((p for p in list(REPO_ROOT.glob("model_*.pt"))
                            + list((REPO_ROOT / "roboparty_train" / "checkpoints").glob("model_*.pt"))
                            if _num(p) >= 0), key=_num)
        if cands:
            resume_ckpt = str(cands[-1])
            print(f"[RESUME] auto-detected mounted checkpoint: {resume_ckpt}")
    if resume_ckpt:
        src = Path(resume_ckpt)
        if not src.is_file():
            raise FileNotFoundError(f"F1_RESUME_CKPT not found: {src}")
        resume_run = REPO_ROOT / "logs" / "rsl_rl" / EXP / "_resume_src"
        resume_run.mkdir(parents=True, exist_ok=True)
        # v1g lesson: NEVER let a scan-visible extension (.pt/.pth/.ckpt)
        # sit under _resume_src — the SDK PT-uploader LATCHES onto the first
        # .pt-containing directory and ignores the training run dir forever
        # (v1d/e/f starved); deleting it later CRASHES the scanner (v1g,
        # Errno 2 at 23:17:38 -> task killed). train.py's --checkpoint is a
        # REGEX against filenames — a .bin name is invisible to the SDK but
        # resolves fine for resume.
        dst = resume_run / (src.stem + ".bin")
        shutil.copy2(src, dst)
        cmd += ["--resume", "--load_run", "_resume_src",
                "--checkpoint", src.stem + ".bin",  # regex; src.stem renamed .bin above
                "--max_iterations", os.environ.get("F1_FINE_TUNE_ITERS", "1500")]
        print(f"[RESUME] fine-tune from {src.name} (+.bin stealth rename) "
              f"+{os.environ.get('F1_FINE_TUNE_ITERS', '1500')}")

    stop_monitor = threading.Event()
    mirrored = set()

    def monitor():
        last = None
        while not stop_monitor.is_set():
            try:
                names = sorted(all_checkpoints())
                if names != last:
                    print(f"[MONITOR] checkpoints on disk: {names}")
                    last = names
                # v1d/v1e/v1f: _resume_src with a .pt starved the run dir of
                # SDK uploads (single-dir claim latch). Fixed upstream by the
                # .bin stealth rename — nothing to clean here; DELETING the
                # dir mid-run crashes the SDK scanner (v1g).
                for nm, ck in sorted(all_checkpoints().items()):
                    if nm.startswith("model_") and nm not in mirrored:
                        print(f"[MONITOR] new checkpoint mirror: {nm}")
                        mirror_checkpoint(ck, TAG)
                        mirrored.add(nm)
            except Exception as e:
                print(f"[MONITOR] error: {e}")
            # visibility (TASK_20261110_121 hung 95min with zero platform
            # log lines): echo the train_stdout.log tail every monitor cycle
            try:
                if TRAIN_LOG.exists():
                    tail = TRAIN_LOG.read_text(errors="replace").splitlines()[-3:]
                    print(f"[STDOUT-TAIL] {' | '.join(t[-160:] for t in tail)}")
            except Exception:
                pass
            stop_monitor.wait(60)

    threading.Thread(target=monitor, daemon=True).start()

    print(f"[INFO] {' '.join(cmd)}")
    print(f"[INFO] cwd={REPO_ROOT}  stdout -> {TRAIN_LOG.name}")
    with open(TRAIN_LOG, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            print(f"[TRAIN] {line}", file=logf)
            logf.flush()
            # echo reward lines for live platform-log visibility
            if "mean_reward" in line or "iteration" in line.lower()[:20]:
                print(f"[TRAIN] {line[-160:]}")
        rc = proc.wait()
    stop_monitor.set()
    print(f"[INFO] training exit={rc} (full log: {TRAIN_LOG.name})")
    if rc != 0 and TRAIN_LOG.exists():
        try:
            tail = TRAIN_LOG.read_text(errors="replace").splitlines()[-80:]
            print("[TRAIN-FAIL-TAIL] " + "\n".join(tail))
        except Exception:
            pass
    return rc


# ------------------------------------------------------------------ play
def phase_play_video(ckpt):
    print("\n=== Phase 3: Play rollout + video ===")
    if ckpt is None:
        print("[WARN] No checkpoint — skipping play/video")
        return None
    play_script = REPO_ROOT / "roboparty_train" / "robolab" / "scripts" / "rsl_rl" / "play_amp.py"
    cmd = [sys.executable, str(play_script),
           "--task", "F1W-AMP-Play",
           "--num_envs", "1",
           "--checkpoint", str(ckpt),
           "--video", "--video_length", "600",   # 600 steps = 12 s @ 50 Hz
           "--headless"]
    print(f"[INFO] {' '.join(cmd)}")
    try:
        with open(PLAY_LOG, "wb") as logf:
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf,
                                stderr=subprocess.STDOUT, timeout=1500).returncode
    except subprocess.TimeoutExpired:
        rc = -1
        print("[WARN] play timed out after 1500s — continuing")
    print(f"[INFO] play exit={rc} (log: {PLAY_LOG.name})")
    videos = []
    for root in [REPO_ROOT / "logs", Path.cwd() / "logs", REPO_ROOT,
                 ckpt.parent / "videos", OUTSIDE_DIR]:
        try:
            if root.exists():
                videos += [v for v in root.rglob("*.mp4") if "acceptance" not in v.parts]
        except OSError:
            pass
    print(f"[INFO] mp4 search: {len(videos)} found")
    video = max(videos, key=lambda v: v.stat().st_mtime) if videos else None
    if video:
        try:
            shutil.copy2(video, OUTSIDE_DIR / TAG / video.name)
        except OSError:
            pass
        print(f"[INFO] latest video: {video}")
    else:
        print("[WARN] No mp4 from Isaac play")
    return video


def _alias_env():
    for k in ("ITERS", "RESUME_CKPT", "FINE_TUNE_ITERS", "PLAY_VEL"):
        if os.environ.get(f"F1W_{k}"):
            os.environ[f"F1_{k}"] = os.environ[f"F1W_{k}"]


def main():
    t0 = time.time()
    # optional per-run env overrides committed as roboparty_train/f1_env.txt
    # (KEY=VALUE lines) — startScript cannot pass env vars to gm-run.
    # MUST load BEFORE _alias_env(): F1W_FINE_TUNE_ITERS etc. from the file
    # need to reach F1_FINE_TUNE_ITERS via the alias map (v1g bug: file was
    # loaded after aliasing, so the +4500 resume budget silently fell back
    # to the +1500 default and the task stopped at 3000/6000).
    env_file = SCRIPT_DIR / "f1w_env.txt"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        print(f"[ENV] f1w_env.txt overrides: "
              f"{[l.split('=')[0] for l in env_file.read_text().splitlines() if '=' in l and not l.strip().startswith('#')]}")
    _alias_env()
    if not phase_dataset_gate():
        sys.exit(2)
    rc = phase_train()
    ck = final_checkpoint()
    print(f"[INFO] final checkpoint: {ck}")
    video = phase_play_video(ck)
    # final mirror pass (belt and braces)
    if ck is not None:
        mirror_checkpoint(ck, TAG)
    # v1d/v1e lesson: the SDK scan cycle is ~16-17 min and the PT uploader
    # only queues files it sees fresh; wait spans a full cycle AFTER the
    # final mirror so model_5498 actually reaches the platform before the
    # pod dies.
    if ck is not None:
        wait_for_sdk(1400, "final checkpoint registration")
    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===")
    print(f"checkpoint={ck}")
    print(f"video={video}")
    print(f"train_exit={rc}")
    sys.exit(rc if rc != 0 else 0)


if __name__ == "__main__":
    main()
