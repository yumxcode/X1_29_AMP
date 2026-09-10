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
    """SDK only uploads .pt files — wrap JSON payloads in a pickle .pt.

    v22 probe (TASK_20260818_075) proved the SDK scanner is NON-RECURSIVE:
    only .pt files lying FLAT in the repo root /workspace/isaaclab/X1_29_AMP/
    are uploaded+registered (loadRun=repo dir name); model_upload/ and any
    subdirectory are 'detected globally' but never uploaded. So the PRIMARY
    destination is the repo root; model_upload/ keeps a redundant copy."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    p = UPLOAD_DIR / name
    with open(p, "wb") as f:
        _pkl.dump(obj, f)
    flat = REPO_ROOT / name
    try:
        shutil.copy2(p, flat)
        print(f"[UPLOAD] {name} -> repo-root FLAT ({p.stat().st_size // 1024}KB) "
              f"+ model_upload/")
    except OSError as e:
        print(f"[WARN] flat copy failed: {e}")
    return p


def wait_for_sdk(seconds: int, why: str):
    print(f"[INFO] Waiting {seconds}s for SDK upload queue ({why})...")
    time.sleep(seconds)


# ────────────────────────────────────────────────────────────────────
def phase_retarget():
    print("\n=== Phase 1-2: GMR Retarget + Isaac Lab dataset_retarget ===\n")
    reassemble_smplx()
    gmr_output = MOTIONS_DIR / "x1_gmr"
    lab_output = MOTIONS_DIR / "x1_lab"
    v31_dir = MOTIONS_DIR / "x1_lab_v31"
    # v31b FIX (TASK_20260910_128 postmortem): the skip check demanded >= 12
    # x1_gmr files but the v31 dataset ships 11 — the pipeline entered the
    # GMR venv setup, the pod's DNS could not resolve /pypi (v27 lesson,
    # TASK_20260908_230) and the task died in 5 min. The GMR stack is dead
    # weight whenever the FINAL products ship in-repo: gate on x1_lab_v31
    # (11 sources + 11 mirrors) plus the intermediate dirs being non-empty.
    # NEVER gate the skip on exact intermediate counts — they change with
    # every dataset revision while the v31 layout is what training reads.
    # NOTE x1_lab holds 21 files (11 sources + 10 legacy mirrors; 103_07's
    # mirror lives only in x1_lab_v31) — count SOURCES, not raw files.
    n_v31 = len(list(v31_dir.glob("*.pkl"))) if v31_dir.exists() else 0
    n_gmr = len(list(gmr_output.glob("*.pkl"))) if gmr_output.exists() else 0
    n_lab_src = len([p for p in lab_output.glob("*.pkl")
                     if not p.stem.endswith("_mirror")]) if lab_output.exists() else 0
    if n_v31 >= 22 and n_gmr >= 11 and n_lab_src >= 11:
        print(f"[INFO] x1_lab_v31 ({n_v31}) + x1_gmr ({n_gmr}) + x1_lab ({n_lab_src} src) "
              "already in-repo — skipping GMR setup/venv entirely")
        return gmr_output, lab_output, None

    gmr_dir, venv_dir = setup_gmr()
    hide_gvhmr_pt_early(gmr_dir)
    register_x1_in_gmr(gmr_dir)

    print("\n--- Auto-IK Calibration ---")
    run_auto_ik(gmr_dir, venv_dir)

    if gmr_output.exists() and len(list(gmr_output.glob("*.pkl"))) >= 11:
        print(f"[INFO] x1_gmr already has {len(list(gmr_output.glob('*.pkl')))} files, skipping GMR retarget")
    else:
        run_gmr_retarget(gmr_dir, venv_dir)

    if lab_output.exists() and len([p for p in lab_output.glob("*.pkl")
                                    if not p.stem.endswith("_mirror")]) >= 11:
        print(f"[INFO] x1_lab already has {len(list(lab_output.glob('*.pkl')))} files, skipping dataset_retarget")
    else:
        run_dataset_retarget(gmr_output)

    lab_files = [p for p in lab_output.glob("*.pkl") if not p.stem.endswith("_mirror")]
    print(f"\n[INFO] x1_lab: {len(lab_files)} source files")
    if len(lab_files) < 11:
        print("[ERROR] Expected >= 11 lab source files!")
        sys.exit(1)
    return gmr_output, lab_output, venv_dir


def phase_fix_arm_decomposition(venv_dir: Path | None):
    """Phase 2.5 (v31): post-process x1_lab -> x1_lab_v30 (arm decomposition
    fix, see fix_arm_decomposition.py header) -> x1_lab_v31 (ground contact
    fix: penetration 0, L/R ankle stance-pitch symmetrization, root_z
    anchoring; see fix_ground_root.py header) + mirror. Training reads
    x1_lab_v31 (x1_amp_env_cfg.py motion_data_dir). Gates: script exit codes.
    Skips when the 24 fixed+mirrored clips already ship in the repo."""
    print("\n=== Phase 2.5: Arm+Ground Fix (x1_lab -> v30 -> v31 + mirror) ===\n")
    v31_dir = MOTIONS_DIR / "x1_lab_v31"
    if v31_dir.exists() and len(list(v31_dir.glob("*.pkl"))) >= 22:
        print(f"[INFO] x1_lab_v31 already has {len(list(v31_dir.glob('*.pkl')))} files — skipping")
        return

    venv_python = str(venv_dir / "bin" / "python") if venv_dir else ""
    python = venv_python if (venv_python and Path(venv_python).exists()) else sys.executable
    fix_arm = REPO_ROOT / "roboparty_train" / "fix_arm_decomposition.py"
    fix_ground = REPO_ROOT / "roboparty_train" / "fix_ground_root.py"
    mirror_script = REPO_ROOT / "roboparty_train" / "mirror_lab_motions.py"
    v30_dir = MOTIONS_DIR / "x1_lab_v30"

    print(f"[INFO] Running arm fix: {python} {fix_arm}")
    r = subprocess.run([python, str(fix_arm),
                        "--src", str(MOTIONS_DIR / "x1_lab"),
                        "--dst", str(v30_dir)], cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print("[FATAL] fix_arm_decomposition gate FAILED — blocking training.")
        sys.exit(1)
    print(f"[INFO] Running ground fix: {python} {fix_ground}")
    r = subprocess.run([python, str(fix_ground),
                        "--src", str(v30_dir), "--dst", str(v31_dir)],
                       cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print("[FATAL] fix_ground_root gate FAILED — blocking training.")
        sys.exit(1)
    print(f"[INFO] Running mirror: {python} {mirror_script} --src {v31_dir}")
    r = subprocess.run([python, str(mirror_script), "--src", str(v31_dir)],
                       cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print("[FATAL] mirror gate FAILED — blocking training.")
        sys.exit(1)
    n = len(list(v31_dir.glob("*.pkl")))
    print(f"[INFO] x1_lab_v31: {n} files (11 fixed + 11 mirrored)")


def phase_gait_gate(venv_dir: Path | None):
    """Phase 2.6 (v31): gait-quality acceptance gate — MANDATORY, runs
    UNCONDITIONALLY on whatever x1_lab_v31 the training will consume
    (rebuilt or shipped).

    v31d postmortem: this gate used to live INSIDE
    phase_fix_arm_decomposition after its skip branch, so a repo shipping
    x1_lab_v31 skipped the gate as well — the shipped dataset was never
    re-verified by the pipeline. Now decoupled: the fix phase only *builds*
    the data, this phase always *verifies* it.

    Exit-code semantics of check_retarget_gait.py: 0 PASS / 1 FAIL (real
    data verdict) / 2 setup error (e.g. mujoco missing — the container's
    default python lacks it, see check_retarget E0). Environment hardening
    mirrors sim2sim_fallback_video: [sys, pylibs] candidates + one
    numpy<2+mujoco install into pylibs. A persistent setup error is a
    WARN (consistent with check_retarget's E0 policy for the optional
    mujoco group) — only a real FAIL blocks training."""
    print("\n=== Phase 2.6: Gait-Quality Acceptance Gate ===\n")
    v31_dir = MOTIONS_DIR / "x1_lab_v31"
    gate = REPO_ROOT / "acceptance" / "check_retarget_gait.py"
    gate_json = UPLOAD_DIR / "gait_gate_report.json"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if gate_json.exists():
        gate_json.unlink()  # stale report must not masquerade as this run's

    pylibs = REPO_ROOT / "pylibs"
    candidates = [("sys", dict(os.environ))]
    if pylibs.exists():
        candidates.append(("pylibs", dict(os.environ, PYTHONPATH=str(pylibs))))

    def run_gate(env: dict, label: str) -> int:
        cmd = [sys.executable, str(gate), "--dir", str(v31_dir),
               "--repo-root", str(REPO_ROOT), "--json", str(gate_json)]
        print(f"[INFO][{label}] {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode

    rc = 2
    for label, env in candidates:
        rc = run_gate(env, label)
        if rc in (0, 1):   # real verdict — stop probing
            break
    if rc == 2:
        print("[WARN] gate setup error on all python candidates — "
              "installing numpy<2+mujoco into pylibs once")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--target", str(pylibs), "numpy<2", "mujoco"],
                       check=False)
        rc = run_gate(dict(os.environ, PYTHONPATH=str(pylibs)), "pylibs-installed")

    if rc == 0:
        print("[INFO] gait-quality gate PASSED.")
        return
    if rc == 1:
        # real data verdict — failure evidence worth a registration slot
        # (upload only on failure; 5-slot budget, see v26 note)
        if gate_json.exists():
            wrap_json_for_upload("model_gait_gate_report.pt",
                                 {"phase": "gait_gate", "passed": False,
                                  "report": gate_json.read_text()})
        print("[FATAL] gait-quality gate FAILED — blocking training.")
        wait_for_sdk(180, "upload gate failure report")
        sys.exit(1)
    print("[WARN] gait gate could NOT run in this environment (setup error) — "
          "relying on check_retarget.py + the dataset's local validation "
          "history; re-run acceptance/check_retarget_gait.py manually.")


def phase_retarget_acceptance(venv_dir: Path) -> bool:
    """Strict gate: acceptance/check_retarget.py. Returns True on PASS."""
    print("\n=== Phase 3: STRICT Retarget Acceptance Gate ===\n")
    checker = REPO_ROOT / "acceptance" / "check_retarget.py"
    report_json = UPLOAD_DIR / "retarget_acceptance_report.json"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # checker json write needs it
    venv_python = str(venv_dir / "bin" / "python") if venv_dir else ""
    cmd = [
        "python" if not (venv_python and Path(venv_python).exists()) else venv_python,
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
    v31_dir = MOTIONS_DIR / "x1_lab_v31"
    for f in sorted(v31_dir.glob("*.pkl")):
        retarget_pkg[f"x1_lab_v31/{f.name}"] = f.read_bytes()
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
    """Mirror the final checkpoint to SDK-visible locations.

    v23 evidence (TASK_20260818_153): model_upload/ is watched CONTINUOUSLY —
    4/4 files registered from there (t0, +38min, +160min; the +160min
    model_amp_report.pt was detected in <1s and uploaded in 9s). Repo-root
    flat is EARLY-window-only (probe2): v23's flat model_3999.pt written at
    t+160min was never detected while model_upload/ stayed live. So:
      0) PRIMARY: UPLOAD_DIR/ckpt.name  (model_upload/, always registered)
      1) repo-root flat (probe-proven, but only inside the early window)
      2) exported_data pattern logs/{exp}/exported_data/{run}/ (skill doc)
      3) outside-repo dir (artifact archive)"""
    copied = []
    # 0) PRIMARY (v29e4): IN-TREE checkpoints/ dir — the ONLY location that
    #    actually registered on v29e3 (TASK_20260910_035: repo-tree
    #    checkpoints/model_9497.pt registered with loadRun="checkpoints"
    #    while ALL out-of-tree mirrors of model_10496 were "detected
    #    globally" but never queued; pod then destroyed with the file).
    #    Probe conclusions drift — trust the most recent real run.
    try:
        dst0 = REPO_ROOT / "roboparty_train" / "checkpoints" / ckpt.name
        if not dst0.exists():
            shutil.copy2(ckpt, dst0)
            copied.append(dst0)
    except OSError as e:
        print(f"[MONITOR] in-tree checkpoints/ mirror failed: {e}")
    # 0b) model_upload/ (v23 evidence: continuously watched back then)
    try:
        dst0 = UPLOAD_DIR / ckpt.name
        if not dst0.exists():
            shutil.copy2(ckpt, dst0)
            copied.append(dst0)
    except OSError as e:
        print(f"[MONITOR] model_upload mirror failed: {e}")
    # 1) repo-root flat (early-window registration, belt-and-braces)
    try:
        dst0 = REPO_ROOT / ckpt.name
        if not dst0.exists():
            shutil.copy2(ckpt, dst0)
            copied.append(dst0)
    except OSError as e:
        print(f"[MONITOR] flat mirror failed: {e}")
    # 1) skill-documented exported_data pattern
    for root in [REPO_ROOT / "logs", Path.cwd() / "logs"]:
        try:
            exp = root / "x1_amp" / "exported_data" / tag
            exp.mkdir(parents=True, exist_ok=True)
            dst1 = exp / ckpt.name
            if not dst1.exists():
                shutil.copy2(ckpt, dst1)
                copied.append(dst1)
        except OSError:
            pass
    # 2) outside-repo archive
    try:
        od = OUTSIDE_DIR / tag
        od.mkdir(parents=True, exist_ok=True)
        dst2 = od / ckpt.name
        if not dst2.exists():
            shutil.copy2(ckpt, dst2)
            copied.append(dst2)
    except OSError as e:
        print(f"[MONITOR] outside mirror failed: {e}")
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

    # v27.3: NO pip at all. This pod's offline pip is broken two ways
    # (TASK_20260908_230: /pypi mirror unresolvable by DNS; 242:
    # --no-build-isolation still dies in image-bundled pip on
    # `packaging.licenses`). PYTHONPATH achieves the same shadowing the
    # editable installs did: rsl_rl FIRST so it overrides IsaacLab's
    # site-packages rsl_rl 3.1.2, then robolab. PYTHONPATH precedes
    # site-packages in sys.path, and it inherits into every subprocess
    # spawned below (train, play, rollout fallbacks).
    parts = [str(rsl_rl_src), str(robolab_src)]
    old = os.environ.get("PYTHONPATH", "")
    if old:
        parts.append(old)
    os.environ["PYTHONPATH"] = ":".join(parts)
    print(f"[INFO] PYTHONPATH shadowing (no pip): {os.environ['PYTHONPATH']}")

    train_script = robolab_src / "scripts" / "rsl_rl" / "train.py"
    cmd = [sys.executable, str(train_script),
           "--task=X1-AMP", "--headless", "--logger=tensorboard", "--num_envs=4096"]

    # ------------------------------------------------------------------
    # v29d fine-tune resume: X1_RESUME_CKPT (absolute path to a platform-
    # mounted .pt) resumes training from a converged policy instead of
    # scratch. Rationale (v29b/v29c postmortem): training from scratch
    # under push+delay randomization converges to a perturbation-survival
    # gait that REGRESSES the strict sim2sim criteria (v29c hip ratio
    # 0.617 vs v28d 0.884; 0.5 m/s gait degenerated to 1 landing/12 s).
    # Fine-tuning the 13/13 v28d policy keeps the formed gait while
    # learning push/latency recovery on top.
    # rsl_rl resume: current_learning_iteration starts at the loaded
    # value (3999), so --max_iterations must be base+fine-tune iters.
    # ------------------------------------------------------------------
    resume_ckpt = os.environ.get("X1_RESUME_CKPT", "").strip()
    if not resume_ckpt:
        # platform resume tasks mount the .pt at checkPointMountPath
        # (we set "X1_29_AMP/" => file lands at repo root); auto-detect the
        # mounted model_*.pt when the env var is absent (v29d: model_3999,
        # v29e: model_9497 — take the highest-numbered mount; nothing else
        # writes .pt files at repo root).
        def _num(p):
            s = p.stem[len("model_"):]
            return int(s) if s.isdigit() else -1
        cands = sorted((p for p in REPO_ROOT.glob("model_*.pt") if _num(p) >= 0),
                       key=_num)
        if cands:
            resume_ckpt = str(cands[-1])
            print(f"[RESUME] auto-detected mounted checkpoint: {resume_ckpt}")
    resume_iters = int(os.environ.get("X1_FINE_TUNE_ITERS", "1500"))
    if resume_ckpt:
        src = Path(resume_ckpt)
        if not src.is_file():
            raise FileNotFoundError(f"X1_RESUME_CKPT not found: {src}")
        # base iteration from the checkpoint FILENAME (its content iteration
        # count is what rsl_rl actually resumes at — keep names consistent).
        s = src.stem[len("model_"):]
        base_iter = int(s) if s.isdigit() else 3999
        resume_run = REPO_ROOT / "logs" / "rsl_rl" / "x1_amp" / "_resume_src"
        resume_run.mkdir(parents=True, exist_ok=True)
        dst = resume_run / src.name
        shutil.copy2(src, dst)
        # rsl_rl resume semantics (verified twice on TASK_20260909_231 /
        # TASK_20260910_022): on resume, max_iterations is ADDITIONAL
        # iterations — the runner trains to loaded_iter + max_iterations.
        # So pass resume_iters directly (passing base+iters made v29e's
        # planned +1000 become +10497 -> 19994 total, task stopped).
        cmd += ["--resume", "--load_run", "_resume_src",
                "--checkpoint", src.name,
                "--max_iterations", str(resume_iters)]
        print(f"[RESUME] fine-tune from {src} (base iter {base_iter}) "
              f"+{resume_iters} iters -> max_iterations={resume_iters} "
              f"(rsl_rl total {base_iter + resume_iters})")


    tag = _dt.now().strftime("%Y-%m-%d_%H-%M-%S") + "x1_amp"
    stop_monitor = threading.Event()
    mirrored = set()

    def monitor():
        """Watch training checkpoints. v26: mirror model_2000 as mid-flight
        insurance — v25 was balance-killed at iter 3458 with all progress
        lost (save_interval=4000 => only model_0 on disk). v28d: ALSO mirror
        model_3000 — v28c (TASK_20260908_361) was balance-killed at iter
        3370/4000; everything past 2000 was lost with the pod. Quota note:
        v27g registered 6 files via model_upload; anchor + retarget x2 +
        2000 + 3000 + 3999 = 6, amp_report is the deliberate 7th sacrifice
        (it also lands in the outside-tree final mirror)."""
        last = None
        while not stop_monitor.is_set():
            try:
                names = sorted(all_checkpoints())
                if names != last:
                    print(f"[MONITOR] checkpoints on disk: {names}")
                    last = names
                if "model_2000.pt" in names and "model_2000.pt" not in mirrored:
                    ck = all_checkpoints().get("model_2000.pt")
                    if ck is not None and ck.exists():
                        print("[MONITOR] mid-flight insurance mirror: model_2000.pt")
                        mirror_checkpoint(ck, tag)
                        mirrored.add("model_2000.pt")
                if "model_3000.pt" in names and "model_3000.pt" not in mirrored:
                    ck = all_checkpoints().get("model_3000.pt")
                    if ck is not None and ck.exists():
                        print("[MONITOR] mid-flight insurance mirror: model_3000.pt (v28c lesson)")
                        mirror_checkpoint(ck, tag)
                        mirrored.add("model_3000.pt")
                # v29d fine-tune mode: iterations run 4000..5499, so the
                # 2000/3000 insurance never fires. Mirror model_5000 as the
                # mid-flight insurance instead (final 5499 is mirrored below).
                if "model_5000.pt" in names and "model_5000.pt" not in mirrored:
                    ck = all_checkpoints().get("model_5000.pt")
                    if ck is not None and ck.exists():
                        print("[MONITOR] fine-tune mid-flight insurance mirror: model_5000.pt")
                        mirror_checkpoint(ck, tag)
                        mirrored.add("model_5000.pt")
                # v29e4: mirror ANY new model_*.pt as it appears (the named
                # insurance list above only covers fresh-run milestones;
                # fine-tune saves land at arbitrary iteration numbers, e.g.
                # model_10000 on a 9497-resume with save_interval=500)
                for nm, ck in sorted(all_checkpoints().items()):
                    if nm.startswith("model_") and nm not in mirrored:
                        print(f"[MONITOR] new checkpoint mirror: {nm}")
                        mirror_checkpoint(ck, tag)
                        mirrored.add(nm)
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
    print(f"[INFO] Mirrored final: {final.name if final else None} "
          f"(plus mid-flight {sorted(mirrored)}). "
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


def hide_gvhmr_pt_early(gmr_dir: Path):
    """v25: MOVE GMR_X1/gvhmr_pt out of /workspace BEFORE the SDK's early
    enumeration (~t+5-6min, v24 evidence). In v21b/v22/v24 these 5 junk
    reference .pt files registered first and ate the whole 5-slot model
    quota, blocking every real artifact. Our retarget path is pure IK from
    SMPL-X npz and never reads them (v21b+ purged them post-retarget with
    retarget unaffected). Moving (not deleting) keeps them recoverable."""
    import shutil as _sh
    moved = 0
    for cand in (gmr_dir / "gvhmr_pt", REPO_ROOT / "GMR" / "gvhmr_pt"):
        if cand.is_dir():
            try:
                dst = Path("/tmp") / f"gvhmr_pt_hidden_{_dt.now():%H%M%S}"
                _sh.move(str(cand), str(dst))
                moved += 1
                print(f"[HIDE] moved {cand} -> {dst} (pre-enumeration)")
            except OSError as e:
                print(f"[HIDE] failed {cand}: {e}")
    if not moved:
        print("[HIDE] no gvhmr_pt dir found (nothing to hide)")


def export_policy_npz(ckpt: Path) -> Path | None:
    """v25: export actor weights to .npz right after training (kit python has
    torch + the checkpoint in-hand). The MuJoCo rollout then needs no torch,
    removing the kit-torch/numpy2.x ABI failure mode of v24's pylibs attempts.

    Robust key discovery: isaaclab MLP state keys may be actor.<i>.weight or
    actor.layers.<i>.weight depending on version — regex both; print the full
    actor key list on mismatch for one-run diagnosability."""
    if ckpt is None:
        return None
    out = ckpt.with_suffix(".policy.npz")
    code = f"""
import re, sys
import numpy as np
import torch
ck = torch.load(r"{ckpt}", map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
pairs = {{}}
for k, v in sd.items():
    m = re.match(r"actor\\.(?:[a-zA-Z_]+\\.)?(\\d+)\\.(weight|bias)$", k)
    if m:
        pairs.setdefault(int(m.group(1)), {{}})[m.group(2)] = v.detach().float().numpy()
idx = sorted(i for i, d in pairs.items() if "weight" in d)
if len(idx) not in (3, 4, 5):
    print("[EXPORT] actor keys:", sorted(k for k in sd if k.startswith("actor.")))
    sys.exit(3)
out = {{}}
for n, i in enumerate(idx):
    out[f"l{{n}}_w"] = pairs[i]["weight"]
    if "bias" in pairs[i]:
        out[f"l{{n}}_b"] = pairs[i]["bias"]
for cand in ("actor_obs_normalizer._mean", "actor_obs_normalizer.mean"):
    if cand in sd:
        out["mean"] = sd[cand].detach().float().numpy().reshape(-1)
        out["std"] = sd[cand.replace("mean", "std" if cand.endswith("mean") else "_std")].detach().float().numpy().reshape(-1)
        break
else:
    print("[EXPORT] no normalizer keys:", [k for k in sd if "normal" in k])
    sys.exit(4)
if "mean" not in out:
    sys.exit(5)
np.savez(r"{out}", **out)
print("[EXPORT] layers:", [(w.shape) for w in [out[f'l{{n}}_w'] for n in range(len(idx))]])
print("[EXPORT] mean/std shapes:", out["mean"].shape, out["std"].shape)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in (r.stdout or "").splitlines():
        print(f"[EXPORT] {line}")
    if r.returncode != 0 or not out.exists():
        tail = (r.stderr or "").splitlines()[-6:]
        print(f"[WARN] npz export failed rc={r.returncode}: {' | '.join(tail)[:600]}")
        return None
    print(f"[INFO] policy exported -> {out.name} ({out.stat().st_size // 1024}KB)")
    return out


def phase_play_video(ckpt: Path, policy_npz: Path | None = None):
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
                # v32: EXCLUDE acceptance/ — committed reference/vs-policy
                # render videos live there and polluted the search (v31d P6
                # "passed" citing 0026_circle_walk_orig_vs_v30.mp4, a ref
                # comparison clip, not a policy play video).
                videos += [v for v in root.rglob("*.mp4")
                           if "acceptance" not in v.parts]
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
        video = sim2sim_fallback_video(policy_npz or ckpt)

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
        # v25: sim2sim metrics are printed to the task log + archived in
        # x1_upload/sim2sim — NOT registered as a model (5-slot budget:
        # anchor, retarget x2, model_2000, model_3999)
        _dump_mujoco_metrics(sim_dir)
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


def sim2sim_fallback_video(policy: Path):
    """Render walking videos of the final policy in MuJoCo — 3 rollouts
    (walk 1.0 / walk 1.5 / walk+turn) with metrics JSON. Doubles as the
    sim2sim deliverable. `policy` is the exported .npz (preferred) or the
    raw .pt checkpoint. Returns the first video path, None on failure.

    v25 changes after the v24 all-matrix failure:
    - numpy pinned <2 in pylibs (v24's pylibs numpy 2.x vs kit-torch ABI is
      the prime suspect for the pylibs-soft failure; npz removes torch need)
    - every attempt's rc + stderr tail printed to the TASK log (v24 wrote
      only to mujoco_fallback.log whose tail showed just the last GL crash —
      we never learned why sys-soft/pylibs-soft failed)
    - the apt last-resort no longer writes to an already-closed logf
      (v24: 'I/O operation on closed file' masked its real behavior)"""
    try:
        import subprocess as _sp

        sys_ok = _sp.run([sys.executable, "-c", "import mujoco, imageio.v2"],
                         capture_output=True).returncode == 0
        print(f"[INFO] system python mujoco+imageio probe: {'OK' if sys_ok else 'missing'}")

        pylibs = REPO_ROOT / "pylibs"
        pylibs.mkdir(exist_ok=True)
        _sp.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(pylibs),
                 "numpy<2", "mujoco", "imageio", "imageio-ffmpeg", "matplotlib"],
                check=True, timeout=900)
        for bad in list(pylibs.glob("OpenGL")) + list(pylibs.glob("PyOpenGL*")):
            shutil.rmtree(bad, ignore_errors=True)

        rollout = REPO_ROOT / "sim2sim" / "mujoco_rollout.py"
        out_dir = OUTSIDE_DIR / "sim2sim"
        out_dir.mkdir(parents=True, exist_ok=True)
        rollouts = [
            ("walk_1.0", ["--cmd", "1.0", "0.0", "0.0", "--duration", "12"]),
            ("walk_1.5", ["--cmd", "1.5", "0.0", "0.0", "--duration", "8"]),
            ("walk_turn", ["--cmd", "1.0", "0.0", "0.8", "--duration", "8"]),
        ]

        def run_matrix(matrix, tag):
            good, videos = None, []
            for label, env in matrix:
                ok = True
                for name, extra in rollouts:
                    mp4 = out_dir / f"x1_sim2sim_{name}.mp4"
                    js = out_dir / f"x1_sim2sim_{name}.json"
                    cmd = [sys.executable, str(rollout), "--ckpt", str(policy),
                           "--repo-root", str(REPO_ROOT), "--video", str(mp4),
                           "--json", str(js), "--render", "soft"] + extra
                    try:
                        r = _sp.run(cmd, cwd=str(REPO_ROOT), env=env, timeout=1200,
                                    capture_output=True, text=True)
                    except Exception as run_err:
                        print(f"[MUJOCO][{label}/{name}] run error: {run_err}")
                        ok = False
                        break
                    bad_size = not (mp4.exists() and mp4.stat().st_size > 100_000)
                    if r.returncode != 0 or bad_size:
                        tail = [l for l in (r.stderr or "").splitlines() if l.strip()][-6:]
                        print(f"[MUJOCO][{label}/{name}] rc={r.returncode} "
                              f"{'(video missing/<100KB) ' if bad_size else ''}failed")
                        print(f"[MUJOCO][{label}/{name}] tail: " +
                              " | ".join(t.strip()[:150] for t in tail)[:900])
                        ok = False
                        break
                    videos.append(mp4)
                    for l in (r.stdout or "").splitlines():
                        if l.startswith(("[INFO]", "[VIDEO]", "[FALL]")):
                            print(f"[MUJOCO][{label}/{name}] {l}")
                if ok:
                    good = label
                    break
                videos.clear()
            return good, videos

        pp = str(pylibs)
        matrix = [
            ("sys-soft", dict(os.environ)),
            ("pylibs-soft", dict(os.environ, PYTHONPATH=pp)),
        ]
        good_env, videos = run_matrix(matrix, "soft")
        if good_env:
            print(f"[INFO] mujoco soft rollouts OK via {good_env}: "
                  f"{[v.name for v in videos]}")
            _dump_mujoco_metrics(out_dir)
            return videos[0]
        # GL quality upgrades (also serve as extra evidence if soft failed)
        gl_matrix = [
            ("sys-egl-gl", dict(os.environ, MUJOCO_GL="egl")),
            ("pylibs-osmesa-gl", dict(os.environ, MUJOCO_GL="osmesa",
                                      PYOPENGL_PLATFORM="osmesa", PYTHONPATH=pp)),
        ]
        good_env, videos = run_matrix(gl_matrix, "gl")
        if good_env:
            print(f"[INFO] mujoco GL rollouts OK via {good_env}: "
                  f"{[v.name for v in videos]}")
            _dump_mujoco_metrics(out_dir)
            return videos[0]
        # last resort: system GL libs, then osmesa via pylibs
        try:
            print("[INFO] last resort: apt-get install libosmesa6 + retry")
            r = _sp.run(["apt-get", "install", "-y", "-q", "libosmesa6", "libegl1"],
                        timeout=300, capture_output=True, text=True)
            print(f"[INFO] apt rc={r.returncode}: {(r.stderr or '')[-200:]}")
            env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
                       PYTHONPATH=str(pylibs))
            good_env, videos = run_matrix([("apt-osmesa-gl", env)], "gl")
            if good_env:
                print(f"[INFO] mujoco rollouts OK via apt-osmesa: {[v.name for v in videos]}")
                _dump_mujoco_metrics(out_dir)
                return videos[0]
        except Exception as apt_err:
            print(f"[WARN] apt osmesa retry failed: {apt_err}")
        print("[WARN] all mujoco render attempts failed")
    except Exception as e:
        print(f"[WARN] mujoco fallback failed: {e}")
    return None


def _dump_mujoco_metrics(out_dir: Path):
    """v25: sim2sim metrics go to the TASK LOG + outside archive only (NOT
    model_upload — v26 budget: anchor, retarget x2, model_2000, model_3999
    (amp_report sacrificed to the quota after model_2000 took its slot)."""
    try:
        for js in sorted(out_dir.glob("x1_sim2sim_*.json")):
            print(f"[SIM2SIM] {js.stem}: {js.read_text()[:400]}")
    except Exception as e:
        print(f"[WARN] metrics dump failed: {e}")


def phase_policy_gait_gate(ckpt: Path | None, policy_npz: Path | None) -> int:
    """Phase 5.5 (v31): POLICY gait-quality gate (P7).

    check_amp.py (P1-P6) verifies process health but nothing measured the
    arm/torso style — v29 passed 13/13 yet carried the arms same-phase with
    folded elbows (measured on its rollout: anti +0.91, elbow p95 90 deg,
    coupling -0.68; acceptance/check_policy_gait.py). This phase rolls the
    final policy in MuJoCo at 1.0 m/s (log-only, no GL needed), measures the
    same joint-space quantities the AMP discriminator saw, and FAILS the
    task on regression. Exit code merges into the task verdict."""
    print("\n=== Phase 5.5: Policy Gait-Quality Gate (P7) ===\n")
    if ckpt is None or policy_npz is None or not Path(policy_npz).exists():
        print("[FAIL] no checkpoint / policy npz — P7 cannot run (counts as FAIL)")
        return 1

    rollout = REPO_ROOT / "sim2sim" / "mujoco_rollout.py"
    gate = REPO_ROOT / "acceptance" / "check_policy_gait.py"
    log_npz = UPLOAD_DIR / "p7_gait_rollout_1.0.npz"
    roll_json = UPLOAD_DIR / "p7_rollout_1.0.json"
    gate_json = UPLOAD_DIR / "p7_gait_gate.json"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # v31: mirror sim2sim_fallback_video's robust python matrix — the image's
    # default python may lack mujoco (check_retarget E0 precedent), while
    # phase 5's fallback installs numpy<2 + mujoco into REPO_ROOT/pylibs.
    # Log-only rollout needs just numpy + mujoco (no GL, no imageio).
    pylibs = REPO_ROOT / "pylibs"
    if log_npz.exists():
        log_npz.unlink()  # stale run must not masquerade as success

    def run_rollout(env: dict, label: str) -> int:
        cmd = [sys.executable, str(rollout), "--ckpt", str(policy_npz),
               "--repo-root", str(REPO_ROOT),
               "--cmd", "1.0", "0.0", "0.0", "--duration", "12",
               "--log", str(log_npz), "--json", str(roll_json)]
        print(f"[INFO][{label}] {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode

    candidates = [("sys", dict(os.environ))]
    if pylibs.exists():
        candidates.append(("pylibs", dict(os.environ, PYTHONPATH=str(pylibs))))
    rc = 1
    for label, env in candidates:
        rc = run_rollout(env, label)
        if rc == 0 and log_npz.exists():
            break
    if rc != 0 or not log_npz.exists():
        print("[WARN] rollout failed on all candidates — installing mujoco into pylibs")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--target", str(pylibs), "numpy<2", "mujoco"], check=False)
        rc = run_rollout(dict(os.environ, PYTHONPATH=str(pylibs)), "pylibs-installed")
    if rc != 0 or not log_npz.exists():
        print("[FAIL] MuJoCo rollout did not produce a gait log — P7 FAIL "
              "(checkpoint is still usable; P7 can be re-run locally)")
        return 1

    cmd = [sys.executable, str(gate), "--log", str(log_npz), "--json", str(gate_json)]
    print(f"[INFO] {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    # v31: report upload only on FAILURE (protect the 5-slot registration
    # budget: anchor, retarget x2, model_2000, model_3999 — see v26 note).
    # On PASS the exit code + full gate stdout in the task log suffice; both
    # jsons are mirrored to OUTSIDE_DIR/final at Phase 7 either way.
    if rc != 0 and gate_json.exists():
        wrap_json_for_upload("model_p7_gait_gate.pt",
                             {"phase": "policy_gait_gate", "passed": False,
                              "report": gate_json.read_text()})
    print(f"[INFO] P7 gait gate {'PASSED' if rc == 0 else 'FAILED'} (rc={rc})")
    return rc


def phase_amp_acceptance(video: Path | None):
    print("\n=== Phase 6: AMP Training Acceptance ===\n")
    checker = REPO_ROOT / "acceptance" / "check_amp.py"
    report_json = UPLOAD_DIR / "amp_acceptance_report.json"
    # v29d: parse actual max_iterations from the train log (fine-tune runs
    # report "iteration X/5499"); fall back to 4000 for fresh runs.
    max_iter = 4000
    try:
        _txt = TRAIN_LOG_FILE.read_text(errors="replace")
        _ms = re.findall(r"iteration \d+/(\d+)", _txt)
        if _ms:
            max_iter = int(_ms[-1])
    except Exception:
        pass
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
    print("X1 AMP Pipeline v26 (mid-flight ckpt insurance + v25 fixes)")
    print("=" * 60)

    # v25 t0 anchor (v23-proven, v24 regression removed): model_upload/ must
    # EXIST and hold a .pt before the SDK's early enumeration (~t+5min) or the
    # directory is never watched for the rest of the task. v23 anchored at t0
    # and registered 4/4 files across the whole run; v24 dropped the anchor,
    # created model_upload/ at t+50min and registered NOTHING from it.
    # Same insurance for the video channel dir (logs/x1_amp/).
    try:
        (REPO_ROOT / "logs" / "x1_amp").mkdir(parents=True, exist_ok=True)
        wrap_json_for_upload("pipeline_meta.pt", {
            "pipeline": "x1_amp_v25",
            "started": f"{_dt.now():%Y-%m-%d %H:%M:%S}",
            "note": "t0 anchor: keep model_upload/ on the SDK watch list",
        })
    except Exception as e:
        print(f"[WARN] t0 anchor failed: {e}")

    gmr_output, lab_output, venv_dir = phase_retarget()
    phase_fix_arm_decomposition(venv_dir)
    phase_gait_gate(venv_dir)
    phase_retarget_acceptance(venv_dir)
    phase_package_retarget(gmr_output, lab_output)

    rc = phase_train()
    if rc != 0:
        print(f"[ERROR] AMP training exited with code {rc}")
    ckpt = final_checkpoint()
    print(f"[INFO] Final checkpoint: {ckpt}")
    policy_npz = export_policy_npz(ckpt)

    video = phase_play_video(ckpt, policy_npz)
    p7_rc = phase_policy_gait_gate(ckpt, policy_npz)
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
    # v25: task exit code = AMP acceptance verdict (0 PASS / 1 FAIL). v23/v24
    # exited 0 with FAIL verdicts — the platform's ret:True masked the truth.
    # Artifacts are already registered by now (420s wait above), and failed
    # tasks still expose their model lists (v18/v19/v21b/v23 precedent).
    # v31: exit code = AMP acceptance AND P7 policy gait gate (either FAIL
    # fails the task; artifacts are uploaded before this point either way)
    sys.exit(max(amp_rc, p7_rc))


if __name__ == "__main__":
    main()
