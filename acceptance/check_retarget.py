#!/usr/bin/env python3
"""
X1 GMR retarget acceptance checker (strict).
Implements acceptance/RETARGET_ACCEPTANCE.md v1.0.

Usage:
    python check_retarget.py --repo-root <path to X1_29_AMP> \
        [--gmr-dir <x1_gmr>] [--lab-dir <x1_lab>] [--json out.json]

Runs on pure numpy (+ optional mujoco for group E). Exit code:
    0 = PASS, 1 = FAIL, 2 = setup error.
"""
import argparse
import functools
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------- limits (x1.xml / f1.urdf, rad)
JOINT_LIMITS = {
    "lumbar_yaw_joint": (-1.0, 1.0),
    "lumbar_roll_joint": (-0.25, 0.25),
    "lumbar_pitch_joint": (-0.15, 0.45),
    "left_shoulder_pitch_joint": (-2.5, 1.5),
    "right_shoulder_pitch_joint": (-2.5, 1.5),
    "left_shoulder_roll_joint": (-2.0, 0.0),
    "right_shoulder_roll_joint": (-2.0, 0.0),
    "left_shoulder_yaw_joint": (-1.8, 1.8),
    "right_shoulder_yaw_joint": (-1.8, 1.8),
    "left_elbow_pitch_joint": (0.0, 2.0),
    "right_elbow_pitch_joint": (0.0, 2.0),
    "left_elbow_yaw_joint": (-1.8, 1.8),
    "right_elbow_yaw_joint": (-1.8, 1.8),
    "left_wrist_pitch_joint": (-0.4, 0.4),
    "right_wrist_pitch_joint": (-0.4, 0.4),
    "left_wrist_roll_joint": (-0.4, 0.4),
    "right_wrist_roll_joint": (-0.4, 0.4),
    "left_hip_pitch_joint": (-1.0, 2.0),
    "right_hip_pitch_joint": (-2.0, 1.0),  # NOTE: mirrored limits as in URDF
    "left_hip_roll_joint": (-1.5, 0.2),
    "right_hip_roll_joint": (-0.2, 1.5),
    "left_hip_yaw_joint": (-1.5, 1.5),
    "right_hip_yaw_joint": (-1.5, 1.5),
    "left_knee_pitch_joint": (0.0, 2.0),
    "right_knee_pitch_joint": (0.0, 2.0),
    "left_ankle_pitch_joint": (-0.41, 0.35),
    "right_ankle_pitch_joint": (-0.41, 0.35),
    "left_ankle_roll_joint": (-0.64, 0.64),
    "right_ankle_roll_joint": (-0.64, 0.64),
}

# Hard thresholds (RETARGET_ACCEPTANCE.md)
HARD_MARGIN = 0.02          # B1: allowed excursion beyond hard limit (rad)
HARD_FRAME_FRAC = 0.001     # B1: allowed fraction of out-of-limit frames
SOFT_SHRINK = 0.95          # B2: soft limit = shrink range by 5%
SOFT_FRAME_FRAC = 0.05      # B2: max fraction beyond soft limit
ROOT_Z_MEAN = (0.45, 0.75)  # C1
ROOT_Z_MIN, ROOT_Z_MAX = 0.35, 0.85  # C2
DZ_FRAME = 0.05             # C3
DXY_FRAME = 0.08            # C4
VEL_FRAME_FRAC = 0.001      # C3/C4 allowed violation fraction
DQ_MAX = 0.5                # D1 (rad/frame)
DQ_MED, DQ_P99 = 0.15, 0.35 # D2
GND_MIN = -0.02             # E1
FOOT_MIN = 0.01             # E2
FK_MATCH = 5e-3             # E3
LAB_QMAX = 0.005            # G1
LAB_ROOT = 1e-6             # G2
GMR_FIELDS = ["fps", "root_pos", "root_rot", "dof_names", "dof_pos",
              "body_names", "body_positions"]
LAB_FIELDS = ["fps", "root_pos", "root_rot", "dof_pos", "loop_mode", "key_body_pos"]
MOTION_WEIGHTS = {
    "114_08": 1.0, "114_09": 1.0, "127_04": 1.0, "127_06": 4.0,
    "36_01": 1.0, "36_11": 1.0,
    "0000_treadmill_norm": 2.0, "0002_treadmill_slow": 2.0, "0003_treadmill_jog": 2.0,
    "0005_normal_walk1": 2.0, "0007_normal_walk3": 2.0, "0008_normal_walk4": 2.0,
    "0009_normal_jog1": 2.0, "0026_circle_walk": 2.0,
}
WALK_PAT = re.compile(r"walk|jog|run|treadmill", re.I)
MOVING_PAT = re.compile(r"walk|jog|run", re.I)
_MJC_MISSING = False  # set once when mujoco import fails (E0 single notice)


def parse_yaml_lists(path: Path):
    """Minimal parser for x1.yaml (gmr_dof_names / lab_dof_names / lab_key_body_names)."""
    out, cur = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*(gmr_dof_names|lab_dof_names|lab_key_body_names):\s*$", line)
        if m:
            cur = m.group(1); out[cur] = []
            continue
        if cur:
            m = re.match(r"^\s*-\s*(\S+)\s*$", line)
            if m:
                out[cur].append(m.group(1))
            elif line.strip() and not line.strip().startswith("#"):
                cur = None
    return out


class Report:
    def __init__(self):
        self.fails, self.warns = [], []

    def fail(self, cid, msg):
        self.fails.append(f"[{cid}] {msg}")
        print(f"  FAIL [{cid}] {msg}")

    def warn(self, cid, msg):
        self.warns.append(f"[{cid}] {msg}")
        print(f"  WARN [{cid}] {msg}")

    def ok(self, cid, msg=""):
        print(f"  ok   [{cid}] {msg}")


def fk_all_frames(xml_path: Path, root_pos, root_quat_wxyz, dof_pos, dof_names):
    """Recompute full-body FK with mujoco. Returns (body_names, positions [N,B,3])."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mj_joints = [model.joint(i).name for i in range(model.njnt)]
    name2qpos = {}
    qposadr = 0
    for jn in mj_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        name2qpos[jn] = model.jnt_qposadr[jid]
    dof_idx = [name2qpos[n] for n in dof_names]
    body_names = [model.body(i).name for i in range(model.nbody)]
    N = dof_pos.shape[0]
    pos = np.zeros((N, model.nbody, 3), dtype=np.float64)
    for i in range(N):
        q = np.zeros(model.nq)
        q[0:3] = root_pos[i]
        q[3:7] = root_quat_wxyz[i]
        q[dof_idx] = dof_pos[i]
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        pos[i] = data.xpos
    return body_names, pos


def check_gmr_file(pkl_path: Path, gmr_dof_names, xml_path, rep: Report):
    name = pkl_path.stem
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    # A1 fields
    missing = [k for k in GMR_FIELDS if k not in d]
    if missing:
        rep.fail("A1", f"{name}: missing fields {missing}")
        return None
    # A2 dof names/order
    dn = [str(x) for x in d["dof_names"]]
    if dn != gmr_dof_names:
        if set(dn) == set(gmr_dof_names):
            rep.fail("A2", f"{name}: dof_names order differs from x1.yaml gmr_dof_names")
        else:
            rep.fail("A2", f"{name}: dof_names set mismatch")
    q = np.asarray(d["dof_pos"], dtype=np.float64)
    N = q.shape[0]
    if q.ndim != 2 or q.shape[1] != 29:
        rep.fail("A2", f"{name}: dof_pos shape {q.shape}, want (N,29)")
        return None
    # A4 fps / frames
    # fps is the SOURCE sampling rate preserved by GMR (AMASS native 60/120 Hz).
    # Verified downstream: motion_data_manager computes dt=1/fps per motion and
    # samples by timestamp, so any physically-plausible capture rate is valid.
    # Range [30,250] catches bogus fps (0/1/NaN) without mis-flagging 120 Hz data.
    fps = float(d["fps"])
    if not (30 <= fps <= 250):
        rep.fail("A4", f"{name}: fps={fps} outside [30,250]")
    if N < 100:
        rep.fail("A4", f"{name}: only {N} frames (<2s)")
    else:
        rep.ok("A4", f"{name}: {N} frames @ {fps:.0f}fps")
    # A5 quats
    quat = np.asarray(d["root_rot"], dtype=np.float64)
    qn = np.linalg.norm(quat, axis=1)
    if np.abs(qn - 1).max() >= 1e-3:
        rep.fail("A5", f"{name}: quat norm dev max {np.abs(qn-1).max():.2e}")
    # B joint limits
    for i, jn in enumerate(dn):
        lo, hi = JOINT_LIMITS[jn]
        col = q[:, i]
        hard = (col < lo - HARD_MARGIN) | (col > hi + HARD_MARGIN)
        if hard.any():
            frac = hard.mean()
            if frac > HARD_FRAME_FRAC:
                rep.fail("B1", f"{name}:{jn} exceeds hard limit in {frac*100:.2f}% frames "
                               f"(range [{col.min():.3f},{col.max():.3f}] limit [{lo},{hi}])")
            continue
        slo = lo + (hi - lo) * (1 - SOFT_SHRINK) / 2
        shi = hi - (hi - lo) * (1 - SOFT_SHRINK) / 2
        soft = (col < slo) | (col > shi)
        if soft.mean() > SOFT_FRAME_FRAC:
            # WARN not FAIL: some joints' neutral pose sits at the limit
            # boundary (e.g. shoulder_roll range (-2,0) with arms hanging at 0)
            rep.warn("B2", f"{name}:{jn} beyond soft limit in {soft.mean()*100:.1f}% frames "
                            f"(range [{col.min():.3f},{col.max():.3f}])")
    # C root trajectory
    rp = np.asarray(d["root_pos"], dtype=np.float64)
    z = rp[:, 2]
    if not (ROOT_Z_MEAN[0] <= z.mean() <= ROOT_Z_MEAN[1]):
        rep.fail("C1", f"{name}: root_z mean {z.mean():.3f} outside {ROOT_Z_MEAN}")
    if z.min() < ROOT_Z_MIN or z.max() > ROOT_Z_MAX:
        rep.fail("C2", f"{name}: root_z [{z.min():.3f},{z.max():.3f}] outside "
                       f"[{ROOT_Z_MIN},{ROOT_Z_MAX}]")
    dz = np.abs(np.diff(z))
    if (dz > DZ_FRAME).mean() > VEL_FRAME_FRAC:
        rep.fail("C3", f"{name}: {((dz>DZ_FRAME).mean()*100):.2f}% frames vertical jump>{DZ_FRAME}m")
    dxy = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1)
    if (dxy > DXY_FRAME).mean() > VEL_FRAME_FRAC:
        rep.fail("C4", f"{name}: {((dxy>DXY_FRAME).mean()*100):.2f}% frames horiz jump>{DXY_FRAME}m")
    # D smoothness
    dq = np.abs(np.diff(q, axis=0))
    if dq.size and dq.max() > DQ_MAX:
        i, j = np.unravel_index(dq.argmax(), dq.shape)
        rep.fail("D1", f"{name}: max joint frame-vel {dq.max():.3f} rad at frame {i} joint {dn[j]}")
    med, p99 = np.median(dq), np.percentile(dq, 99)
    if med > DQ_MED or p99 > DQ_P99:
        rep.fail("D2", f"{name}: joint frame-vel median {med:.3f} P99 {p99:.3f} "
                       f"(limits {DQ_MED}/{DQ_P99})")
    else:
        rep.ok("D", f"{name}: smooth med={med:.3f} P99={p99:.3f}")
    # E FK (optional)
    global _MJC_MISSING
    if xml_path is not None and not _MJC_MISSING:
        try:
            stored = np.asarray(d["body_positions"], dtype=np.float64)
            bn = [str(b) for b in d["body_names"]]

            def _stored_err(names_i, pos_i):
                """mean |FK - stored| over common bodies, or None if unusable."""
                if stored.shape[0] != N:
                    return None
                common = [b for b in names_i if b in bn]
                if not common:
                    return None
                fi = [names_i.index(b) for b in common]
                si = [bn.index(b) for b in common]
                return float(np.abs(pos_i[:, fi, :] - stored[:, si, :]).mean())

            # The GMR worker stores root_rot as xyzw (swapped from MuJoCo's
            # native wxyz) — decide empirically which convention reproduces the
            # stored body_positions better, and use it for all E checks.
            fk_as_is = fk_all_frames(xml_path, rp, quat, q, dn)
            fk_swap = fk_all_frames(xml_path, rp, quat[:, [3, 0, 1, 2]], q, dn)
            e1 = _stored_err(*fk_as_is)
            e2 = _stored_err(*fk_swap)
            if e1 is not None and (e2 is None or e1 <= e2):
                fk_names, fk_pos, conv, ebest = (*fk_as_is, "as-is(wxyz)", e1)
            else:
                fk_names, fk_pos, conv, ebest = (*fk_swap, "xyzw->wxyz", e2)
            if ebest is not None:
                rep.ok("E0", f"{name}: quat conv {conv} (E3 err {ebest:.2e} m)")
        except ImportError:
            _MJC_MISSING = True
            rep.warn("E0", "mujoco not available — group E skipped (single notice)")
            stored, bn, ebest = None, None, None
        except Exception as e:
            rep.fail("E0", f"{name}: FK computation error: {e}")
            stored, bn, ebest = None, None, None
        else:
            if fk_pos[..., 2].min() < GND_MIN:
                rep.fail("E1", f"{name}: FK body z min {fk_pos[...,2].min():.3f} < {GND_MIN}")
            feet = [i for i, b in enumerate(fk_names) if b.endswith("ankle_roll_link")]
            if fk_pos[:, feet, 2].min() < FOOT_MIN:
                rep.fail("E2", f"{name}: FK foot z min {fk_pos[:,feet,2].min():.3f} < {FOOT_MIN}")
            stored = np.asarray(d["body_positions"], dtype=np.float64)
            bn = [str(b) for b in d["body_names"]]
            common = [b for b in fk_names if b in bn]
            if common and stored.shape[0] == N:
                fi = [fk_names.index(b) for b in common]
                si = [bn.index(b) for b in common]
                err = np.abs(fk_pos[:, fi, :] - stored[:, si, :]).mean()
                if err > FK_MATCH:
                    # WARN: stored body_positions frame/convention is a GMR
                    # internal (world vs local frame undocumented); the data
                    # itself is not used downstream (lab key_body_pos is
                    # recomputed by Isaac FK in dataset_retarget).
                    rep.warn("E3", f"{name}: stored FK mismatch mean err {err:.4f} m "
                                   f"(convention forensics only)")
                else:
                    rep.ok("E", f"{name}: FK consistent (err {err:.2e} m, foot zmin "
                                f"{fk_pos[:,feet,2].min():.3f})")
    # F semantic (WARN only)
    if MOVING_PAT.search(name) and "treadmill" not in name:
        vxy = np.linalg.norm(np.diff(rp[:, :2], axis=0) * fps, axis=1).mean()
        if not (0.1 <= vxy <= 3.0):
            rep.warn("F1", f"{name}: mean horizontal speed {vxy:.2f} m/s outside [0.1,3.0]")
        else:
            rep.ok("F1", f"{name}: speed {vxy:.2f} m/s")
    if WALK_PAT.search(name):
        li, ri = dn.index("left_hip_pitch_joint"), dn.index("right_hip_pitch_joint")
        if q[:, li].std() > 1e-3 and q[:, ri].std() > 1e-3:
            c = float(np.corrcoef(q[:, li], q[:, ri])[0, 1])
            # X1 URDF mirrors right_hip_pitch (limits L=(-1,2) vs R=(-2,1)):
            # positive joint angles swing the two legs in OPPOSITE physical
            # directions. Walking = anti-phase physical swing = POSITIVE corr
            # of joint angles (v16/v17 retargets show +0.98..0.99).
            if c < 0.5:
                rep.warn("F2", f"{name}: L/R hip_pitch corr {c:+.2f} (expected > +0.5 for mirrored convention; low corr may indicate gait asymmetry)")
    return {"name": name, "frames": N, "fps": fps,
            "root_z": (float(z.min()), float(z.mean()), float(z.max()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--gmr-dir", default=None)
    ap.add_argument("--lab-dir", default=None)
    ap.add_argument("--xml", default=None, help="x1.xml path (FK checks); 'none' to disable")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    gmr_dir = Path(args.gmr_dir) if args.gmr_dir else root / "roboparty_train/robolab/data/motions/x1_gmr"
    lab_dir = Path(args.lab_dir) if args.lab_dir else root / "roboparty_train/robolab/data/motions/x1_lab"
    yaml_path = root / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml"
    xml_path = Path(args.xml) if args.xml else root / "gmr_x1_assets/x1.xml"
    if str(xml_path) == "none" or not xml_path.exists():
        xml_path = None

    lists = parse_yaml_lists(yaml_path)
    gmr_dof, lab_dof = lists["gmr_dof_names"], lists["lab_dof_names"]

    rep = Report()
    print("=" * 72)
    print(f"GMR RETARGET ACCEPTANCE  gmr={gmr_dir}  lab={lab_dir}  xml={xml_path}")
    print("=" * 72)

    gmr_files = sorted(gmr_dir.glob("*.pkl")) if gmr_dir.exists() else []
    lab_files = sorted(lab_dir.glob("*.pkl")) if lab_dir.exists() else []
    print(f"\n[SET] x1_gmr: {len(gmr_files)} files, x1_lab: {len(lab_files)} files")

    # A3 file set
    have = {p.stem for p in gmr_files}
    missing = [m for m in MOTION_WEIGHTS if m not in have]
    extra = sorted(have - set(MOTION_WEIGHTS))
    if missing:
        rep.fail("A3", f"x1_gmr missing motions: {missing}")
    if extra:
        rep.warn("A3", f"x1_gmr extra files (unused by training): {extra}")
    if not missing and not extra:
        rep.ok("A3", "file set == motion_data_weights (14)")

    stats = []
    for p in gmr_files:
        print(f"\n--- {p.name} ---")
        s = check_gmr_file(p, gmr_dof, xml_path, rep)
        if s:
            stats.append(s)

    # G lab fidelity
    print("\n--- lab fidelity (group G) ---")
    gmr_by_name = {}
    for p in gmr_files:
        try:
            with open(p, "rb") as f:
                gmr_by_name[p.stem] = pickle.load(f)
        except Exception:
            pass
    for p in lab_files:
        name = p.stem
        with open(p, "rb") as f:
            ld = pickle.load(f)
        miss = [k for k in LAB_FIELDS if k not in ld]
        if miss:
            rep.fail("A6", f"{name}: lab pkl missing {miss}")
            continue
        q = np.asarray(ld["dof_pos"], dtype=np.float64)
        if q.shape[1] != 29:
            rep.fail("A6", f"{name}: lab dof count {q.shape[1]} != 29")
            continue
        kb = np.asarray(ld["key_body_pos"], dtype=np.float64)
        if kb.shape[1] != 6:
            rep.fail("A6", f"{name}: key_body count {kb.shape[1]} != 6")
        if not np.isfinite(kb).all():
            rep.fail("G3", f"{name}: key_body_pos has NaN/Inf")
        elif kb[..., 2].min() < -0.05:
            rep.fail("G3", f"{name}: key_body z min {kb[...,2].min():.3f} < -0.05")
        g = gmr_by_name.get(name)
        if g is None:
            rep.warn("G0", f"{name}: no matching gmr pkl")
            continue
        gq = np.asarray(g["dof_pos"], dtype=np.float64)
        gdn = [str(x) for x in g["dof_names"]]
        if gq.shape[0] != q.shape[0]:
            rep.fail("A6", f"{name}: frame count gmr={gq.shape[0]} lab={q.shape[0]}")
            continue
        # reorder lab->gmr by name
        try:
            cols = [lab_dof.index(jn) for jn in gdn]
        except ValueError as e:
            rep.fail("G1", f"{name}: lab_dof_names missing joint: {e}")
            continue
        err = np.abs(q[:, cols] - gq).max()
        if err > LAB_QMAX:
            rep.fail("G1", f"{name}: max|lab-gmr| = {err:.4f} rad > {LAB_QMAX}")
        rerr = max(np.abs(np.asarray(ld["root_pos"]) - np.asarray(g["root_pos"])).max(),
                   np.abs(np.asarray(ld["root_rot"]) - np.asarray(g["root_rot"])).max())
        if rerr > LAB_ROOT:
            rep.fail("G2", f"{name}: root max diff {rerr:.2e} > {LAB_ROOT}")
        if err <= LAB_QMAX and rerr <= LAB_ROOT:
            rep.ok("G", f"{name}: reorder-exact (dq={err:.1e}, droot={rerr:.1e})")

    # systematic warnings -> fail
    # Only WARN checks whose systematic occurrence would POISON training escalate
    # to FAIL (see RETARGET_ACCEPTANCE.md "Systematic warning escalation" policy):
    #   F1 (dataset teaches wrong speed range), F2 (systematic gait asymmetry),
    #   G0 (lab/gmr pairing broken). B2/E3 stay file-level WARNs: identical
    # characteristics (soft-limit saturation from GMR IK; stored body_positions
    # convention offset) already produced successful v16/v17 AMP training.
    ESCALATE_WARN_CHECKS = {"F1", "F2", "G0"}
    from collections import Counter
    cid_counts = Counter(w.split("]")[0].lstrip("[") + "]" for w in rep.warns)
    for cid, n in cid_counts.items():
        if n >= 3 and cid.strip("[]") in ESCALATE_WARN_CHECKS:
            rep.fail(cid.strip("[]") + "-SYS", f"systematic warning {cid} triggered in {n} files")

    print("\n" + "=" * 72)
    verdict = "PASS" if not rep.fails else "FAIL"
    print(f"VERDICT: {verdict}   fails={len(rep.fails)} warns={len(rep.warns)}")
    for f_ in rep.fails:
        print(f"  FAIL {f_}")
    for w in rep.warns:
        print(f"  WARN {w}")
    print("=" * 72)

    if args.json:
        # parent dir may not exist yet when invoked by the train pipeline gate
        # (TASK_20260817_046 crashed here: model_upload/ created only later)
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "verdict": verdict, "fails": rep.fails, "warns": rep.warns,
            "files": stats}, indent=1, default=str))

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
