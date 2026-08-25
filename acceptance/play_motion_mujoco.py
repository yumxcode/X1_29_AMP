#!/usr/bin/env python
"""Play retargeted X1 motion (.pkl from AMP motion lib) in MuJoCo viewer.

Data convention (verified against run_gmr_retarget.py + gmr_x1_assets/x1.xml):
  x1_gmr pkl : root_rot stored as XYZW (!) — converted to WXYZ on load;
               dof already in MJCF order (pkl carries dof_names)
  x1_lab pkl : root_rot already WXYZ; dof in Isaac Lab order — remapped via
               LAB_DOF_ORDER (from robolab retarget config x1.yaml)
  Both verified: FK vs stored body/key positions, err 0.000 m (gmr) / 0.001 m (lab)

Model: X1_29DOF/mjcf/xyber_x1_flat.xml  (freejoint base_link, nq=36)

Usage (use the x1 conda env which has mujoco 3.8):
  python  play_motion_mujoco.py            # pick motion interactively (auto switches to mjpython)
  mjpython play_motion_mujoco.py <pkl>     # viewer on macOS must use mjpython
  python  play_motion_mujoco.py <pkl> --record out.gif   # headless, no mjpython needed
"""
import argparse
import os
import pickle
import sys
import time

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(REPO, "X1_29DOF/mjcf/xyber_x1_flat.xml")
DEFAULT_DATA_ROOTS = [
    os.path.join(REPO, "acceptance/v25_unpacked"),
    os.path.join(REPO, "acceptance/v25_artifacts"),
]

# Isaac Lab joint order of x1_lab pkl dof columns (those files store no
# dof_names). Source of truth:
# roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml :: lab_dof_names
# (verified against Isaac Lab TASK_20260810_036)
LAB_DOF_ORDER = [
    "left_hip_pitch_joint", "lumbar_yaw_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "lumbar_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "lumbar_pitch_joint", "right_hip_yaw_joint",
    "left_knee_pitch_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "right_knee_pitch_joint",
    "left_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "right_ankle_roll_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]


def find_motions(path):
    """Return list of .pkl motion files for a given path (file / dir / search roots)."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        pkls = sorted(
            os.path.join(dp, f)
            for dp, _, fs in os.walk(path)
            for f in fs if f.endswith(".pkl")
        )
        return pkls
    # bare name → search under default roots
    for root in DEFAULT_DATA_ROOTS:
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f == path or f == path + ".pkl":
                    return [os.path.join(dp, f)]
    return []


def load_motion(path):
    with open(path, "rb") as f:
        m = pickle.load(f)
    fps = float(m.get("fps", 120.0))
    root_pos = np.asarray(m["root_pos"], dtype=np.float64)
    root_rot = np.asarray(m["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(m["dof_pos"], dtype=np.float64)
    if "dof_names" in m:
        # GMR format: quaternions saved as XYZW (run_gmr_retarget.py:331
        # applies [1,2,3,0] i.e. wxyz->xyzw before saving). Convert back.
        fmt = "gmr"
        root_rot = root_rot[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        dof_names = list(m["dof_names"])
    else:
        # Isaac Lab training format: quat already WXYZ, dof in Isaac order.
        fmt = "lab"
        dof_names = list(LAB_DOF_ORDER)
    return fps, root_pos, root_rot, dof_pos, dof_names, fmt, m


def remap_dof(model, dof_pos, dof_names, joint_names):
    """Reorder dof columns from data names to model qpos order, if names given."""
    if dof_names is None:
        if dof_pos.shape[1] != len(joint_names):
            sys.exit(f"DOF count mismatch: data {dof_pos.shape[1]} vs model {len(joint_names)}"
                     " and pkl has no dof_names to remap")
        return dof_pos  # assume identical order (verified: gmr xml == mjcf xml)
    idx = {n: i for i, n in enumerate(dof_names)}
    missing = [n for n in joint_names if n not in idx]
    if missing:
        sys.exit(f"data dof_names missing model joints: {missing}")
    cols = np.array([idx[n] for n in joint_names])
    return dof_pos[:, cols]


def check_quat_wxyz(root_rot):
    """Heuristic sanity: a WXYZ quat is fine; warn if it looks flipped/nan."""
    norms = np.linalg.norm(root_rot, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print(f"[warn] quat norms deviate [{norms.min():.4f},{norms.max():.4f}], renormalizing")
        root_rot /= norms[:, None]


def verify_against_stored(model, data, motion, qpos, fmt):
    """Sanity-check loaded conventions vs stored FK truth (if present)."""
    names_truth = None
    if fmt == "gmr" and "body_positions" in motion:
        truth = np.asarray(motion["body_positions"])       # (T,30,3)
        names_truth = list(motion["body_names"])
    elif fmt == "lab" and "key_body_pos" in motion:
        truth = np.asarray(motion["key_body_pos"])         # (T,6,3)
        names_truth = list(LAB_KEY_BODY_ORDER)
    else:
        return
    frames = np.linspace(0, qpos.shape[0] - 1, 8).astype(int)
    errs = []
    for t in frames:
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        root = data.xpos[model.body("base_link").id]
        rel_mj = np.array([data.xpos[model.body(n).id] - root for n in names_truth])
        rel_tr = truth[t] - truth[t, 0] if fmt == "gmr" else \
            truth[t] - np.asarray(motion["root_pos"])[t]
        errs.append(np.linalg.norm(rel_mj - rel_tr, axis=1).mean())
    tag = "OK" if np.mean(errs) < 0.05 else "MISMATCH"
    print(f"[verify-{fmt}] FK vs stored positions: mean err {np.mean(errs):.4f} m  [{tag}]")


LAB_KEY_BODY_ORDER = [
    "left_knee_pitch_link", "right_knee_pitch_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_elbow_yaw_link", "right_elbow_yaw_link",
]


def build_qpos(model, root_pos, root_rot, dof_pos, dof_names):
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    hinge_names = [n for n in joint_names if n != "floating_base"]
    dof_pos = remap_dof(model, dof_pos, dof_names, hinge_names)

    qpos = np.zeros((root_pos.shape[0], model.nq))
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_rot
    # map each hinge joint's qpos address in order
    addr = 0
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        qpos[:, model.jnt_qposadr[i]] = dof_pos[:, addr]
        addr += 1
    assert addr == dof_pos.shape[1]
    # NaN check
    if not np.isfinite(qpos).all():
        bad = np.where(~np.isfinite(qpos).all(axis=1))[0]
        sys.exit(f"non-finite qpos at frames {bad[:5]} ... aborting")
    return qpos


def play_interactive(model, data, qpos, fps, speed, loop):
    import platform
    import mujoco.viewer

    # MuJoCo viewer on macOS must run under mjpython (AppKit owns the main
    # thread). Auto re-exec ourselves with mjpython if needed.
    if platform.system() == "Darwin" and not isinstance(
            mujoco.viewer._MJPYTHON, mujoco.viewer._MjPythonBase):
        import shutil
        mjp = shutil.which("mjpython")
        if not mjp:
            sys.exit("macOS viewer needs mjpython (ships with the mujoco pkg).\n"
                     "Run instead:  mjpython " + " ".join(sys.argv))
        print("[info] re-launching under mjpython (required for viewer on macOS)",
              flush=True)
        os.execv(mjp, [mjp, os.path.abspath(__file__)] + sys.argv[1:])

    dt = 1.0 / (fps * speed)
    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 130, -15, 4.5
        print("viewer running — close the window to exit")
        while v.is_running():
            for t in range(qpos.shape[0]):
                if not v.is_running():
                    return
                data.qpos[:] = qpos[t]
                mujoco.mj_forward(model, data)
                v.sync()
                time.sleep(dt)
            if not loop:
                print("motion finished (use --loop to repeat)")
                # hold last frame a moment so it doesn't vanish instantly
                for _ in range(120):
                    if not v.is_running():
                        return
                    time.sleep(dt)
                return


def record_video(model, data, qpos, fps, out_path, speed=1.0, width=960, height=540):
    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None
    if imageio is None and not out_path.lower().endswith((".gif", ".png")):
        alt = os.path.splitext(out_path)[0] + ".gif"
        print(f"[info] imageio unavailable — falling back to GIF: {alt}")
        out_path = alt

    model.vis.global_.offwidth = width    # enlarge offscreen framebuffer
    model.vis.global_.offheight = height  # (default is 640x480)
    rend = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 130, -15, 4.5
    frames = []
    step = max(1, int(round(speed)))  # speed N = keep every Nth frame (N× faster)
    for t in range(0, qpos.shape[0], step):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        rend.update_scene(data, camera=cam)
        frames.append(rend.render())

    if imageio is not None:
        imageio.mimwrite(out_path, frames, fps=int(fps / step), quality=8)
    else:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / (fps / step)), loop=0)
    print(f"recorded {len(frames)} frames -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motion", nargs="?", help=".pkl file, dir, or bare name to search")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true", help="loop playback")
    ap.add_argument("--record", metavar="MP4", help="offscreen render to mp4 instead of viewer")
    ap.add_argument("--list", action="store_true", help="list available motions and exit")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    # locate motion file(s)
    motions = find_motions(args.motion) if args.motion else []
    if args.list or not motions:
        motions = find_motions(DEFAULT_DATA_ROOTS[0])
    if not motions:
        sys.exit(f"no .pkl found under {DEFAULT_DATA_ROOTS[0]}")
    if args.list or not args.motion:
        print("available motions:")
        for i, p in enumerate(motions):
            print(f"  [{i:2d}] {os.path.relpath(p, REPO)}")
        if args.list:
            return
        try:
            sel = input("select index: ").strip()
            path = motions[int(sel)]
        except (ValueError, IndexError, EOFError):
            sys.exit("no valid selection")
    else:
        path = motions[0] if len(motions) == 1 else None
        if path is None:
            print("multiple matches:")
            for i, p in enumerate(motions):
                print(f"  [{i:2d}] {os.path.relpath(p, REPO)}")
            sel = input("select index: ").strip()
            path = motions[int(sel)]

    fps, root_pos, root_rot, dof_pos, dof_names, fmt, motion = load_motion(path)
    check_quat_wxyz(root_rot)
    qpos = build_qpos(model, root_pos, root_rot, dof_pos, dof_names)
    verify_against_stored(model, data, motion, qpos, fmt)

    print(f"\nmotion : {os.path.relpath(path, REPO)}  [{fmt}]")
    print(f"frames: {qpos.shape[0]}  @ {fps:.0f} fps  ({qpos.shape[0]/fps:.1f}s)  "
          f"| nq={model.nq} | speed={args.speed}x")

    if args.record:
        record_video(model, data, qpos, fps, args.record, speed=args.speed)
    else:
        play_interactive(model, data, qpos, fps, args.speed, args.loop)


if __name__ == "__main__":
    main()
