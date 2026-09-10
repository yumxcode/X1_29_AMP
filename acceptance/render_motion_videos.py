#!/usr/bin/env python3
"""Render retargeted X1 motion clips to MP4 videos on disk (v30 evidence).

Two products per clip:
  1. side-by-side comparison  ORIG (x1_lab, contorted v29 refs) | v30 (fixed)
  2. v30-only render

Conventions identical to fix_arm_decomposition.py (proven there: FK vs stored
key_body_pos <= 12 mm): x1_lab / x1_lab_v30 pkl = wxyz quat + Isaac Lab dof
order, remapped onto gmr_x1_assets/x1.xml (build_model adds the floor).

Camera tracks the root (walk clips travel several meters; a static camera
loses the robot). Labels are burned in with PIL.

Usage:
  ./.venv_test/bin/python acceptance/render_motion_videos.py \
      [--clips 0005_normal_walk1 ...] [--out acceptance/videos_v30] [--only-cmp]
"""
import argparse
import functools
import pickle
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim2sim"))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list  # noqa: E402

XML = ROOT / "gmr_x1_assets" / "x1.xml"
YAML = ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml"
ORIG_DIR = ROOT / "roboparty_train/robolab/data/motions/x1_lab"
V30_DIR = ROOT / "roboparty_train/robolab/data/motions/x1_lab_v30"

# the 10 clips kept in v30 training weights (source, non-mirror)
DEFAULT_CLIPS = ["0005_normal_walk1", "0007_normal_walk3", "0008_normal_walk4",
                 "0000_treadmill_norm", "0002_treadmill_slow",
                 "0009_normal_jog1", "0026_circle_walk", "36_01", "36_11"]


def build_qpos_array(model, clip, qadr):
    rp = np.asarray(clip["root_pos"], dtype=np.float64)
    rr = np.asarray(clip["root_rot"], dtype=np.float64)
    q = np.asarray(clip["dof_pos"], dtype=np.float64)
    T = len(q)
    qpos = np.zeros((T, model.nq))
    qpos[:, 0:3] = rp
    qpos[:, 3:7] = rr / np.linalg.norm(rr, axis=1, keepdims=True)
    qpos[:, qadr] = q
    return qpos


def render_clip(model, data, qpos, label, height=540, width=960):
    """Yield rendered frames with the camera tracking the root."""
    import mujoco
    rend = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 130, -15, 4.2
    for t in range(len(qpos)):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = qpos[t, 0:3] + np.array([0.05, 0.0, 0.05])
        rend.update_scene(data, camera=cam)
        yield label_frame(rend.render(), label)
    rend.close()


def label_frame(frame_bgr, text):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame_bgr)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((10, 8), text, font=font)
    d.rectangle([x0 - 6, y0 - 4, x1 + 6, y1 + 6], fill=(0, 0, 0, 170))
    d.text((10, 8), text, fill=(255, 210, 60), font=font)
    return np.asarray(img)


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--out", default=str(ROOT / "acceptance/videos_v30"))
    ap.add_argument("--only-cmp", action="store_true", help="skip v30-only renders")
    ap.add_argument("--step", type=int, default=2, help="keep every Nth frame (2 -> 60fps)")
    ap.add_argument("--orig-dir", default=None)
    ap.add_argument("--fixed-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    global ORIG_DIR, V30_DIR
    if args.orig_dir:
        ORIG_DIR = Path(args.orig_dir)
    if args.fixed_dir:
        V30_DIR = Path(args.fixed_dir)

    lab_names = parse_yaml_list(YAML, "lab_dof_names")
    model, _ = build_model(XML)
    data = mujoco.MjData(model)
    # v30.2 FIX (critical): qadr must be built in LAB order so that q's lab
    # column i lands in lab_names[i]'s qpos slot. The first version indexed
    # qadr by MuJoCo hinge order while q stayed in lab order -> every joint
    # received a DIFFERENT joint's data (left hip drove the waist, etc.),
    # producing exactly the "chaotic feet / legs out of sync / clipping" the
    # user saw in the first videos. NEVER use chained fancy-index writes
    # (qpos[qadr][lab2mj] = q writes to a copy, not data.qpos).
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in lab_names}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in lab_names])
    assert len(qadr) == 29

    # FK self-check gate: rendered qpos must reproduce the pkl's stored
    # key_body_pos (schema order = KEY_BODIES below) within 15 mm, else abort.
    KEY_BODIES = ["left_knee_pitch_link", "right_knee_pitch_link",
                  "left_ankle_roll_link", "right_ankle_roll_link",
                  "left_elbow_yaw_link", "right_elbow_yaw_link"]
    kb_id = {b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in KEY_BODIES}

    def fk_check(clip, qpos):
        kb = np.asarray(clip["key_body_pos"])
        errs = []
        for t in np.linspace(0, len(qpos) - 1, 8).astype(int):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            errs.append(max(
                np.linalg.norm(data.xpos[kb_id[b]] - kb[t, k])
                for k, b in enumerate(KEY_BODIES)))
        e = float(np.mean(errs))
        if e > 0.015:
            sys.exit(f"[FATAL] render FK self-check failed: {e*1000:.1f} mm vs stored "
                     "key_body_pos — joint mapping is wrong, refusing to render")
        return e

    import imageio.v2 as imageio

    for name in args.clips:
        po, pv = ORIG_DIR / f"{name}.pkl", V30_DIR / f"{name}.pkl"
        if not pv.exists():
            print(f"[SKIP] {name}: no v30 clip")
            continue
        co = pickle.load(open(po, "rb"), encoding="latin1") if po.exists() else None
        cv = pickle.load(open(pv, "rb"), encoding="latin1")
        fps = float(cv["fps"])

        if co is not None:
            qo = build_qpos_array(model, co, qadr)
            qv = build_qpos_array(model, cv, qadr)
            e_o, e_v = fk_check(co, qo), fk_check(cv, qv)
            print(f"[CHK ] {name}: FK self-check orig {e_o*1000:.1f}mm v30 {e_v*1000:.1f}mm (<=15mm)")
            T = min(len(qo), len(qv))
            path = out / f"{name}_orig_vs_v30.mp4"
            gen_o = render_clip(model, data, qo, "ORIG x1_lab (v29 refs: elbP~106deg, lumY swing ~55deg)")
            gen_v = render_clip(model, data, qv, "v30 fixed (elbP 20deg, lumY ~24deg)")
            frames = []
            for t, (fo, fv) in enumerate(zip(gen_o, gen_v)):
                if t % args.step == 0:
                    frames.append(np.hstack([fo, fv]))
                if t >= T - 1:
                    break
            imageio.mimwrite(path, frames, fps=int(fps / args.step), quality=8,
                             macro_block_size=2)
            print(f"[CMP ] {name}: {len(frames)} frames @ {fps/args.step:.0f}fps -> {path.name} "
                  f"({path.stat().st_size // 1024}KB)")

        if not args.only_cmp:
            qv = build_qpos_array(model, cv, qadr)
            fk_check(cv, qv)
            path = out / f"{name}_v30.mp4"
            frames = [f for t, f in enumerate(render_clip(
                model, data, qv, f"v30 {name}")) if t % args.step == 0]
            imageio.mimwrite(path, frames, fps=int(fps / args.step), quality=8,
                             macro_block_size=2)
            print(f"[V30 ] {name}: {len(frames)} frames -> {path.name} "
                  f"({path.stat().st_size // 1024}KB)")

    print(f"\n[DONE] videos in {out}")


if __name__ == "__main__":
    main()
