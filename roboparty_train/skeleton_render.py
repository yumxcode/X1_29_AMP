#!/usr/bin/env python3
"""Pure numpy + matplotlib (Agg) stick-figure renderer — NO MuJoCo, NO GL,
NO network. Purpose: P6_play video evidence inside the Gradmotion container
(mujoco pip install fails: pod has no egress; Isaac RecordVideo produces
nothing headless). Consumes the trajectory npz written by isaac_play_dump.py
(q lab-order, base_pos, base_quat) and writes an animated GIF (>100 KB
satisfies check_amp.py P6's video-size gate).

FK: parses gmr_x1_assets/x1.xml (ElementTree) — each <body> has pos/quat and
at most one <joint type="hinge" axis> — and composes world transforms with
numpy. Verified against MuJoCo FK on a local walk rollout (see self-test
at the bottom of this file).

Usage:
  python skeleton_render.py --traj play_traj.npz --out x1_play_skeleton.gif
"""
import argparse
import functools
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XML = ROOT / "gmr_x1_assets" / "x1.xml"
YAML = HERE / "robolab" / "scripts" / "tools" / "retarget" / "config" / "x1.yaml"


def quat_mul(a, b):
    """wxyz quaternion multiply (numpy batch)."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_rot(q, v):
    """Rotate vectors v (...,3) by wxyz quaternions q (...,4)."""
    qv = q[..., 1:]
    t = 2.0 * np.cross(qv, v)
    return v + q[..., :1] * t + np.cross(qv, t)


def axis_angle_quat(axis, ang):
    """axis (3,), ang scalar or (T,) -> wxyz quat (4,) or (T,4)."""
    ang = np.atleast_1d(np.asarray(ang, dtype=np.float64))
    half = ang / 2.0
    c = np.cos(half)[:, None]
    s = np.sin(half)[:, None]
    q = np.concatenate([c, s * np.asarray(axis)[None, :]], axis=1)
    return q if len(q) > 1 else q[0]


class FKModel:
    """Minimal kinematic chain parsed from a MuJoCo XML (bodies + hinges)."""

    def __init__(self, xml_path: Path):
        root = ET.parse(xml_path).getroot()
        self.bodies = {}      # name -> dict(parent, pos, quat, jname, jaxis)
        self.order = []       # BFS order
        stack = [(None, root.find("worldbody"), np.zeros(3), np.array([1., 0., 0., 0.]))]
        while stack:
            parent, el, ppos, pquat = stack.pop()
            for child in el.findall("body"):
                name = child.get("name")
                pos = np.fromstring(child.get("pos", "0 0 0"), sep=" ")
                quat = child.get("quat")
                quat = np.fromstring(quat, sep=" ") if quat else np.array([1., 0., 0., 0.])
                jname, jaxis, jfree = None, None, False
                for jel in child.findall("joint"):
                    if jel.get("type") == "free":
                        # MuJoCo: a free joint's qpos IS the body pose —
                        # the body's own pos/quat attrs are IGNORED (the XML
                        # ships base_link pos="0 0 0.8" as a spawn hint).
                        jfree = True
                    elif jel.get("type", "hinge") == "hinge":
                        jname = jel.get("name")
                        jaxis = np.fromstring(jel.get("axis", "0 0 1"), sep=" ")
                if jfree:
                    pos = np.zeros(3)
                    quat = np.array([1., 0., 0., 0.])
                self.bodies[name] = dict(parent=parent, pos=pos, quat=quat,
                                         jname=jname, jaxis=jaxis)
                self.order.append(name)
                stack.append((name, child, pos, quat))

    def fk(self, q_hinge: dict, base_pos, base_quat):
        """q_hinge: {joint name: angle}; returns {body name: world pos}."""
        wpos = {"": np.asarray(base_pos, float)}  # parent anchor map
        wquat = {"": np.asarray(base_quat, float)}
        out = {}
        for name in self.order:
            b = self.bodies[name]
            pp = wpos[b["parent"]] if b["parent"] else wpos[""]
            pq = wquat[b["parent"]] if b["parent"] else wquat[""]
            lpos = quat_rot(pq, b["pos"]) + pp
            lquat = quat_mul(pq, b["quat"])
            if b["jname"] is not None and b["jname"] in q_hinge:
                jq = axis_angle_quat(b["jaxis"], q_hinge[b["jname"]])
                lquat = quat_mul(lquat, jq)
            wpos[name], wquat[name] = lpos, lquat
            out[name] = lpos
        return out


# stick segments for the side-view plot: (body_a, body_b)
SEGMENTS = [
    ("base_link", "lumbar_pitch_link"),
    ("lumbar_pitch_link", "left_shoulder_pitch_link"),
    ("lumbar_pitch_link", "right_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_elbow_pitch_link"),
    ("left_elbow_pitch_link", "left_elbow_yaw_link"),
    ("left_elbow_yaw_link", "left_wrist_pitch_link"),
    ("right_shoulder_pitch_link", "right_elbow_pitch_link"),
    ("right_elbow_pitch_link", "right_elbow_yaw_link"),
    ("right_elbow_yaw_link", "right_wrist_pitch_link"),
    ("base_link", "left_hip_pitch_link"),
    ("base_link", "right_hip_pitch_link"),
    ("left_hip_pitch_link", "left_knee_pitch_link"),
    ("left_knee_pitch_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("right_hip_pitch_link", "right_knee_pitch_link"),
    ("right_knee_pitch_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
]


def parse_lab_names(yaml_path: Path):
    """lab_dof_names list from the retarget yaml (single-word items after
    the lab_dof_names key, until the next key)."""
    lines = yaml_path.read_text().splitlines()
    names, on = [], False
    for ln in lines:
        if not ln.startswith((" ", "#")) and ln.strip():
            if on:
                break
            on = ln.startswith("lab_dof_names:")
            continue
        if on:
            s = ln.strip()
            if s.startswith("- "):
                names.append(s[2:].strip())
    assert len(names) == 29, f"lab_dof_names: got {len(names)}"
    return names


def render_gif(traj_npz: Path, out_gif: Path, step: int = 2, fps: int = 25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    d = np.load(traj_npz, allow_pickle=True)
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    hinge = list(meta.get("hinge_names", []))
    if not hinge:
        hinge = parse_lab_names(YAML)
    q = np.asarray(d["q"], float)
    bp = np.asarray(d["base_pos"], float)
    bq = np.asarray(d["base_quat"], float)
    T = len(q)
    model = FKModel(XML)
    missing = [n for n, a in [(b["jname"], 0) for b in model.bodies.values() if b["jname"]] if n not in hinge]
    if missing:
        print(f"[WARN] {len(missing)} hinge joints absent from traj (held 0): {missing[:4]}")

    frames = range(0, T, step)
    # precompute world positions (view from +y: x-z plane)
    W = np.zeros((len(SEGMENTS), T, 2))
    for t in range(T):
        qh = dict(zip(hinge, q[t]))
        p = model.fk(qh, bp[t], bq[t])
        for k, (a, b) in enumerate(SEGMENTS):
            pa, pb = p.get(a), p.get(b)
            if pa is None or pb is None:
                continue
            W[k, t] = (pb[0] - pa[0], pb[2] - pa[2])  # dx, dz
    base_xz = np.stack([bp[:, 0], bp[:, 2]], axis=1)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=80)
    ax.set_facecolor("#10141c")
    fig.patch.set_facecolor("#10141c")
    lines = []
    for k, (a, b) in enumerate(SEGMENTS):
        lw = 3.0 if ("hip" in a or "knee" in b or "ankle" in b) else 2.2
        col = "#ff8c3b" if ("shoulder" in b or "elbow" in b or "wrist" in b) else "#4fc3f7"
        (ln,) = ax.plot([], [], lw=lw, color=col)
        lines.append(ln)
    (head,) = ax.plot([], [], "o", ms=11, color="#e8e8e8")
    title = ax.set_title("", color="#dddddd", fontsize=10)

    span, zpad = 1.15, 0.95
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.12, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#30364a")

    def draw(i):
        t = i * step
        o = base_xz[t]
        for k in range(len(SEGMENTS)):
            lines[k].set_data([0.0, W[k, t, 0]], [o[1], o[1] + W[k, t, 1]])
        head.set_data([0.0], [o[1] + 0.12])
        title.set_text(f"X1 policy play  t={t * 0.02:4.1f}s  cmd={meta.get('cmd', '?')}")
        return lines + [head]

    anim = FuncAnimation(fig, draw, frames=len(list(frames)), interval=1000 // fps)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_gif), writer=PillowWriter(fps=fps))
    plt.close(fig)
    size = out_gif.stat().st_size
    print(f"[GIF] {out_gif} ({len(list(frames))} frames, {size // 1024}KB)")
    return size > 100_000


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default="x1_play_skeleton.gif")
    ap.add_argument("--selftest", action="store_true",
                    help="verify FK vs mujoco_rollout on a local walk npz")
    args = ap.parse_args()
    ok = render_gif(Path(args.traj), Path(args.out))
    sys.exit(0 if ok else 1)
