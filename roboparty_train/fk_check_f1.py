#!/usr/bin/env python3
"""Zero-pose + bent-knee FK sanity for F1_29dof vs X1 (pure numpy, no sim deps).

Checks:
  1. world-frame rotation axes of the 12 leg joints (hip redesign shift)
  2. foot (ankle_roll_link / toe) placement at q=0: symmetry, height, hip width
  3. standing height with knee=0.6, ankle compensation (compare robots)
"""
import numpy as np
import xml.etree.ElementTree as ET

ROOT = "/Users/yumx/code/X1_29_AMP"


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def load(path):
    r = ET.parse(path).getroot()
    joints = {}
    for j in r.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        ax = j.find("axis")
        axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()])
        joints[j.get("name")] = dict(
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=xyz, R=rpy_to_R(*rpy), axis=axis, type=j.get("type"),
            limit=(j.find("limit") is not None and dict(j.find("limit").attrib) or None))
    return joints


def fk(joints, chain, q):
    """chain: joint names from base; q: angles for revolute (0 for fixed)."""
    T = np.eye(4)
    axes_world = []
    qi = iter(q)
    for jn in chain:
        jd = joints[jn]
        T = T @ pose(jd, (next(qi) if jd["type"] == "revolute" else 0.0))
        axes_world.append(T[:3, :3] @ jd["axis"])
    return T, axes_world


def pose(jd, q):
    T = np.eye(4)
    T[:3, :3] = jd["R"]
    T[:3, 3] = jd["xyz"]
    if q:
        a = jd["axis"] / np.linalg.norm(jd["axis"])
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        T[:3, :3] = T[:3, :3] @ (np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K)
    return T


LEG_CHAINS = {
    "L": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
          "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    "R": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
          "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
}
TOE_CHAIN = {"L": ["leg_l_toe_a_link_joint", "leg_l_toe_a_ball_joint", "leg_l_toe_a_loop_joint"],
             "R": ["leg_r_toe_a_link_joint", "leg_r_toe_a_ball_joint", "leg_r_toe_a_loop_joint"]}


def report(name, path, toe_from_f1_naming):
    joints = load(path)
    print(f"\n===== {name} =====")
    # axes at zero pose
    for side, chain in LEG_CHAINS.items():
        _, axes = fk(joints, chain, [0.0] * 6)
        for jn, a in zip(chain, axes):
            print(f"  {side} {jn:28s} axis_world={np.round(a, 4)}")
        break  # left only for brevity
    for qk, qs in [("q=0", [0] * 6), ("knee=0.6", [0, 0, 0, 0.6, -0.3, 0])]:
        pos = {}
        for side, chain in LEG_CHAINS.items():
            T, _ = fk(joints, chain, qs)
            pos[side] = T[:3, 3]
        mid = (pos["L"] + pos["R"]) / 2
        print(f"  [{qk}] ankle_L={np.round(pos['L'],4)} ankle_R={np.round(pos['R'],4)}")
        print(f"        hip_width(y)={abs(pos['L'][1]-pos['R'][1])*1000:.1f} mm  ankle_z={pos['L'][2]*1000:.1f}/{pos['R'][2]*1000:.1f} mm")
        if qs[3]:
            print(f"        mid_ankle={np.round(mid,4)} (x asym {(pos['L'][0]-pos['R'][0])*1000:+.2f} mm)")
    # toe tips (naming differs: X1 leg_l_toe_a_joint vs F1 leg_l_toe_a_link_joint)
    toe = TOE_CHAIN["L"] if toe_from_f1_naming else ["leg_l_toe_a_joint", "leg_l_toe_a_ball_joint", "leg_l_toe_a_loop_joint"]
    Tl, _ = fk(joints, LEG_CHAINS["L"] + toe, [0.0] * 9)
    print(f"  toe_a_loop frame origin (q=0): {np.round(Tl[:3,3],4)}")
    # joint limits
    print("  limits(L):", {c: (joints[c]['limit']['lower'], joints[c]['limit']['upper']) for c in LEG_CHAINS['L']})


if __name__ == "__main__":
    report("X1 (reference)", f"{ROOT}/X1_29DOF/urdf/f1.urdf", False)
    report("F1 merged", f"{ROOT}/F1_29DOF/urdf/f1_29dof.urdf", True)
