#!/usr/bin/env python3
"""Generate the F1 sim2sim MJCF from F1_29DOF/urdf/f1_29dof.urdf.

Output (X1-serial-xml compatible — mujoco_rollout.py / gait_metrics.py
consume this structure):
  F1_29DOF/mjcf/robot/xyber_f1/xyber_f1_serial.xml
  F1_29DOF/mjcf/xyber_f1_flat.xml            (include: robot + flat env)
  F1_29DOF/mjcf/environment/flat.xml         (copied from X1)

Structure follows X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml exactly:
  - worldbody -> base_link(pos 0 0 0.8) with freejoint, 29-joint tree in
    body order [lumbar, L arm, R arm, L leg, R leg] (qpos = 17 upper + 12 legs)
  - body pos/quat from URDF joint origins (eulerseq XYZ)
  - inertial: URDF mass/com + principal-axis decomposition of the URDF
    inertia tensor composed with the inertial-origin rpy
  - hinge joints: axis from URDF; damping/frictionloss/range from the X1
    serial xml mapping (joint names are identical between X1 and F1)
  - visual geoms from URDF visual origins (pos/euler/scale; F1 ships some
    mm-scale meshes with scale 0.001 — carried over verbatim)
  - collision: base mesh (class collision) + 4 contact spheres per foot
    (X1 foot corner coordinates; F1 foot geometry is identical)
  - 29 <motor> actuators, X1 ctrlranges
  - sensors: jointpos + jointvel x29, X1 set
  - keyframe home_default: FK-verified F1 standing pose, base z from
    sole-contact computation

Run: .venv-mac/bin/python roboparty_train/build_f1_mjcf.py
"""
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F1_URDF = os.path.join(ROOT, "F1_29DOF/urdf/f1_29dof.urdf")
OUT_DIR = os.path.join(ROOT, "F1_29DOF/mjcf/robot/xyber_f1")
OUT_XML = os.path.join(OUT_DIR, "xyber_f1_serial.xml")
X1_XML = os.path.join(ROOT, "X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml")

# body (and joint) order — X1 serial xml tree order (qpos layout!)
BODY_ORDER = [
    # upper chain (base_link -> lumbar_yaw -> ...)
    ("lumbar_yaw_joint", "lumbar_yaw_link"),
    ("lumbar_roll_joint", "lumbar_roll_link"),
    ("lumbar_pitch_joint", "lumbar_pitch_link"),
    ("left_shoulder_pitch_joint", "left_shoulder_pitch_link"),
    ("left_shoulder_roll_joint", "left_shoulder_roll_link"),
    ("left_shoulder_yaw_joint", "left_shoulder_yaw_link"),
    ("left_elbow_pitch_joint", "left_elbow_pitch_link"),
    ("left_elbow_yaw_joint", "left_elbow_yaw_link"),
    ("left_wrist_pitch_joint", "left_wrist_pitch_link"),
    ("left_wrist_roll_joint", "left_wrist_roll_link"),
    ("right_shoulder_pitch_joint", "right_shoulder_pitch_link"),
    ("right_shoulder_roll_joint", "right_shoulder_roll_link"),
    ("right_shoulder_yaw_joint", "right_shoulder_yaw_link"),
    ("right_elbow_pitch_joint", "right_elbow_pitch_link"),
    ("right_elbow_yaw_joint", "right_elbow_yaw_link"),
    ("right_wrist_pitch_joint", "right_wrist_pitch_link"),
    ("right_wrist_roll_joint", "right_wrist_roll_link"),
    # legs
    ("left_hip_pitch_joint", "left_hip_pitch_link"),
    ("left_hip_roll_joint", "left_hip_roll_link"),
    ("left_hip_yaw_joint", "left_hip_yaw_link"),
    ("left_knee_pitch_joint", "left_knee_pitch_link"),
    ("left_ankle_pitch_joint", "left_ankle_pitch_link"),
    ("left_ankle_roll_joint", "left_ankle_roll_link"),
    ("right_hip_pitch_joint", "right_hip_pitch_link"),
    ("right_hip_roll_joint", "right_hip_roll_link"),
    ("right_hip_yaw_joint", "right_hip_yaw_link"),
    ("right_knee_pitch_joint", "right_knee_pitch_link"),
    ("right_ankle_pitch_joint", "right_ankle_pitch_link"),
    ("right_ankle_roll_joint", "right_ankle_roll_link"),
]
PARENT_OF = {
    "lumbar_yaw_joint": "base_link",
    "left_shoulder_pitch_joint": "lumbar_pitch_joint",
    "right_shoulder_pitch_joint": "lumbar_pitch_joint",
    "left_hip_pitch_joint": "base_link",
    "right_hip_pitch_joint": "base_link",
}
for (j, b) in BODY_ORDER:
    if j.endswith("_roll_joint") and ("wrist" in j or "shoulder" in j):
        PARENT_OF.setdefault(j, None)  # sequential in tree

# damping / frictionloss / range from X1 serial (by joint name)
X1_JOINT_ATTRS = {
    "lumbar_yaw_joint": dict(damping=1.5, frictionloss=3.0, range=(-1.0, 1.0)),
    "lumbar_roll_joint": dict(damping=1.5, frictionloss=3.0, range=(-0.25, 0.25)),
    "lumbar_pitch_joint": dict(damping=2.0, frictionloss=4.0, range=(-0.15, 0.45)),
    "left_shoulder_pitch_joint": dict(damping=0.2, frictionloss=0.5, range=(-2.5, 1.5)),
    "left_shoulder_roll_joint": dict(damping=0.2, frictionloss=0.5, range=(-2.0, 0.0)),
    "left_shoulder_yaw_joint": dict(damping=0.2, frictionloss=0.5, range=(-1.8, 1.8)),
    "left_elbow_pitch_joint": dict(damping=0.2, frictionloss=0.5, range=(0.0, 2.0)),
    "left_elbow_yaw_joint": dict(damping=0.2, frictionloss=0.5, range=(-1.8, 1.8)),
    "left_wrist_pitch_joint": dict(damping=0.1, frictionloss=0.2, range=(-0.4, 0.4)),
    "left_wrist_roll_joint": dict(damping=0.1, frictionloss=0.2, range=(-1.0, 1.0)),
    "right_shoulder_pitch_joint": dict(damping=0.2, frictionloss=0.5, range=(-2.5, 1.5)),
    "right_shoulder_roll_joint": dict(damping=0.2, frictionloss=0.5, range=(-2.0, 0.0)),
    "right_shoulder_yaw_joint": dict(damping=0.2, frictionloss=0.5, range=(-1.8, 1.8)),
    "right_elbow_pitch_joint": dict(damping=0.2, frictionloss=0.5, range=(0.0, 2.0)),
    "right_elbow_yaw_joint": dict(damping=0.2, frictionloss=0.5, range=(-1.8, 1.8)),
    "right_wrist_pitch_joint": dict(damping=0.1, frictionloss=0.2, range=(-0.4, 0.4)),
    "right_wrist_roll_joint": dict(damping=0.1, frictionloss=0.2, range=(-1.0, 1.0)),
    ".*_hip_pitch_joint": dict(damping=2.0, frictionloss=4.0, range=(-3.14, 3.14)),
    ".*_hip_roll_joint": dict(damping=1.5, frictionloss=3.0, range=(-2.7, 0.3)),
    "left_hip_yaw_joint": dict(damping=1.5, frictionloss=3.0, range=(-1.5, 1.5)),
    "right_hip_yaw_joint": dict(damping=1.5, frictionloss=3.0, range=(-1.5, 1.5)),
    ".*_knee_pitch_joint": dict(damping=2.0, frictionloss=4.0, range=(0.0, 2.7)),
    ".*_ankle_pitch_joint": dict(damping=0.8, frictionloss=1.5, range=(-0.5, 0.8)),
    ".*_ankle_roll_joint": dict(damping=0.8, frictionloss=1.5, range=(-0.5, 0.5)),
}
# X1 serial ctrlranges (motor effort, Nm)
CTRLRANGE = {
    "hip_pitch": (-180, 180), "hip_roll": (-150, 150), "hip_yaw": (-150, 150),
    "knee_pitch": (-180, 180), "ankle_pitch": (-80, 80), "ankle_roll": (-80, 80),
    "lumbar_yaw": (-150, 150), "lumbar_roll": (-150, 150), "lumbar_pitch": (-180, 180),
    "shoulder_pitch": (-20, 20), "shoulder_roll": (-20, 20), "shoulder_yaw": (-20, 20),
    "elbow_pitch": (-20, 20), "elbow_yaw": (-20, 20),
    "wrist_pitch": (-5, 5), "wrist_roll": (-5, 5),
}
# foot corner spheres in ankle_roll frame (X1 values — F1 foot identical)
FOOT_SPHERES = [(0.03, -0.0408, 0.07), (-0.03, -0.0408, 0.07),
                (0.03, -0.0408, -0.07), (-0.03, -0.0408, -0.07)]
FOOT_SIGN = {"left": -1, "right": 1}  # y-flip for right foot (X1 pattern)

F1_STAND = {  # FK-verified IK standing pose (retarget_f1_from_x1lab)
    "left_hip_pitch_joint": -0.23847, "left_hip_roll_joint": 0.02102,
    "left_hip_yaw_joint": 0.00600, "left_knee_pitch_joint": 0.39231,
    "left_ankle_pitch_joint": -0.14583, "left_ankle_roll_joint": 0.02171,
    "right_hip_pitch_joint": -0.23715, "right_hip_roll_joint": -0.02104,
    "right_hip_yaw_joint": -0.00587, "right_knee_pitch_joint": 0.39412,
    "right_ankle_pitch_joint": -0.14649, "right_ankle_roll_joint": -0.02125,
}


def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def R_to_quat_wxyz(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def principal_inertia(I):
    """Symmetric inertia -> (quat, diag). Right-handed eigenvectors."""
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    return R_to_quat_wxyz(V), w


def f(v, nd=8):
    return " ".join(f"{x:.{nd}g}" for x in np.atleast_1d(v))


def main():
    urdf = ET.parse(F1_URDF).getroot()
    links = {l.get("name"): l for l in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}

    def joint_attrs(name):
        # exact match first, then .* wildcard keys
        if name in X1_JOINT_ATTRS:
            return X1_JOINT_ATTRS[name]
        import re as _re
        for k, v in X1_JOINT_ATTRS.items():
            if k.startswith(".*") and _re.search(k, name):
                return v
        raise KeyError(name)

    # ---- mesh assets
    mesh_files = {}
    def register(mesh_name):
        mesh_files.setdefault(mesh_name, f"m{len(mesh_files)}")

    def visual_of(link_name):
        vis = []
        for v in links[link_name].findall("visual"):
            o = v.find("origin")
            g = v.find("geometry/mesh")
            if g is None:
                continue
            fn = os.path.basename(g.get("filename"))
            mname = os.path.splitext(fn)[0]
            register(mname)
            vis.append(dict(mesh=mname,
                            pos=(o.get("xyz", "0 0 0") if o is not None else "0 0 0"),
                            euler=(o.get("rpy", "0 0 0") if o is not None else "0 0 0"),
                            scale=g.get("scale")))
        return vis

    visuals = {b: visual_of(b) for _, b in BODY_ORDER}
    base_vis = visual_of("base_link")
    base_col = links["base_link"].findall("collision")
    base_col_mesh = None
    for c in base_col:
        g = c.find("geometry/mesh")
        if g is not None:
            fn = os.path.basename(g.get("filename"))
            mname = os.path.splitext(fn)[0]
            register(mname)
            o = c.find("origin")
            base_col_mesh = dict(mesh=mname,
                                 pos=(o.get("xyz", "0 0 0") if o is not None else "0 0 0"),
                                 euler=(o.get("rpy", "0 0 0") if o is not None else "0 0 0"),
                                 scale=g.get("scale"))

    # ---- inertials
    def inertial_of(link_name):
        i = links[link_name].find("inertial")
        o = i.find("origin")
        pos = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        mass = float(i.find("mass").get("value"))
        it = i.find("inertia")
        I = np.array([[float(it.get("ixx")), float(it.get("ixy")), float(it.get("ixz"))],
                      [float(it.get("ixy")), float(it.get("iyy")), float(it.get("iyz"))],
                      [float(it.get("ixz")), float(it.get("iyz")), float(it.get("izz"))]])
        R = rpy_to_R(*rpy)
        Ic = R @ I @ R.T
        Ic = (Ic + Ic.T) / 2
        quat, diag = principal_inertia(Ic)
        return pos, mass, quat, diag

    # ---- keyframe base height: FK standing pose, soles on spheres
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fk_lib import UrdfFK
    fk = UrdfFK(F1_URDF)
    Ts = fk.fk_all(dict(F1_STAND), np.eye(4))
    min_sole = 0.0
    for side in ("left", "right"):
        Ta = Ts[f"{side}_ankle_roll_link"]
        for sx, sy, sz in FOOT_SPHERES:
            p = Ta[:3, :3] @ np.array([sx, FOOT_SIGN[side] * sy, sz]) + Ta[:3, 3]
            min_sole = min(min_sole, p[2])
    base_z = -min_sole + 0.002
    print(f"[INFO] standing FK: min sole z = {min_sole*1000:.1f} mm -> keyframe base z = {base_z:.4f}")

    # ---- emit xml
    L = []
    L.append("<?xml version='1.0' encoding='utf-8'?>")
    L.append('<mujoco model="f1 scene">')
    L.append('  <compiler angle="radian" eulerseq="XYZ" meshdir="../meshes" autolimits="true" />')
    L.append('  <option timestep="0.001" />')
    L.append('  <visual>')
    L.append('    <global azimuth="150" elevation="-20" />')
    L.append('    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0 0 0" />')
    L.append('    <scale forcewidth="0.03" contactwidth="0.01" contactheight="0.02" com="1" framelength="0.6" framewidth="0.05" />')
    L.append('    <rgba haze="0.15 0.25 0.35 1" com="0.502 1 0 0.5" contactpoint="1 1 0.6 0.4" contactforce="0.98 0.4 0.4 1" />')
    L.append('  </visual>')
    L.append('  <default>')
    L.append('      <geom contype="0" conaffinity="0" solref="0.005 1" condim="3" friction="1 1" />')
    L.append('      <equality solref="0.005 1" />')
    L.append('      <default class="zhiyuan">')
    L.append('      </default>')
    L.append('      <default class="collision">')
    L.append('          <geom contype="1" group="3" type="capsule" />')
    L.append('          <default class="collision-left">')
    L.append('              <geom contype="2" conaffinity="4" />')
    L.append('          </default>')
    L.append('          <default class="collision-right">')
    L.append('              <geom contype="4" conaffinity="2" />')
    L.append('          </default>')
    L.append('      </default>')
    L.append('    </default>')
    L.append('  <asset>')
    for mname, mid in sorted(mesh_files.items(), key=lambda kv: kv[1]):
        L.append(f'    <mesh name="{mname}" file="{mname}.STL" />')
    L.append('  </asset>')
    L.append('  <worldbody>')
    L.append('    <body name="base_link" pos="0 0 0.8">')
    ip, m, q, dg = inertial_of("base_link")
    L.append(f'      <inertial pos="{f(ip)}" mass="{m:.7g}" quat="{f(q)}" diaginertia="{f(dg)}" />')
    L.append('      <freejoint name="floating_base" />')
    for v in base_vis:
        sc = f' scale="{v["scale"]}"' if v["scale"] else ""
        L.append(f'      <geom type="mesh" rgba="0.8 0.4 0 1" mesh="{v["mesh"]}" pos="{v["pos"]}" euler="{v["euler"]}"{sc} />')
    if base_col_mesh:
        v = base_col_mesh
        sc = f' scale="{v["scale"]}"' if v["scale"] else ""
        L.append(f'      <geom type="mesh" rgba="0.8 0.4 0 1" mesh="{v["mesh"]}" class="collision" pos="{v["pos"]}" euler="{v["euler"]}"{sc} />')
    L.append('      <camera name="track" pos="0 -3 1" quat="0.850651 0.525731 0 0" mode="track" />')
    L.append('      <site name="imu" size="0.01" pos="0 0 0" />')

    indent0 = "      "

    def owning_joint(link_name):
        return next((jn for jn, bn in BODY_ORDER if bn == link_name), None)

    def depth_of(i):
        """tree depth of BODY_ORDER[i] (base_link = 0)."""
        d, jn = 0, BODY_ORDER[i][0]
        link = joints[jn].find("parent").get("link")
        while link != "base_link":
            pj = owning_joint(link)
            d += 1
            link = joints[pj].find("parent").get("link")
        return d

    depths = [depth_of(i) for i in range(len(BODY_ORDER))]
    # sanity: pre-order property — depth[i] <= depth[i-1] + 1
    for i in range(1, len(depths)):
        assert depths[i] <= depths[i - 1] + 1, (BODY_ORDER[i], depths[i], depths[i - 1])

    for i, (jn, bn) in enumerate(BODY_ORDER):
        ind = indent0 + "  " * depths[i]
        j = joints[jn]
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        quat = R_to_quat_wxyz(rpy_to_R(*rpy))
        ip, m, q, dg = inertial_of(bn)
        ax = j.find("axis").get("xyz")
        ja = joint_attrs(jn)
        L.append(f'{ind}<body name="{bn}" pos="{f(xyz)}" quat="{f(quat)}" gravcomp="0">')
        L.append(f'{ind}  <inertial pos="{f(ip)}" mass="{m:.7g}" quat="{f(q)}" diaginertia="{f(dg)}" />')
        L.append(f'{ind}  <joint name="{jn}" type="hinge" pos="0 0 0" axis="{ax}" damping="{ja["damping"]}" range="{ja["range"][0]} {ja["range"][1]}" frictionloss="{ja["frictionloss"]}" />')
        for v in visuals[bn]:
            sc = f' scale="{v["scale"]}"' if v["scale"] else ""
            L.append(f'{ind}  <geom type="mesh" mesh="{v["mesh"]}" pos="{v["pos"]}" euler="{v["euler"]}"{sc} rgba="1 0.4 0.4 1" />')
        if bn.endswith("ankle_roll_link"):
            side = "left" if bn.startswith("left") else "right"
            cls = "collision-left" if side == "left" else "collision-right"
            for sx, sy, sz in FOOT_SPHERES:
                y = FOOT_SIGN[side] * sy
                L.append(f'{ind}  <geom type="sphere" size="0.002" pos="{sx} {y} {sz}" class="{cls}" />')
                L.append(f'{ind}  <geom type="sphere" size="0.02" pos="{sx} {y} {sz}" rgba="255 0 0 0.5" />')
        # close this subtree when the next body is not a descendant
        nxt = depths[i + 1] if i + 1 < len(depths) else 0
        for d in range(depths[i], nxt - 1, -1):
            L.append(f'{indent0 + "  " * d}</body>')

    L.append(f'{indent0}</body>')   # close base_link
    L.append('  </worldbody>')

    # ---- actuators
    L.append('  <actuator>')
    for jn, bn in BODY_ORDER:
        for k, cr in CTRLRANGE.items():
            if k in jn:
                L.append(f'    <motor name="motor_{jn.replace("_joint", "")}" joint="{jn}" ctrlrange="{cr[0]} {cr[1]}" />')
                break
    L.append('  </actuator>')

    # ---- sensors (jointpos + jointvel for all 29)
    L.append('  <sensor>')
    for jn, bn in BODY_ORDER:
        L.append(f'    <jointpos name="jointpos_{jn.replace("_joint", "")}" joint="{jn}" noise="0" />')
        L.append(f'    <jointvel name="jointvel_{jn.replace("_joint", "")}" joint="{jn}" noise="0" />')
    L.append('  </sensor>')

    # ---- keyframe: [x y z qw qx qy qz] + 29 joints in BODY_ORDER
    q_stand = [F1_STAND.get(jn, 0.0) for jn, _ in BODY_ORDER]
    L.append('  <keyframe>')
    L.append(f'    <key name="home_default" qpos="0 0 {base_z:.4f} 1 0 0 0 {" ".join(f"{v:.5f}" for v in q_stand)}" />')
    L.append('  </keyframe>')
    L.append('</mujoco>')

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_XML, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[OK] wrote {OUT_XML} ({len(L)} lines, {len(mesh_files)} meshes, {len(BODY_ORDER)} joints)")

    # ---- flat env include + environment copy
    env_dst = os.path.join(ROOT, "F1_29DOF/mjcf/environment")
    os.makedirs(env_dst, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "X1_29DOF/mjcf/environment/flat.xml"), env_dst + "/flat.xml")
    flat = os.path.join(ROOT, "F1_29DOF/mjcf/xyber_f1_flat.xml")
    with open(flat, "w") as fh:
        fh.write('<mujoco model = "f1 scene">\n\n'
                 '    <include file = "robot/xyber_f1/xyber_f1_serial.xml" />\n'
                 '    <include file = "environment/flat.xml" />\n\n'
                 '</mujoco>\n')
    print(f"[OK] wrote {flat} + environment/flat.xml")

    # ---- gmr-style standalone copy for sim2sim (mujoco_rollout.build_model
    # loads the xml as the MAIN file: meshdir must resolve from its own dir)
    gmr_dir = os.path.join(ROOT, "gmr_f1_assets")
    os.makedirs(os.path.join(gmr_dir, "meshes"), exist_ok=True)
    txt = "\n".join(L) + "\n"
    txt = txt.replace('meshdir="../meshes"', 'meshdir="meshes"')
    with open(os.path.join(gmr_dir, "f1.xml"), "w") as fh:
        fh.write(txt)
    for mname in mesh_files:
        src = os.path.join(ROOT, "F1_29DOF/meshes", f"{mname}.STL")
        if not os.path.exists(src):
            src = os.path.join(ROOT, "F1_29DOF/meshes", f"{mname}.stl")
        shutil.copy(src, os.path.join(gmr_dir, "meshes", os.path.basename(src)))
    print(f"[OK] wrote gmr_f1_assets/f1.xml + {len(mesh_files)} meshes")

    # ---- self-check: xml parses + joint count + mesh files exist
    root = ET.parse(OUT_XML).getroot()
    bodies = root.findall(".//body")
    hinges = [j for j in root.findall(".//joint") if j.get("type") == "hinge"]
    motors = root.findall(".//motor")
    assert len(bodies) == 30, len(bodies)
    assert len(hinges) == 29, len(hinges)
    assert len(motors) == 29, len(motors)
    mesh_dir = os.path.join(ROOT, "F1_29DOF/meshes")
    missing = [m.get("file") for m in root.findall(".//asset/mesh")
               if not os.path.exists(os.path.join(mesh_dir, m.get("file")))]
    assert not missing, missing
    print(f"[OK] self-check: 30 bodies, 29 hinges, 29 motors, meshes present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
