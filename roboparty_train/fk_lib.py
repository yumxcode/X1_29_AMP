#!/usr/bin/env python3
"""Shared numpy URDF FK library for X1 / F1_29dof (Mac-local, mujoco-free).

Convention (verified against x1_lab pkls): the pkl root_pos/root_rot (wxyz)
are the WORLD pose of the URDF base_link frame — identical to the MuJoCo
gmr_x1_assets/x1.xml floating base. key_body_pos = world positions of the
KEY10 bodies.
"""
import numpy as np
import xml.etree.ElementTree as ET


def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def quat_wxyz_to_R(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R):
    """Shepperd's method, w >= 0 canonical."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def axis_angle(R):
    """rotation matrix -> rotation vector (axis * angle)."""
    q = R_to_quat_wxyz(R)
    w = np.clip(q[0], -1.0, 1.0)
    ang = 2 * np.arccos(w)
    if ang < 1e-10:
        return np.zeros(3)
    s = np.sqrt(max(1e-20, 1 - w * w))
    return q[1:] / s * ang


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    a = w / th
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


class UrdfFK:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()])
            nrm = np.linalg.norm(axis)
            axis = axis / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])
            lim = j.find("limit")
            self.joints[j.get("name")] = dict(
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                xyz=xyz, R=rpy_to_R(*rpy), axis=axis / np.linalg.norm(axis),
                type=j.get("type"),
                lower=float(lim.get("lower", "-3.14")) if lim is not None and lim.get("lower") else -np.pi,
                upper=float(lim.get("upper", "3.14")) if lim is not None and lim.get("upper") else np.pi)
        # children map
        self.children = {}
        for n, jd in self.joints.items():
            self.children.setdefault(jd["parent"], []).append(n)

    def chain(self, leaf_link):
        """joint-name chain base -> link that owns leaf_link as child."""
        # find joint whose child is leaf_link, walk parents
        chain = []
        cur = leaf_link
        while True:
            jn = next((n for n, jd in self.joints.items() if jd["child"] == cur), None)
            if jn is None:
                break
            chain.append(jn)
            cur = self.joints[jn]["parent"]
        return list(reversed(chain))

    @staticmethod
    def joint_T(jd, q):
        T = np.eye(4)
        T[:3, :3] = jd["R"]
        T[:3, 3] = jd["xyz"]
        if q:
            a = jd["axis"]
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            T[:3, :3] = T[:3, :3] @ (np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K))
        return T

    def fk_chain(self, chain, qs, T_base=None):
        T = np.eye(4) if T_base is None else T_base.copy()
        for jn, q in zip(chain, qs):
            T = T @ self.joint_T(self.joints[jn], q)
        return T

    def fk_all(self, dof_map, T_base=None):
        """dof_map: {joint_name: angle}. Returns {link: 4x4 world T}."""
        out = {}
        T0 = np.eye(4) if T_base is None else T_base.copy()
        stack = [("base_link", T0)]
        while stack:
            link, T = stack.pop()
            out[link] = T
            for jn in self.children.get(link, []):
                jd = self.joints[jn]
                q = dof_map.get(jn, 0.0) if jd["type"] == "revolute" else 0.0
                stack.append((jd["child"], T @ self.joint_T(jd, q)))
        return out


LAB_DOF_NAMES = None  # filled by load_lab_order()


def load_lab_order(x1_yaml):
    names, in_block = [], False
    for line in open(x1_yaml):
        if line.startswith("lab_dof_names:"):
            in_block = True
            continue
        if in_block:
            s = line.strip()
            if s.startswith("- "):
                names.append(s[2:].strip())
            elif s and not s.startswith("#"):
                break
    return names


KEY10 = ["left_knee_pitch_link", "right_knee_pitch_link",
         "left_ankle_roll_link", "right_ankle_roll_link",
         "left_elbow_yaw_link", "right_elbow_yaw_link",
         "left_shoulder_pitch_link", "right_shoulder_pitch_link",
         "left_wrist_pitch_link", "right_wrist_pitch_link"]
