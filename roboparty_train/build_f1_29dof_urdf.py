#!/usr/bin/env python3
"""Build F1_29DOF full 29-DOF URDF.

F1 vs X1 (user contract): ONLY the lower-limb 12 DOF (and its pelvis) differ.
Upper body (3 lumbar + 14 arm joints, head/sensor links) is X1's, unchanged.

Source of truth:
  - X1_29DOF/urdf/f1.urdf            : full 29-DOF tree (upper body taken from here)
  - F1_29DOF/urdf/hip_perfect_mirrored/x1_12dof_hip_perfect_mirrored.urdf
                                      : vendor F1 lower body (base_link + legs, upper fixed)

Output: F1_29DOF/urdf/f1_29dof.urdf  (mesh prefix ../meshes -> F1_29DOF/meshes)

Verified equivalences before writing this script (xml diff):
  - lumbar_yaw_joint origin identical in both files -> upper body bolts on unchanged
  - joint names of the 29 actuated DOF identical between X1 and F1
  - F1 lower body = redesigned hip (pitch/roll orthogonal axes instead of X1's
    45-deg tilted hip pitch) + right leg made perfect mirror of left
  - cosmetic fixed chains (waist_motor on lumbar_pitch_link, arm wrist motors on
    elbow links) hang off UPPER body -> keep X1's; toe cosmetic chains hang off
    knee_pitch_link -> take F1's (renamed *_link_joint, p4_mirrored meshes)
"""
import copy
import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X1_URDF = os.path.join(ROOT, "X1_29DOF/urdf/f1.urdf")
F1_URDF = os.path.join(ROOT, "F1_29DOF/urdf/hip_perfect_mirrored/x1_12dof_hip_perfect_mirrored.urdf")
OUT_URDF = os.path.join(ROOT, "F1_29DOF/urdf/f1_29dof.urdf")

# ---- which parts come from F1 (lower body) ---------------------------------
LEG_JOINT_RX = re.compile(
    r"^((left|right)_(hip_(pitch|roll|yaw)|knee_pitch|ankle_pitch|ankle_roll)_joint"
    r"|leg_[lr]_toe_[ab]_(link|ball|loop)_joint)$"
)
LEG_LINK_RX = re.compile(
    r"^((left|right)_(hip_(pitch|roll|yaw)|knee_pitch|ankle_pitch|ankle_roll)_link"
    r"|leg_[lr]_toe_[ab]_(link|ball_link|loop_link))$"
)
# X1-side names for the same chains (toe cosmetic names differ)
X1_LEG_JOINT_RX = re.compile(
    r"^((left|right)_(hip_(pitch|roll|yaw)|knee_pitch|ankle_pitch|ankle_roll)_joint"
    r"|leg_[lr]_toe_[ab]_joint)$"
)
X1_LEG_LINK_RX = re.compile(
    r"^((left|right)_(hip_(pitch|roll|yaw)|knee_pitch|ankle_pitch|ankle_roll)_link"
    r"|leg_[lr]_toe_[ab]_(link|ball|loop))$"
)

REV29 = [
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def fix_mesh_paths(el):
    """F1 file sits two levels deep (urdf/hip_perfect_mirrored/) -> merged file
    sits one level deep (urdf/). Rewrite ../../meshes/ -> ../meshes/."""
    for m in el.iter("mesh"):
        fn = m.get("filename", "")
        if fn.startswith("../../meshes/"):
            m.set("filename", "../meshes/" + fn[len("../../meshes/"):])
    return el


def main():
    x1 = ET.parse(X1_URDF).getroot()
    f1 = ET.parse(F1_URDF).getroot()

    # 1) drop X1 lower body (joints + links)
    dropped_j = [j for j in x1.findall("joint") if X1_LEG_JOINT_RX.match(j.get("name"))]
    dropped_l = [l for l in x1.findall("link") if X1_LEG_LINK_RX.match(l.get("name"))]
    for j in dropped_j:
        x1.remove(j)
    for l in dropped_l:
        x1.remove(l)

    # 2) graft F1 lower body joints + links (mesh paths fixed)
    n_j = n_l = 0
    for j in f1.findall("joint"):
        if LEG_JOINT_RX.match(j.get("name")):
            x1.append(fix_mesh_paths(copy.deepcopy(j)))
            n_j += 1
    for l in f1.findall("link"):
        if LEG_LINK_RX.match(l.get("name")):
            x1.append(fix_mesh_paths(copy.deepcopy(l)))
            n_l += 1

    # 3) base_link = F1 pelvis (inertial/visual/collision wholesale)
    f1_base = f1.find("link[@name='base_link']")
    x1_base = x1.find("link[@name='base_link']")
    new_base = fix_mesh_paths(copy.deepcopy(f1_base))
    x1.remove(x1_base)
    # insert right after <robot> opening for readability
    x1.insert(0, new_base)

    # 4) global material defs: copy any F1 materials not already in X1 tree
    x1_mats = {m.get("name") for m in x1.findall("material")}
    for m in f1.findall("material"):
        if m.get("name") not in x1_mats:
            x1.append(copy.deepcopy(m))

    # 5) robot name
    x1.set("name", "f1_29dof")

    ET.indent(x1, space="  ")
    ET.ElementTree(x1).write(OUT_URDF, encoding="utf-8", xml_declaration=True)

    # ---------------- validation ----------------
    tree = ET.parse(OUT_URDF)
    r = tree.getroot()
    links = [l.get("name") for l in r.findall("link")]
    joints = {j.get("name"): j for j in r.findall("joint")}
    rev = [n for n, j in joints.items() if j.get("type") == "revolute"]

    errs = []
    # exactly 29 revolute, matching the X1 actuated set
    if sorted(rev) != sorted(REV29):
        errs.append(f"revolute set mismatch: extra={set(rev)-set(REV29)} missing={set(REV29)-set(rev)}")
    # single-root tree connectivity (BFS over parent->children edges)
    child_of = {j.find("child").get("link") for j in joints.values()}
    children_of = {}
    for j in joints.values():
        children_of.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))
    roots = [l for l in links if l not in child_of]
    if roots != ["base_link"]:
        errs.append(f"roots={roots}")
    seen = set()
    stack = ["base_link"]
    while stack:
        cur = stack.pop()
        seen.add(cur)
        stack.extend(children_of.get(cur, []))
    unreachable = set(links) - seen
    if unreachable:
        errs.append(f"unreachable links: {sorted(unreachable)}")
    # parent link of every joint must exist
    for jn, j in joints.items():
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        if p not in links or c not in links:
            errs.append(f"{jn}: dangling parent/child {p}->{c}")
    # meshes exist
    missing = set()
    for m in r.iter("mesh"):
        fn = m.get("filename")
        if not fn.startswith("../meshes/"):
            errs.append(f"bad mesh path {fn}")
            continue
        if not os.path.exists(os.path.join(os.path.dirname(OUT_URDF), fn)):
            missing.add(fn)
    if missing:
        errs.append(f"missing meshes: {sorted(missing)}")

    total_mass = 0.0
    for l in r.findall("link"):
        m = l.find("inertial/mass")
        if m is not None:
            total_mass += float(m.get("value"))

    print(f"links={len(links)} joints={len(joints)} revolute={len(rev)}")
    print(f"grafted from F1: {n_j} joints, {n_l} links + base_link")
    print(f"total mass = {total_mass:.3f} kg")
    print(f"VALIDATION: {'FAIL ' + str(errs) if errs else 'PASS'}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
