#!/usr/bin/env python3
"""v68 sim2real parameter transcription check (x1_armature_pd.md -> code).

Runs LOCALLY (no isaaclab needed — AST-parses x1.py, imports mujoco_rollout's
PD/ARMATURE tables via regex) and asserts, per joint:

  1. x1.py actuators (kp/kd/armature) == the md table  (§1)
  2. sim2sim mujoco_rollout.py PD kp/kd + ARMATURE == the same table (§3)
  3. recomputed zeta = kd/(2*sqrt(kp*I_tot)) and fn = sqrt(kp/I_tot)/2pi
     match the md design table (header block) within tolerance
  4. Isaac 200 Hz explicit-PD stability: kd*dt/I_tot < 0.8 (md principle 4)
  5. deployment yaml lists in POLICY order (§4) generated from the table
     match the md-provided arrays exactly; writes
     sim2sim/deploy_rl_walk_amp_pd_v68.yaml when --write-yaml is passed.

Exit 0 = every transcription exact. This is the pre-flight gate before the
platform fine-tune launch.
"""
import ast
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# ----------------------------------------------------------------------------
# Ground truth: x1_armature_pd.md (2026-10-14, 方案 B). joint -> (I_tot, kp, kd)
# ζ/fn columns are the DESIGN targets recomputed below.
# ----------------------------------------------------------------------------
MD_TABLE = {
    #                I_tot    kp    kd    zeta  fn[Hz]
    "hip_pitch":     (0.49,  65.0, 8.0, 0.71, 1.8),
    "hip_roll":      (0.74,  60.0, 8.0, 0.60, 1.4),
    "hip_yaw":       (0.042, 100.0, 3.3, 0.81, 7.8),
    "knee_pitch":    (0.39,  85.0, 8.0, 0.70, 2.4),
    "ankle_pitch":   (0.15,  50.0, 4.0, 0.73, 2.9),
    "ankle_roll":    (0.036, 50.0, 2.5, 0.93, 5.9),
    "lumbar_yaw":    (0.73,  90.0, 8.0, 0.49, 1.8),
    "lumbar_roll":   (1.35,  70.0, 8.0, 0.41, 1.1),
    "lumbar_pitch":  (0.76, 100.0, 8.0, 0.46, 1.8),
    "shoulder_pitch": (0.19, 40.0, 4.0, 0.72, 2.3),
    "shoulder_roll": (0.20, 40.0, 4.0, 0.71, 2.3),
    "shoulder_yaw":  (0.021, 40.0, 2.0, 1.08, 6.9),
    "elbow_pitch":   (0.064, 30.0, 1.9, 0.69, 3.5),
    "elbow_yaw":     (0.011, 20.0, 1.0, 1.06, 6.8),
}
WRIST = {"kp": 15.0, "kd": 1.0, "I_tot": 0.011, "zeta": 1.24, "fn": 5.9}

# armature (md §1 actuators block), joint-family -> value
MD_ARMATURE = {
    "lumbar_yaw": 0.022, "lumbar_roll": 0.021, "lumbar_pitch": 0.012,
    "hip_pitch": 0.208, "hip_roll": 0.25, "hip_yaw": 0.0148, "knee_pitch": 0.273,
    "ankle_pitch": 0.15, "ankle_roll": 0.035,
    "shoulder_pitch": 0.02, "shoulder_roll": 0.013, "shoulder_yaw": 0.01,
    "elbow_pitch": 0.01, "elbow_yaw": 0.01,
}
WRIST_ARMATURE = 0.01

# deployment policy order (md §4 顺序对照)
POLICY_ORDER = [
    "left_hip_pitch_joint", "lumbar_yaw_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "lumbar_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "lumbar_pitch_joint", "right_hip_yaw_joint",
    "left_knee_pitch_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "right_knee_pitch_joint", "left_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "right_ankle_roll_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint", "left_elbow_yaw_joint",
    "right_elbow_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]
MD_YAML = {
    "stiffness": [65.0, 90.0, 65.0, 60.0, 70.0, 60.0,
                  100.0, 100.0, 100.0, 85.0, 40.0, 40.0,
                  85.0, 50.0, 40.0, 40.0, 50.0, 50.0,
                  40.0, 40.0, 50.0, 30.0, 30.0, 20.0,
                  20.0, 15.0, 15.0, 15.0, 15.0],
    "damping":   [8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                  3.3, 8.0, 3.3, 8.0, 4.0, 4.0,
                  8.0, 4.0, 4.0, 4.0, 4.0, 2.5,
                  2.0, 2.0, 2.5, 1.9, 1.9, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0],
}

FAMILY_RE = re.compile(r"^(?:left|right)_(.+)_joint$")


def family(joint):
    m = FAMILY_RE.match(joint)
    return m.group(1) if m else joint.replace("_joint", "")


def resolve(pattern):
    """'.*_hip_pitch_joint' -> both sides; plain name -> itself."""
    if pattern.startswith(".*_"):
        return [f"left_{pattern[3:]}", f"right_{pattern[3:]}"]
    return [pattern]


def parse_x1_actuators(path):
    """AST-walk X1_CFG.actuators; return joint -> {kp, kd, armature}."""
    tree = ast.parse(path.read_text())
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "DelayedPDActuatorCfg"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        names = []
        for el in ast.literal_eval(kw["joint_names_expr"]):
            names += resolve(el)
        def grab(key):
            if key not in kw:
                return None
            try:
                val = ast.literal_eval(kw[key])
            except Exception:
                return None
            if not isinstance(val, dict):
                return {n: val for n in names}
            out = {}
            for pat, v in val.items():     # keys are joint-name PATTERNS
                for n in resolve(pat):
                    out[n] = v
            return out
        st, da, ar = grab("stiffness"), grab("damping"), grab("armature")
        for n in names:
            out[n] = {
                "kp": st[n] if st else None,
                "kd": da[n] if da else None,
                "armature": ar[n] if ar else None,
            }
    return out


def parse_sim2sim(path):
    """Extract PD {joint: (kp,kd,effort)} and ARMATURE {joint: v}."""
    text = path.read_text()

    def extract_dict(name):
        m = re.search(rf"^{name}\s*=\s*\{{", text, re.M)
        if not m:
            return {}
        depth, i = 1, m.end()
        while depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        inner = text[m.end():i - 1]
        d = {}
        for mm in re.finditer(r'"([^"]+)"\s*:\s*(\(([^)]*)\)|[0-9.]+)', inner):
            key, val = mm.group(1), mm.group(2)
            d[key] = (tuple(float(x) for x in val[1:-1].split(","))
                      if val.startswith("(") else float(val))
        return d

    return extract_dict("PD"), extract_dict("ARMATURE")


def main():
    failures = []

    def check(cond, msg):
        print(("  PASS " if cond else "  FAIL ") + msg)
        if not cond:
            failures.append(msg)

    # -- 1/2: transcription ------------------------------------------------
    x1 = parse_x1_actuators(REPO / "roboparty_train" / "robolab" / "robolab"
                            / "assets" / "robots" / "x1.py")
    pd_t, arm_t = parse_sim2sim(REPO / "sim2sim" / "mujoco_rollout.py")

    print(f"[1] x1.py actuators parsed: {len(x1)} joints")
    check(len(x1) == 29, f"x1.py covers 29 joints (got {len(x1)})")

    print("[2] per-joint kp/kd/armature vs md table (x1.py & sim2sim)")
    all_joints = sorted(set(list(x1) + list(pd_t)))
    for j in all_joints:
        fam = family(j)
        if fam.startswith("wrist"):
            e_kp, e_kd, e_ar = WRIST["kp"], WRIST["kd"], WRIST_ARMATURE
            e_I, e_zeta, e_fn = WRIST["I_tot"], WRIST["zeta"], WRIST["fn"]
        else:
            exp = MD_TABLE[fam]
            e_kp, e_kd, e_ar = exp[1], exp[2], MD_ARMATURE[fam]
            e_I, e_zeta, e_fn = exp[0], exp[3], exp[4]
        got = x1.get(j, {})
        ok = (got.get("kp") == e_kp and got.get("kd") == e_kd
              and abs(got.get("armature", -1) - e_ar) < 1e-9)
        sp = pd_t.get(j, (None, None, None))
        ok &= (sp[0] == e_kp and sp[1] == e_kd
               and abs(arm_t.get(j, -1) - e_ar) < 1e-9)
        check(ok, f"{j}: kp={e_kp} kd={e_kd} arm={e_ar} "
                  f"(x1={got.get('kp')}/{got.get('kd')}/{got.get('armature')}, "
                  f"s2s={sp[0]}/{sp[1]}/{arm_t.get(j)})")
        # -- 3: dynamics recomputation -------------------------------------
        zeta = e_kd / (2.0 * math.sqrt(e_kp * e_I))
        fn = math.sqrt(e_kp / e_I) / (2 * math.pi)
        check(abs(zeta - e_zeta) <= 0.015 + 0.02 * abs(e_zeta),
              f"{fam}: zeta {zeta:.3f} vs md {e_zeta}")
        check(abs(fn - e_fn) <= 0.15, f"{fam}: fn {fn:.2f} Hz vs md {e_fn}")
        # -- 4: 200 Hz explicit-PD stability --------------------------------
        check(e_kd * 0.005 / e_I < 0.8,
              f"{fam}: kd*dt/I_tot = {e_kd * 0.005 / e_I:.3f} < 0.8")

    # -- 5: deployment yaml --------------------------------------------------
    print("[5] deployment yaml arrays (policy order) vs md §4")
    kp_list = [x1[j]["kp"] for j in POLICY_ORDER]
    kd_list = [x1[j]["kd"] for j in POLICY_ORDER]
    check(kp_list == MD_YAML["stiffness"], "stiffness array matches md §4")
    check(kd_list == MD_YAML["damping"], "damping array matches md §4")
    if kp_list != MD_YAML["stiffness"]:
        for i, (a, b) in enumerate(zip(kp_list, MD_YAML["stiffness"])):
            if a != b:
                print(f"      idx {i} {POLICY_ORDER[i]}: {a} != {b}")
    if kd_list != MD_YAML["damping"]:
        for i, (a, b) in enumerate(zip(kd_list, MD_YAML["damping"])):
            if a != b:
                print(f"      idx {i} {POLICY_ORDER[i]}: {a} != {b}")

    if "--write-yaml" in sys.argv:
        out = REPO / "sim2sim" / "deploy_rl_walk_amp_pd_v68.yaml"
        fmt = lambda v: (f"{v:g}")
        out.write_text(
            "# v68 sim2real PD for motion_control .../cfg/rl_x1.yaml -> rl_walk_amp\n"
            "# joint order = policy order (Isaac Lab interleaved); ankles are host-side\n"
            "# software PD and read the same table. Generated by check_v68_params.py\n"
            "# from x1_armature_pd.md — verified against md §4 arrays.\n"
            "stiffness: [" + ", ".join(fmt(v) for v in kp_list) + "]\n"
            "damping:   [" + ", ".join(fmt(v) for v in kd_list) + "]\n"
            "# 顺序对照:\n#   " + ", ".join(POLICY_ORDER) + "\n")
        print(f"[INFO] wrote {out}")

    print()
    if failures:
        print(f"VERDICT: FAIL ({len(failures)} mismatches)")
        sys.exit(1)
    print("VERDICT: PASS — md table transcribed exactly into x1.py + sim2sim; "
          "zeta/fn/stability reverified; deployment arrays consistent.")


if __name__ == "__main__":
    main()
