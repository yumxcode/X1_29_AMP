#!/usr/bin/env python3
"""Reference-motion gait analysis: do the retargeted AMASS clips (x1_lab)
that v16-v26 AMP-trained on walk heel-to-toe or ball-foot?

FK every frame of each clip in the SAME x1.xml model the sim2sim rollout
uses, take the 4 sole spheres per foot, and reuse gait_metrics' Schmitt
contact segmentation + event metrics. Quat convention (wxyz as-is vs
xyzw->wxyz) is decided per clip by matching FK key bodies (feet/knees/
elbows) against the clip's stored key_body_pos.

Usage: ./.venv_test/bin/python acceptance/ref_gait_analysis.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim2sim"))

from sim2sim.mujoco_rollout import build_model, find_sole_geoms, parse_yaml_list, DEFAULT_Q  # noqa: E402
from sim2sim.gait_metrics import schmitt_contact, stance_windows, SPEC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def analyze_clip(name, blob, model, data, qadr, lab2mj, soles, feet_names, key_ids):
    import mujoco
    clip = pickle.loads(blob, encoding="latin1")
    fps = float(clip["fps"])
    rp = np.asarray(clip["root_pos"])
    rr = np.asarray(clip["root_rot"], dtype=np.float64)
    q = np.asarray(clip["dof_pos"])
    kb = np.asarray(clip["key_body_pos"])          # (T,6,3)
    T = len(rp)

    def fk(quats):
        sole_z, key_err = [], []
        for i in range(T):
            data.qpos[:] = 0
            data.qpos[0:3] = rp[i]
            data.qpos[3:7] = quats[i] / np.linalg.norm(quats[i])
            data.qpos[qadr] = 0.0
            data.qpos[qadr][lab2mj] = q[i]          # lab col -> mj hinge
            mujoco.mj_forward(model, data)
            pts = np.array([[data.geom_xpos[g].copy() for g in row]
                            for row in soles])
            sole_z.append(pts[:, :, 2])
            key_err.append(np.abs(
                np.array([data.xpos[k].copy() for k in key_ids]) - kb[i]).mean())
        return np.array(sole_z), float(np.mean(key_err))

    rr_swap = rr[:, [3, 0, 1, 2]]
    z1, e1 = fk(rr)          # wxyz as-is
    z2, e2 = fk(rr_swap)     # xyzw -> wxyz
    conv = "wxyz" if e1 <= e2 else "xyzw->wxyz"
    z = z1 if e1 <= e2 else z2

    dt = 1.0 / fps
    out = {"clip": name, "fps": fps, "T": T, "conv": conv,
           "key_err_m": min(e1, e2)}
    evs = []
    for f, fname in enumerate(feet_names):
        zz = z[:, f, :] - 0.002                      # sphere bottoms
        contact = schmitt_contact(zz.min(1))
        front, back = zz[:, :2].mean(1), zz[:, 2:].mean(1)
        pitch = np.degrees(np.arctan2(front - back,
                                      np.linalg.norm([1.0])))  # placeholder
        # proper pitch: need horizontal distance; use fixed sole length 0.14 m
        # (spheres at local z +/-0.07 on a rigid foot)
        pitch = np.degrees(np.arctan2(front - back, 0.14))
        for td, lo in stance_windows(contact, 2):
            pre = td - 1
            if pre < 1:
                continue
            span = lo - td + 1
            m0 = td + int(SPEC["settle_skip"] * span)
            m1 = lo - int(SPEC["pushoff_skip"] * span)
            hu = (np.sum(front[m0:m1 + 1] - back[m0:m1 + 1] < -0.04) if m1 > m0 else 0)
            evs.append(dict(foot=fname, pitch_td=float(pitch[pre]),
                            pitch_min=float(pitch[td:lo + 1].min()),
                            pitch_max=float(pitch[td:lo + 1].max()),
                            heelup=float(hu / (m1 - m0 + 1)) if m1 > m0 else np.nan,
                            stance_s=float(span * dt)))
    out["events"] = evs
    if evs:
        pit = np.array([e["pitch_td"] for e in evs])
        pmn = np.array([e["pitch_min"] for e in evs])
        hu = np.array([e["heelup"] for e in evs], dtype=np.float64)
        out.update(n=len(evs), pitch_td_med=float(np.median(pit)),
                   pitch_td_min=float(pit.min()), pitch_td_max=float(pit.max()),
                   pitch_min_med=float(np.median(pmn)),
                   heelup_med=float(np.nanmedian(hu)))
    return out


def main():
    import mujoco
    model, _ = build_model(ROOT / "gmr_x1_assets" / "x1.xml")
    data = mujoco.MjData(model)
    soles, _ = find_sole_geoms(model)
    feet_names = sorted(soles)
    lab_dof = parse_yaml_list(
        ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml",
        "lab_dof_names")
    mj_names = [model.joint(i).name for i in range(model.njnt)]
    hinge = [n for n in mj_names if n in DEFAULT_Q]
    qadr = np.array([model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in hinge])
    lab2mj = np.array([hinge.index(n) for n in lab_dof])
    key_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in [
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_knee_pitch_link", "right_knee_pitch_link",
        "left_elbow_yaw_link", "right_elbow_yaw_link"]]

    pkg = pickle.load(open(ROOT / "acceptance/v26_artifacts/model_retarget_data.pt", "rb"),
                      encoding="latin1")
    print(f"{'clip':24s} {'conv':10s} {'keyerr':7s} {'n':>3s} "
          f"{'TD_med':>7s} {'TD_rng':>13s} {'pMin_med':>8s} {'heelup':>7s}")
    for name in ["0005_normal_walk1", "0007_normal_walk3", "0008_normal_walk4",
                 "0000_treadmill_norm", "0002_treadmill_slow",
                 "0009_normal_jog1", "0026_circle_walk", "127_06"]:
        key = f"x1_lab/{name}.pkl"
        if key not in pkg:
            print(f"{name:24s} MISSING")
            continue
        r = analyze_clip(name, pkg[key], model, data, qadr, lab2mj,
                         np.array([soles[f] for f in feet_names]), feet_names, key_ids)
        if "n" in r:
            print(f"{r['clip']:24s} {r['conv']:10s} {r['key_err_m']*1000:5.1f}mm {r['n']:3d} "
                  f"{r['pitch_td_med']:7.2f} [{r['pitch_td_min']:5.1f},{r['pitch_td_max']:5.1f}] "
                  f"{r['pitch_min_med']:8.2f} {r['heelup_med']:7.3f}")
        else:
            print(f"{r['clip']:24s} {r['conv']:10s} {r['key_err_m']*1000:5.1f}mm   no events")


if __name__ == "__main__":
    main()
