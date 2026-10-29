#!/usr/bin/env python3
"""C1 chest-lean gate (v94, RECORD_BOOK 13): the 腰后仰 user-defect gate.

Measures the MuJoCo-FK chest tilt (base->shoulder-mid, world xz; + = chest
forward) on rollout npz files and applies the pre-registered bands:

  walk10 : chest_tilt_p50 in [-2.0, +20.0] deg   (binding)
  walk05 : chest_tilt_p50 >= -3.0 deg            (binding)
  back05 : chest_tilt_p50 >= -3.0 deg            (binding)

Calibration (test_v94_chestlean.py, 2026-10-29): demo stream medians
0..+6 deg (07_01 +2.3 / 08_03 +0.1 / 38_02 +5.9 / 35_02 +0.8 / 82_12 +0.2);
the v91b-q4499 delivery sits at -9.9 (walk10) / -8.5 (walk05) — ~10 deg
behind every demo. The +27-lumbar lineage (l3999) chest lands at +13.3,
inside the band (upper edge generous: the guard's own hinge is [0, +20.05]).

Usage: chest_lean_gates.py <policy_tag> <walk10.npz> <walk05.npz> <back05.npz>
Writes acceptance/v83_eval/<tag>_chest.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim2sim"))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q  # noqa: E402

BANDS = {"walk10": (-2.0, 20.0), "walk05": (-3.0, 99.0), "back05": (-3.0, 99.0)}


def chest_tilt_series(npz_path):
    model, _ = build_model(ROOT / "gmr_x1_assets/x1.xml")
    data = mujoco.MjData(model)
    BID = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
    b_base, b_shL, b_shR = BID("base_link"), BID("left_shoulder_pitch_link"), BID("right_shoulder_pitch_link")
    qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in DEFAULT_Q}
    d = np.load(npz_path, allow_pickle=True)
    names = json.loads(str(d["meta"].item()))["hinge_names"]
    out = np.zeros(len(d["t"]))
    for i in range(len(d["t"])):
        data.qpos[:] = 0
        data.qpos[0:3] = d["base_pos"][i]
        data.qpos[3:7] = d["base_quat"][i]
        for j, n in enumerate(names):
            data.qpos[qadr[n]] = d["q"][i, j]
        mujoco.mj_forward(model, data)
        v = (data.xpos[b_shL] + data.xpos[b_shR]) / 2 - data.xpos[b_base]
        out[i] = np.degrees(np.arctan2(v[0], v[2]))
    return out


def main():
    tag = sys.argv[1]
    files = {"walk10": sys.argv[2], "walk05": sys.argv[3], "back05": sys.argv[4]}
    res, fails = {}, []
    for scen, path in files.items():
        s = chest_tilt_series(path)[25:]  # skip settle
        p50 = float(np.median(s))
        lo, hi = BANDS[scen]
        ok = lo <= p50 <= hi
        res[scen] = {"p50": p50, "p10": float(np.percentile(s, 10)),
                     "p90": float(np.percentile(s, 90)),
                     "band": [lo, hi], "pass": ok}
        if not ok:
            fails.append(scen)
    out = {"tag": tag, "checks": res, "verdict": "PASS" if not fails else "FAIL",
           "fails": fails}
    dest = ROOT / "acceptance/v83_eval" / f"{tag}_chest.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {dest}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
