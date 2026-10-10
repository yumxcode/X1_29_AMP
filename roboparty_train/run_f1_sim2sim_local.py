#!/usr/bin/env python3
"""F1 sim2sim local driver (Mac, mujoco 3.1.1 — no MjSpec).

Runs the UNMODIFIED mujoco_rollout.main() loop with a build_model shim:
the gmr_f1_assets/f1.xml is loaded via string-injected floor + the exact
v26/v58 model alignment (frictionloss 0, hinge damping 1.0, armature 0.01)
that build_model applies on the pod (verified: PD-hold stands in this shim).

Usage:
  .venv-mac/bin/python roboparty_train/run_f1_sim2sim_local.py \
      --ckpt acceptance/f1_v1/model_3999.pt --tag v1_3999
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim2sim"))

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import mujoco_rollout as MR  # noqa: E402

FLOOR = ('  <geom name="floor" type="plane" size="20 20 0.1" '
         'friction="1 0.005 0.0001" contype="1" conaffinity="1" '
         'rgba="0.35 0.45 0.5 1"/>')


def local_build_model(xml_path, actuator_mode="explicit"):
    src = Path(xml_path).read_text()
    assert "</worldbody>" in src, "unexpected vendor xml"
    src = src.replace("</worldbody>", FLOOR + "\n</worldbody>")
    fd, tmp = tempfile.mkstemp(suffix=".xml", dir=str(Path(xml_path).parent))
    os.close(fd)
    Path(tmp).write_text(src)
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
    m.dof_frictionloss[:] = 0.0
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            m.dof_damping[m.jnt_dofadr[j]] = 1.0
            m.dof_armature[m.jnt_dofadr[j]] = 0.01
    n_feet = 0
    for g in range(m.ngeom):
        bid = m.geom_bodyid[g]
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if nm.endswith("ankle_roll_link") and m.geom_contype[g]:
            n_feet += 1
    return m, n_feet


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="v1_3999")
    ap.add_argument("--cmds", default="1.0,0.5,0.0", help="comma x-speeds")
    ap.add_argument("--duration", type=float, default=12.0)
    args, _ = ap.parse_known_args()

    MR.build_model = local_build_model
    out_dir = ROOT / "acceptance" / "f1_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for vx in [float(v) for v in args.cmds.split(",")]:
        tagx = f"{args.tag}_walk{int(round(vx*10)):02d}"
        npz = str(out_dir / f"{tagx}.npz")
        sys.argv = [
            "mujoco_rollout.py",
            "--ckpt", args.ckpt,
            "--robot", "f1",
            "--cmd", str(vx), "0.0", "0.0",
            "--duration", str(args.duration),
            "--repo-root", str(ROOT),
            "--log", npz,
        ]
        print(f"\n===== sim2sim {tagx}: vx={vx} =====")
        try:
            MR.main()
            ok = os.path.exists(npz)
        except SystemExit as e:
            ok = os.path.exists(npz) and (e.code in (0, None))
        summary[tagx] = dict(npz=npz, ok=ok)
        print(f"[{'OK' if ok else 'FAIL'}] {tagx} -> {npz}")
    print("\n==== gait_metrics ====")
    logs = [s["npz"] for s in summary.values() if s["ok"]]
    if logs:
        sys.argv = ["gait_metrics.py"] + logs + ["--json", str(out_dir / f"{args.tag}_gait_report.json")]
        import gait_metrics
        gait_metrics.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
