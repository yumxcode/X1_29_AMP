#!/usr/bin/env python3
"""Set up the LOCAL GMR clone for X1 retargeting (one-off).

Reuses the exact registration logic of run_gmr_retarget.py (same patches),
plus:
  - SMPLX body models from AMASS_minimal/smplx (pkl)
  - smpl.py ext='pkl' patch
  - auto-IK config (acceptance/v25_unpacked/smplx_to_x1_auto.json from the
    last remote calibration run; robot unchanged -> still valid) as the
    ACTIVE ik_configs/smplx_to_x1.json
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "roboparty_train"))
from run_gmr_retarget import register_x1_in_gmr  # noqa: E402

GMR = ROOT / "_gmr_local" / "GMR"

register_x1_in_gmr(GMR)

# body models
smplx_pkl = ROOT / "AMASS_minimal" / "smplx" / "SMPLX_NEUTRAL.pkl"
bm = GMR / "assets" / "body_models" / "smplx"
bm.mkdir(parents=True, exist_ok=True)
for g in ["NEUTRAL", "MALE", "FEMALE"]:
    t = bm / f"SMPLX_{g}.pkl"
    if not t.exists() and smplx_pkl.exists():
        shutil.copy2(smplx_pkl, t)
        print(f"[OK] SMPLX_{g}.pkl copied")

# smpl.py ext patch
sp = GMR / "general_motion_retargeting" / "utils" / "smpl.py"
c = sp.read_text()
if "ext='pkl'" not in c and 'ext="pkl"' not in c:
    c = c.replace("use_pca=False,", "use_pca=False,\n        ext='pkl',")
    sp.write_text(c)
    print("[OK] patched smpl.py ext='pkl'")
else:
    print("[INFO] smpl.py already patched")

# auto IK config -> active
auto = ROOT / "acceptance" / "v25_unpacked" / "smplx_to_x1_auto.json"
ik_dst = GMR / "general_motion_retargeting" / "ik_configs" / "smplx_to_x1.json"
if auto.exists():
    shutil.copy2(auto, ik_dst)
    print(f"[OK] auto-IK config active: {ik_dst.name}")
else:
    print("[WARN] auto config missing; manual config stays active")
print("[DONE]")
