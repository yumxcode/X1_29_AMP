#!/usr/bin/env python3
"""v38 launcher: discriminator-led gait (disc SEES the arms) + intermediate
randomization + in-container P6 evidence fix. Three audit items in one run.

1. DISCRIMINATOR OBS 6 -> 10 key bodies (the user's goal-4 direction):
   the disc could not see arm swing (only knee/ankle/elbow endpoints) —
   the structural reason style score never shaped arms (v33b style 2.00
   with collapsed arms vs v37 0.80 with good form). v32 dataset
   (x1_lab_v32) extends key_body_pos with L/R shoulder_pitch +
   L/R wrist_pitch via MuJoCo FK (gate <= 1.2 mm); disc_obs_dim is
   injected dynamically (rsl_rl amp.py), so the discriminator input
   grows 18 -> 30 automatically. The checkpoint resume is unaffected
   (only the actor/critic/normalizers are saved — the disc is re-init
   every run by design).
   Also fixed en route: mirror_lab_motions.py parsed gmr_dof_names
   instead of lab_dof_names (wrong column order since v28 — kb was
   correct, dof only fed replay buffers); mirrors regenerated.

2. X1_ROBUST_TRAIN=3 (push +-0.5 @5-10s): v37 (0.4) held all form gates
   but push1.0 fell to 0/5 (v35 full-strength 3/5). Intermediate dose
   targets push robustness + walk10 hip/knee symmetry (0.823/0.837,
   both narrow misses vs 0.85) while keeping the P3 kernels >= 0.82
   (v37 plateau 0.843 gives 0.023 margin).

3. P6_play: isaac_play_dump.py (physics-only Isaac rollout, no GL) +
   skeleton_render.py (pure-numpy stick-figure GIF, matplotlib Agg —
   FK verified 0.0 mm vs MuJoCo locally) wired as the final fallback
   in run_x1_amp_train.py — the container has neither GL nor egress,
   which is why every run since v31 failed P6 on missing video.

Accept gates (v37 baseline in parens): platform P3a >= 0.82 (0.843) AND
P7 8/8 (8/8) AND walk05 knee >= 0.85 (0.927) AND walk10 hip/knee >= 0.85
(0.823/0.837 MISS) AND robustness >= 48/60 with push1.0 >= 2/5 (48, 0/5)
AND drift <= 1.0 m (0.94) AND arm swing joint >= 40 deg (29.1/29.2; ref
92/83) AND P6_play PASS (FAIL).
"""
import os
import runpy
import sys
import urllib.request
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"      # intermediate: push +-0.5 @5-10s
os.environ["X1_FINE_TUNE_ITERS"] = "1500"  # +1500 from v37 model_4798 (total 6298)

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_4798_v37.pt"
_OSS_URL = ("https://limx-gradmotion.oss-cn-beijing.aliyuncs.com/"
            "upload%2F2026%2F9%2F12%2Fmodel_4798_20260912153748A525.pt"
            "?OSSAccessKeyId=LTAI5tMec8RQN1nZuRkVMgxz&Expires=1789804348"
            "&Signature=i6WunKn%2B0riHWTwgc4ndd2KKuA8%3D")

if os.environ.get("X1_RESUME_CKPT", "").strip():
    pass  # explicit override wins
elif _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_4798*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        print(f"[RESUME] fetching base checkpoint from OSS ({_OSS_URL[:60]}...)")
        _EMBEDDED.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_OSS_URL, _EMBEDDED)
        print(f"[RESUME] fetched {_EMBEDDED} ({_EMBEDDED.stat().st_size // 1024}KB)")
        os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
