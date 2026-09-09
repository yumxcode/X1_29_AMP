#!/usr/bin/env python3
"""v29e clean-phase fine-tune launcher.

Resumes from the v29d robust fine-tune checkpoint (model_9497,
TASK_20260909_231) for +1000 iterations with the v29 randomization DISABLED
(exact v28d environment). Hypothesis under test: the policy keeps (most of)
its internalized push/latency recovery while the gait re-converges to the
AMP style — recovering walk05 knee symmetry and the platform kernel gates
that v29d narrowly missed (lin 0.811 vs 0.82, ang 0.472 vs 0.5).

Checkpoint sourcing (in priority order):
  1. repo-embedded roboparty_train/checkpoints/model_9497.pt (16.9MB)
  2. platform resume mount (repo root model_9497*.pt — same-account OSS)
  3. signed OSS download URL below (cross-account path; expires 2026-09-16,
     added after the local GitHub push of the embedded ckpt was blocked by
     a network outage and the OSS mount was unverified cross-account)
"""
import os
import runpy
import sys
import urllib.request
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "0"       # disable push/delay randomization
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # +1000 clean iterations

_HERE = Path(__file__).resolve().parent
_CKPT = _HERE / "checkpoints" / "model_9497.pt"
_OSS_URL = "https://limx-gradmotion.oss-cn-beijing.aliyuncs.com/upload%2F2026%2F9%2F10%2Fmodel_9497_20260910011210A437.pt?OSSAccessKeyId=LTAI5tMec8RQN1nZuRkVMgxz&Expires=1789580413&Signature=HJlaXdkmtymRy3vrPTEuwWrwtRg%3D"

if _CKPT.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_CKPT)
else:
    # fall back to the platform mount if present, else pull from OSS
    mounted = sorted((p for p in _HERE.parent.glob("model_9497*.pt")))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        print(f"[RESUME] fetching base checkpoint from OSS ({_OSS_URL[:60]}...)")
        _CKPT.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_OSS_URL, _CKPT)
        print(f"[RESUME] fetched {_CKPT} ({_CKPT.stat().st_size // 1024}KB)")
        os.environ["X1_RESUME_CKPT"] = str(_CKPT)

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
