"""Fast lumbar/backward-lean survey across many rollout npz (joint-space only).

Reports per file: lumbar_pitch mean/p10/p90 (deg, + = chest FORWARD) and
base pitch mean (deg, + = nose up). Joint-space only — no FK needed.
"""
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def survey(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d['meta'].item()))
    names = meta['hinge_names']
    li = names.index('lumbar_pitch_joint')
    lum = np.degrees(d['q'][:, li])
    bq = d['base_quat']
    w, x, y, z = bq[:, 0], bq[:, 1], bq[:, 2], bq[:, 3]
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    n0 = 25  # settle frames 0.5 s
    return (np.mean(lum[n0:]), np.percentile(lum[n0:], 10), np.percentile(lum[n0:], 90),
            np.mean(pitch[n0:]))


if __name__ == '__main__':
    print(f"{'file':44s} {'lum_mean':>8s} {'lum_p10':>8s} {'lum_p90':>8s} {'base_p':>7s}")
    for p in sorted(sys.argv[1:]):
        m, p10, p90, bp = survey(p)
        tag = Path(p).name.replace('_walk10.npz', '')
        print(f"{tag:44s} {m:8.1f} {p10:8.1f} {p90:8.1f} {bp:7.1f}")
