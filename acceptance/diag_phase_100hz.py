#!/usr/bin/env python3
"""One-off: is soup59c_50@100Hz G2 'in-phase' verdict a metric artifact or real?

Foot-contact timeline shows alternating strides (offset 19-20 of 40 frames
= half period), but gait_metrics best_lag on HIP JOINT signals reports
phase_frac 0.025 anti=False. Measure the hip signals directly and replicate
best_lag exactly."""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/soup59c_at100.npz"
z = np.load(path, allow_pickle=False)
meta = json.loads(str(z["meta"]))
hinge = meta["hinge_names"]
q = z["q"].astype(float)
iL = hinge.index("left_hip_pitch_joint")
iR = hinge.index("right_hip_pitch_joint")
w = q[50:]
a, b = w[:, iL], w[:, iR]


def detrend(x):
    t = np.arange(len(x), dtype=np.float64)
    A = np.stack([t, np.ones_like(t)], 1)
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


da, db = detrend(a), detrend(b)
da = (da - da.mean()) / (da.std() + 1e-12)
db = (db - db.mean()) / (db.std() + 1e-12)
cc = np.correlate(db, da, mode="full")
lags = np.arange(-len(da) + 1, len(db))
k_max, k_min = int(np.argmax(cc)), int(np.argmin(cc))
print(f"MAX (in-phase)   lag={lags[k_max]*0.01:.3f}s corr=+{cc[k_max]:.0f}")
print(f"MIN (anti-phase) lag={lags[k_min]*0.01:.3f}s corr={cc[k_min]:.0f}")
print(f"anti = {abs(cc[k_min]) > abs(cc[k_max])}")
print(f"raw corr(L,R) = {np.corrcoef(a, b)[0,1]:.3f}")
# spectrum: dominant period of each hip signal
fa = np.fft.rfft(da)
fb = np.fft.rfft(db)
fa, fb = np.abs(fa), np.abs(fb)
pa = np.fft.rfftfreq(len(da), d=0.01)
top_a = pa[np.argsort(fa)[-3:]][::-1]
top_b = pa[np.fft.rfftfreq(len(db), d=0.01).argsort() * 0 + np.argsort(fb)[-3:]][::-1]
print(f"dominant freqs L: {[round(1/f,2) for f in top_a]} Hz (periods {[round(f,2) for f in top_a]}s)")
print(f"dominant freqs R: {[round(1/f,2) for f in top_b]} Hz (periods {[round(f,2) for f in top_b]}s)")
