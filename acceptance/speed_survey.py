"""Distance/speed for walk10 rollout npz: x-displacement of base over the
post-settle window / duration."""
import sys, json
from pathlib import Path
import numpy as np

for p in sorted(sys.argv[1:]):
    d = np.load(p, allow_pickle=True)
    t, bp = d['t'], d['base_pos']
    n0 = 25
    dt = t[-1] - t[n0]
    dist = bp[-1, 0] - bp[n0, 0]
    tag = Path(p).name.replace('_walk10.npz', '')
    print(f"{tag:24s} dist={dist:6.2f}  speed={dist/dt:5.3f} m/s")
