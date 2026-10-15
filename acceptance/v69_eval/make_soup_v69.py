#!/usr/bin/env python3
"""v69: soup-average actor weights of ladder checkpoints (v68 family technique).

Averaging the actor MLP + obs normalizer of N checkpoints. Drifts alternate
sign across the v69 ladder (LR-restart adaptation oscillation) — soups cancel
the transient while keeping the converged posture fix.
"""
import functools
import sys

import numpy as np

print = functools.partial(print, flush=True)


def main():
    out = sys.argv[1]
    paths = sys.argv[2:]
    acc = None
    for p in paths:
        z = np.load(p)
        if acc is None:
            acc = {k: z[k].astype(np.float64) for k in z.files}
        else:
            assert set(z.files) == set(acc), f"{p}: key mismatch"
            for k in z.files:
                acc[k] += z[k]
    n = len(paths)
    for k in acc:
        acc[k] /= n
    np.savez(out, **{k: v.astype(np.float32) for k, v in acc.items()})
    print(f"[SOUP] {n} members -> {out}")


if __name__ == "__main__":
    main()
