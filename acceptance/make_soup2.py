"""Same-basin 2-policy soup: weighted average of actor layers + normalizer."""
import sys
import numpy as np

a_path, b_path, out_path, wa = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
A, B = np.load(a_path), np.load(b_path)
out = {}
for k in A.files:
    out[k] = wa * A[k] + (1 - wa) * B[k]
np.savez(out_path, **out)
print(f"soup {out_path}: {wa:g}*{a_path.split('/')[-1]} + {1-wa:g}*{b_path.split('/')[-1]}")
