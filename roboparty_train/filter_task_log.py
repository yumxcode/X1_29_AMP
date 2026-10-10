#!/usr/bin/env python3
"""Filter the SDK noise out of a gradmotion task log."""
import re
import sys

txt = open(sys.argv[1], errors="replace").read()
errs = re.findall(r"2026-10-10 (\d[\d:]+)[^\n]{0,40}ERROR[^\n]{0,200}", txt)
print("SDK ERROR moments:", errs[:6], "total", len(errs))
for pat in [r"\[MONITOR\][^\n]{0,150}", r"\[RESUME\][^\n]{0,120}",
            r"\[WAIT\][^\n]{0,100}", r"\[INFO\][^\n]{0,150}",
            r"\[TRAIN\][^\n]{0,150}", r"\[ENV\][^\n]{0,120}",
            r"Phase \d[^\n]{0,120}", r"DONE in[^\n]{0,60}"]:
    ms = re.findall(pat, txt)
    if ms:
        print(f"-- {pat[:14]}: {len(ms)}")
        for m in ms[-4:]:
            print("   ", m[:170])
