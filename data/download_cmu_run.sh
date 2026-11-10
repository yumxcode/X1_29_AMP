#!/usr/bin/env bash
# Download CMU RUNNING candidate clips (SMPL-H npz, ~0.5MB each) from the
# per-file HF mirror realdream-ai/AMASS. Candidates = CMU index run/jog
# categories (fetched 2026-11-10 from una-dinosauria/cmu-mocap index):
#   09 (run, all trials; IsaacGymEnvs AMP run style = 09_11)
#   16 run/jog straight (35/36/45/46/55/56; veers/turns excluded for v1)
#   35 run/jog (17-20; same subject as the v72 walk family)
#   141/143 plain Run trials; 127 plain Run trials; 02_03 run/jog
set -u
BASE="https://huggingface.co/datasets/realdream-ai/AMASS/resolve/main/raw/CMU"
DST="$(dirname "$0")/../amass/CMU_raw"
mkdir -p "$DST"
CLIPS="
09/09_01 09/09_02 09/09_03 09/09_04 09/09_05 09/09_06 09/09_07 09/09_08 09/09_09 09/09_10 09/09_11
02/02_03
16/16_35 16/16_36 16/16_45 16/16_46 16/16_55 16/16_56
35/35_17 35/35_18 35/35_19 35/35_20
141/141_01 141/141_02 141/141_03 141/141_34
143/143_01 143/143_42
127/127_03 127/127_06 127/127_07 127/127_08
"
ok=0; fail=0
for c in $CLIPS; do
  sub="${c%%/*}"; name="${c##*/}"
  f="$DST/$name.npz"
  [ -s "$f" ] && { ok=$((ok+1)); continue; }
  for attempt in 1 2 3; do
    curl -sL --fail -o "$f" "$BASE/$sub/${name}_poses.npz" && break
    sleep 2
  done
  if [ -s "$f" ]; then ok=$((ok+1)); else fail=$((fail+1)); echo "FAIL $name"; rm -f "$f"; fi
done
echo "downloaded ok=$ok fail=$fail"
du -sh "$DST"
