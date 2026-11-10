#!/usr/bin/env bash
# Download ONLY the CMU walking candidate clips (SMPL-H npz, ~0.5MB each)
# from the per-file HF mirror realdream-ai/AMASS. Selected via the CMU motion
# index (subjects whose descriptions are plain walking trials).
set -u
BASE="https://huggingface.co/datasets/realdream-ai/AMASS/resolve/main/raw/CMU"
DST="$(dirname "$0")/../amass/CMU_raw"
mkdir -p "$DST"
CLIPS="
07/07_01 07/07_02 07/07_03 07/07_04 07/07_05 07/07_06 07/07_07 07/07_08 07/07_09 07/07_10 07/07_11 07/07_12
08/08_01 08/08_02 08/08_03 08/08_04 08/08_05 08/08_06 08/08_07 08/08_08 08/08_09 08/08_10 08/08_11
16/16_11 16/16_12 16/16_13 16/16_14 16/16_15 16/16_16 16/16_21 16/16_22 16/16_31 16/16_32 16/16_33 16/16_34 16/16_47 16/16_58
35/35_01 35/35_02 35/35_03 35/35_04 35/35_05 35/35_06 35/35_07 35/35_08 35/35_09 35/35_10 35/35_11 35/35_12 35/35_13 35/35_14 35/35_15 35/35_16
35/35_28 35/35_29 35/35_30 35/35_31 35/35_32 35/35_33 35/35_34
37/37_01
38/38_01 38/38_02
49/49_01 49/49_02 49/49_03 49/49_04
82/82_08 82/82_09 82/82_10 82/82_11 82/82_12 82/82_13 82/82_14
91/91_02 91/91_04 91/91_10 91/91_11 91/91_12
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
