#!/usr/bin/env bash
# v94 eval battery: rollouts for a ckpt tag (walk10/walk05/back05/walkturn)
set -e
TAG=$1
CKPT=acceptance/v83_eval/${TAG}.policy.npz
OUT=acceptance/v83_eval
export X1_CLOCK_T_A=1.61 X1_CLOCK_T_B=0.66
PY=.venv-mac/bin/python
run() {  # name cmd...
  local name=$1; shift
  [ -f "$OUT/${TAG}_${name}.npz" ] && { echo "skip ${TAG}_${name}"; return; }
  $PY sim2sim/mujoco_rollout.py --ckpt "$CKPT" --log "$OUT/${TAG}_${name}.npz" "$@"
}
run walk10  --cmd 1.0 0.0 0.0 --duration 12
run walk05  --cmd 0.5 0.0 0.0 --duration 12
run back05  --cmd -0.5 0.0 0.0 --duration 12
run walkturn --cmd 0.6 0.0 0.5 --duration 10
echo "battery done: $TAG"
