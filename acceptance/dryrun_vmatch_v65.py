#!/usr/bin/env python3
"""v65 dry-run: X1_DISC_VMATCH speed-gated demo sampling (mock, call-level).

Paid lessons applied (v40 wiring; v55 syntax; v49b paired consumers; v61d
consumer contracts): the new sampler is CALLED on faithful fakes, and the
wiring (cfg field, env lever, weight drop) is verified on the REAL files
via AST/regex.

Checks:
  1. sample_times_speed_gated on a synthetic motion bank:
     a) at cmd 1.0, sampled times land ONLY in speed-matched frames
        (|v_frame - 1.0| <= 0.35) of the chosen clip
     b) stand commands (0.0) on overground-only clips fall back to uniform
        (no NaN, times within duration)
     c) window margin respected: time + window <= duration
     d) all three overground clips get fetched at 1.0 (composition check:
        the demo stream at walking commands = overground segments)
  2. AST/regex on the real files:
     a) motion_data_manager.sample_times_speed_gated exists + frame_speed built
     b) animation_manager._sample_demo_time wired in reset/update
     c) AnimationTermCfg.speed_matched_fetch field exists
     d) x1_amp_env_cfg wires X1_DISC_VMATCH + drops 103_07
     e) run_x1_amp_train.final_checkpoint ranks numeric stems above
        non-digit (the v64 on-pod harness bug fix)
"""
import ast
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def main():
    # --- synthetic motion bank -------------------------------------------
    class FakeTerm:
        pass

    term = FakeTerm()
    term.device = "cpu"
    # 4 clips: in-place (v~0.1), overground 1.0, overground 0.4, mixed
    speeds = torch.full((4, 600), float("nan"))
    import numpy as np
    speeds[0, :600] = 0.1                      # treadmill in-place
    speeds[1, :600] = 1.0                      # 36_01-like
    speeds[2, :600] = 0.4                      # 36_11-like
    speeds[3, :300] = 0.0                      # mixed: first half stand
    speeds[3, 300:] = 1.2                      # second half walking
    term.frame_speed = speeds
    term.motion_num_frames = torch.tensor([600, 600, 600, 600], dtype=torch.int32)
    term.motion_dt = torch.tensor([1 / 120.0] * 4)
    term.motion_durations = torch.tensor([600 * (1 / 120.0)] * 4)

    import importlib.util as ilu
    mdp = types.ModuleType("fake_mod")

    # load only the method source (motion_data_manager imports isaaclab)
    src = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
           / "manager_based" / "amp" / "managers" / "motion_data_manager.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "sample_times_speed_gated")
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {"torch": torch}
    exec(compile(mod, "<fn>", "exec"), ns)
    sampler = ns["sample_times_speed_gated"]

    torch.manual_seed(0)
    ids = torch.tensor([0, 1, 2, 3, 1, 2, 3, 0] * 64)
    tgt = torch.full((len(ids),), 1.0)
    times = sampler(term, ids, tgt, window_s=0.6, num_tries=8)
    check("1a no NaN", bool(torch.isfinite(times).all()))
    # verify speed-match for every sampled time
    frames = (times / (1 / 120.0)).long().clamp(max=599)
    sp = term.frame_speed[ids, frames]
    ok = (torch.abs(sp - 1.0) <= 0.35)
    # clips with matched frames: 1 (v=1.0) and 3 (v=1.2 half). Clips 0/2
    # are fully out of band at cmd 1.0 -> documented uniform fallback.
    mask = (ids == 1) | (ids == 3)
    frac_ok = float(ok[mask].float().mean())
    check("1b walking cmd samples speed-matched frames", frac_ok > 0.95,
          f"frac={frac_ok:.3f}")
    # 1c window margin: time + 0.6s <= duration
    dur = term.motion_durations[ids]
    check("1c window margin", bool((times + 0.6 <= dur + 1e-6).all()),
          f"max_end={(times + 0.6).max():.2f} vs {dur.min():.2f}")
    # 1d composition: in-band overground clips (1, 3) get sampled; the
    # fully out-of-band clip 2 must NOT produce matched samples
    used = set(ids[ok].tolist())
    check("1d in-band clips used, out-of-band excluded",
          {1, 3} <= used and 2 not in used and 0 not in used, f"used={sorted(used)}")
    # 1e stand cmd on overground-only clip falls back uniformly (no NaN)
    t2 = sampler(term, torch.tensor([1] * 64), torch.zeros(64), window_s=0.6)
    check("1e stand fallback finite", bool(torch.isfinite(t2).all()))

    # 1f) motion ASSIGNMENT gating (v65.1 full composition): at cmd 1.0
    # the in-place clip (0) must get ~zero assignment mass
    src2 = src  # motion_data_manager source already loaded
    fn2 = ast.parse(src2)
    fn2m = next((n for n in ast.walk(fn2)
                 if isinstance(n, ast.FunctionDef) and n.name == "sample_motions_speed_gated"), None)
    mod2 = ast.Module(body=[fn2m], type_ignores=[])
    ns2 = {"torch": torch}
    exec(compile(mod2, "<fn2>", "exec"), ns2)
    term.motion_weights = torch.tensor([2.0, 3.0, 1.0, 2.0])  # in-place, 36_01-like, 36_11-like, mixed
    assign = ns2["sample_motions_speed_gated"](term, torch.full((2000,), 1.0))
    hist = torch.bincount(assign, minlength=4).float() / 2000
    check("1f assignment excludes in-place at walk cmd", hist[0] < 0.05,
          f"frac per clip = {[round(x,3) for x in hist.tolist()]}")
    assign0 = ns2["sample_motions_speed_gated"](term, torch.zeros(2000))
    hist0 = torch.bincount(assign0, minlength=4).float() / 2000
    check("1f' in-place serves stand cmds", hist0[0] > 0.3,
          f"stand frac = {[round(x,3) for x in hist0.tolist()]}")

    # --- real-file wiring checks -----------------------------------------
    mm = src
    check("2a sampler + frame_speed in motion_data_manager",
          "def sample_times_speed_gated" in mm and "self.frame_speed" in mm
          and "def sample_motions_speed_gated" in mm)
    am_early = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
                / "manager_based" / "amp" / "managers" / "animation_manager.py").read_text()
    check("2f reset() uses gated motion assignment",
          "sample_motions_speed_gated(env_speeds)" in am_early)
    am = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
          / "manager_based" / "amp" / "managers" / "animation_manager.py").read_text()
    check("2b _sample_demo_time wired (reset+update)",
          am.count("self._sample_demo_time(") >= 3)
    ac = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
          / "manager_based" / "amp" / "managers" / "animation_manager_cfg.py").read_text()
    check("2c cfg field speed_matched_fetch", "speed_matched_fetch: bool" in ac)
    ec = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
          / "manager_based" / "amp" / "x1_amp_env_cfg.py").read_text()
    check("2d env lever + weight drop",
          "X1_DISC_VMATCH" in ec and '("103_07", "103_07_mirror")' in ec)
    rt = (ROOT / "roboparty_train" / "run_x1_amp_train.py").read_text()
    check("2e final_checkpoint numeric-first fix",
          '# trained checkpoints win' in rt and 'def key(name):' in rt)

    print("\n== DRYRUN v65:", "ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
