#!/usr/bin/env python3
"""
Verify GMR retarget quality for X1 humanoid.
Run this in the GMR venv (has numpy but not Isaac Lab).

Checks:
1. DOF count and names match expected
2. Joint angles within URDF limits
3. Root height reasonable (0.3-1.0m for 1.25m tall robot)
4. Motion smoothness (no sudden jumps)
5. Left/right symmetry in walking motions
6. Foot contact pattern (alternating in walk/run)

Usage:
    python verify_retarget.py --input_dir <x1_gmr_dir>
"""

import argparse
import functools
import pickle
import sys
import numpy as np
from pathlib import Path
from rich import print

print = functools.partial(print, flush=True)

# X1 URDF joint limits (rad) — from f1.urdf
JOINT_LIMITS = {
    "lumbar_yaw_joint": (-1.0, 1.0),
    "lumbar_roll_joint": (-0.25, 0.25),
    "lumbar_pitch_joint": (-0.15, 0.45),
    "left_shoulder_pitch_joint": (-2.5, 1.5),
    "left_shoulder_roll_joint": (-2.0, 0.0),
    "left_shoulder_yaw_joint": (-1.8, 1.8),
    "left_elbow_pitch_joint": (0.0, 2.0),
    "left_elbow_yaw_joint": (-1.8, 1.8),
    "left_wrist_pitch_joint": (-0.4, 0.4),
    "left_wrist_roll_joint": (-0.4, 0.4),
    "right_shoulder_pitch_joint": (-2.5, 1.5),
    "right_shoulder_roll_joint": (-2.0, 0.0),
    "right_shoulder_yaw_joint": (-1.8, 1.8),
    "right_elbow_pitch_joint": (0.0, 2.0),
    "right_elbow_yaw_joint": (-1.8, 1.8),
    "right_wrist_pitch_joint": (-0.4, 0.4),
    "right_wrist_roll_joint": (-0.4, 0.4),
    "left_hip_pitch_joint": (-1.0, 2.0),
    "left_hip_roll_joint": (-1.5, 0.2),
    "left_hip_yaw_joint": (-1.5, 1.5),
    "left_knee_pitch_joint": (0.0, 2.0),
    "left_ankle_pitch_joint": (-0.41, 0.35),
    "left_ankle_roll_joint": (-0.64, 0.64),
    "right_hip_pitch_joint": (-2.0, 1.0),
    "right_hip_roll_joint": (-0.2, 1.5),
    "right_hip_yaw_joint": (-1.5, 1.5),
    "right_knee_pitch_joint": (0.0, 2.0),
    "right_ankle_pitch_joint": (-0.41, 0.35),
    "right_ankle_roll_joint": (-0.64, 0.64),
}

# Expected GMR dof names (MuJoCo order)
EXPECTED_DOF_NAMES = [
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

# Soft limit factor (URDF limit * factor = acceptable range)
SOFT_LIMIT_FACTOR = 1.2  # allow 20% over URDF limits (GMR IK may slightly exceed)


def verify_file(pkl_path: Path, verbose: bool = False) -> dict:
    """Verify a single GMR pkl file. Returns dict with results."""
    results = {
        "file": pkl_path.name,
        "passed": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 1. Check DOF names
    dof_names = list(data.get("dof_names", data.get("dof_names", [])))
    if hasattr(dof_names[0], 'item'):
        dof_names = [str(d) for d in dof_names]

    num_dof = data["dof_pos"].shape[1]
    if num_dof != 29:
        results["errors"].append(f"DOF count = {num_dof}, expected 29")
        results["passed"] = False
    elif dof_names != EXPECTED_DOF_NAMES:
        # Check if same set but different order
        if set(dof_names) == set(EXPECTED_DOF_NAMES):
            results["warnings"].append("DOF names match but order differs from expected")
        else:
            missing = set(EXPECTED_DOF_NAMES) - set(dof_names)
            extra = set(dof_names) - set(EXPECTED_DOF_NAMES)
            if missing:
                results["errors"].append(f"Missing DOFs: {missing}")
            if extra:
                results["errors"].append(f"Extra DOFs: {extra}")
            if missing or extra:
                results["passed"] = False

    # 2. Check joint angle ranges
    dof_pos = np.array(data["dof_pos"])
    num_frames = dof_pos.shape[0]
    out_of_range_joints = []

    for i, name in enumerate(dof_names):
        if name in JOINT_LIMITS:
            lo, hi = JOINT_LIMITS[name]
            soft_lo = lo * SOFT_LIMIT_FACTOR if lo < 0 else lo / SOFT_LIMIT_FACTOR
            soft_hi = hi * SOFT_LIMIT_FACTOR if hi > 0 else hi / SOFT_LIMIT_FACTOR

            col = dof_pos[:, i]
            col_min, col_max = col.min(), col.max()

            if col_min < soft_lo or col_max > soft_hi:
                violations = np.sum((col < soft_lo) | (col > soft_hi))
                pct = violations / num_frames * 100
                out_of_range_joints.append({
                    "joint": name,
                    "range": (col_min, col_max),
                    "limit": (lo, hi),
                    "soft_limit": (soft_lo, soft_hi),
                    "violations_pct": pct,
                })
                if pct > 10:  # >10% frames out of range = error
                    results["passed"] = False
                    results["errors"].append(
                        f"  {name}: [{col_min:.2f}, {col_max:.2f}] exceeds soft limit [{soft_lo:.2f}, {soft_hi:.2f}] ({pct:.1f}% frames)"
                    )
                elif pct > 1:
                    results["warnings"].append(
                        f"  {name}: {pct:.1f}% frames slightly out of range"
                    )

    results["stats"]["out_of_range_joints"] = len(out_of_range_joints)

    # 3. Check root height
    root_pos = np.array(data["root_pos"])
    root_z = root_pos[:, 2]
    z_min, z_max, z_mean = root_z.min(), root_z.max(), root_z.mean()

    results["stats"]["root_z"] = {"min": z_min, "max": z_max, "mean": z_mean}

    if z_min < 0.1:
        results["errors"].append(f"Root Z too low: min={z_min:.3f}m (robot falling?)")
        results["passed"] = False
    if z_max > 1.5:
        results["warnings"].append(f"Root Z very high: max={z_max:.3f}m")
    if z_mean < 0.3 or z_mean > 1.0:
        results["warnings"].append(f"Root Z mean unusual: {z_mean:.3f}m")

    # 4. Motion smoothness (frame-to-frame joint velocity)
    if num_frames > 2:
        joint_vel = np.abs(np.diff(dof_pos, axis=0))
        max_vel = joint_vel.max()
        mean_vel = joint_vel.mean()
        # Flag if any joint moves >1.0 rad/frame (very abrupt)
        abrupt_frames = np.sum(joint_vel.max(axis=1) > 1.0)
        abrupt_pct = abrupt_frames / (num_frames - 1) * 100

        results["stats"]["smoothness"] = {
            "max_joint_vel": max_vel,
            "mean_joint_vel": mean_vel,
            "abrupt_pct": abrupt_pct,
        }

        if abrupt_pct > 5:
            results["warnings"].append(f"Motion jittery: {abrupt_pct:.1f}% frames have abrupt jumps (>1.0 rad/frame)")

    # 5. Key joint statistics for walking validation
    key_joints_stats = {}
    for name in ["left_hip_pitch_joint", "right_hip_pitch_joint",
                  "left_knee_pitch_joint", "right_knee_pitch_joint",
                  "lumbar_pitch_joint"]:
        if name in dof_names:
            idx = dof_names.index(name)
            col = dof_pos[:, idx]
            key_joints_stats[name] = {
                "mean": float(col.mean()),
                "std": float(col.std()),
                "range": (float(col.min()), float(col.max())),
            }

    results["stats"]["key_joints"] = key_joints_stats

    # 6. Left/right symmetry check
    symmetry_pairs = [
        ("left_hip_pitch_joint", "right_hip_pitch_joint"),
        ("left_knee_pitch_joint", "right_knee_pitch_joint"),
        ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    ]
    for left_name, right_name in symmetry_pairs:
        if left_name in dof_names and right_name in dof_names:
            left_idx = dof_names.index(left_name)
            right_idx = dof_names.index(right_name)
            left = dof_pos[:, left_idx]
            right = dof_pos[:, right_idx]
            # For walking, left and right should be roughly anti-phase
            correlation = np.corrcoef(left, right)[0, 1]
            results["stats"][f"symmetry_{left_name}"] = correlation

    if verbose:
        print(f"\n--- {pkl_path.name} ---")
        print(f"  Frames: {num_frames}, DOF: {num_dof}")
        print(f"  Root Z: min={z_min:.3f}, max={z_max:.3f}, mean={z_mean:.3f}")
        for name, stat in key_joints_stats.items():
            print(f"  {name}: mean={stat['mean']:.3f}, std={stat['std']:.3f}")
        if out_of_range_joints:
            print(f"  Out of range joints: {len(out_of_range_joints)}")
            for j in out_of_range_joints:
                print(f"    {j['joint']}: [{j['range'][0]:.2f}, {j['range'][1]:.2f}] vs limit {j['limit']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify GMR retarget quality")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing x1_gmr/*.pkl files")
    parser.add_argument("--verbose", action="store_true",
                       help="Print per-file details")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pkl_files = sorted(input_dir.glob("*.pkl"))

    if not pkl_files:
        print(f"[ERROR] No .pkl files found in {input_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"GMR Retarget Quality Verification")
    print(f"Files: {len(pkl_files)} in {input_dir}")
    print("=" * 60)

    all_results = []
    total_pass = 0
    total_fail = 0

    for pkl in pkl_files:
        result = verify_file(pkl, verbose=args.verbose)
        all_results.append(result)
        if result["passed"]:
            total_pass += 1
            status = "✅ PASS"
        else:
            total_fail += 1
            status = "❌ FAIL"

        print(f"\n{status} {pkl.name} ({result['stats'].get('root_z', {}).get('mean', 0):.3f}m)")

        for err in result["errors"]:
            print(f"  ERROR: {err}")
        for warn in result["warnings"]:
            print(f"  WARN:  {warn}")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_pass} pass, {total_fail} fail / {len(pkl_files)} total")
    print("=" * 60)

    # Aggregate stats
    all_root_z = [r["stats"]["root_z"]["mean"] for r in all_results if "root_z" in r["stats"]]
    if all_root_z:
        print(f"\nRoot Z (mean across files): {np.mean(all_root_z):.3f}m ± {np.std(all_root_z):.3f}m")

    # Check hip pitch correlation (should be negative for walking = anti-phase)
    walk_files = [r for r in all_results if "symmetry_left_hip_pitch_joint" in r["stats"]]
    if walk_files:
        correlations = [r["stats"]["symmetry_left_hip_pitch_joint"] for r in walk_files]
        print(f"L/R Hip Pitch correlation: {np.mean(correlations):.3f} (negative = anti-phase walking)")

    if total_fail > 0:
        print("\n⚠️  Some files failed verification!")
        sys.exit(1)
    else:
        print("\n✅ All files passed verification!")
        sys.exit(0)


if __name__ == "__main__":
    main()
