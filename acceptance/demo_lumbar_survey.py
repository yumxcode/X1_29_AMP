"""Demo-library lumbar/chest survey: per-source dof_pos lumbar_pitch stats.

Reports per pkl: lumbar_pitch mean/p10/p90 (deg, + = chest FORWARD),
root pitch mean (deg), chest pitch = root pitch + lumbar (approx, deg).
"""
import sys
import pickle
from pathlib import Path

import numpy as np

MOT = Path('roboparty_train/robolab/data/motions/x1_lab_v72')

# dof column order in the retarget pkls = Isaac Lab order (lab_dof_names in
# scripts/tools/retarget/config/x1.yaml, VERIFIED TASK_20260810_036)
LAB = ["left_hip_pitch", "lumbar_yaw", "right_hip_pitch", "left_hip_roll", "lumbar_roll",
       "right_hip_roll", "left_hip_yaw", "lumbar_pitch", "right_hip_yaw", "left_knee",
       "left_shoulder_pitch", "right_shoulder_pitch", "right_knee", "left_ankle_pitch",
       "left_shoulder_roll", "right_shoulder_roll", "right_ankle_pitch", "left_ankle_roll",
       "left_shoulder_yaw", "right_shoulder_yaw", "right_ankle_roll", "left_elbow_pitch",
       "right_elbow_pitch", "left_elbow_yaw", "right_elbow_yaw", "left_wrist_pitch",
       "right_wrist_pitch", "left_wrist_roll", "right_wrist_roll"]
LI, SHL, SHR, ELL, ELR = (LAB.index(x) for x in ("lumbar_pitch", "left_shoulder_pitch",
                                                 "right_shoulder_pitch", "left_elbow_pitch",
                                                 "right_elbow_pitch"))


def root_pitch_deg(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))


if __name__ == '__main__':
    files = sorted(sys.argv[1:]) or sorted(str(p) for p in MOT.glob('*.pkl') if '_mirror' not in p.name)
    print(f"{'source':36s} {'lum_mean':>8s} {'p10':>6s} {'p90':>6s} {'root_p':>7s} {'shL':>6s} {'shR':>6s} {'elbL':>6s} {'elbR':>6s}")
    for f in files:
        d = pickle.load(open(f, 'rb'))
        q = np.asarray(d['dof_pos'])
        lum = np.degrees(q[:, LI])
        rp = root_pitch_deg(np.asarray(d['root_rot']))
        shL = np.degrees(q[:, SHL]); shR = np.degrees(q[:, SHR])
        elL = np.degrees(q[:, ELL]); elR = np.degrees(q[:, ELR])
        print(f"{Path(f).stem:36s} {np.mean(lum):8.1f} {np.percentile(lum,10):6.1f} {np.percentile(lum,90):6.1f} "
              f"{np.mean(rp):7.1f} {np.mean(shL):6.1f} {np.mean(shR):6.1f} {np.mean(elL):6.1f} {np.mean(elR):6.1f}")
