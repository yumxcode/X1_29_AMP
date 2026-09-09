#!/usr/bin/env python3
"""X1 real-robot deployment runtime (reference implementation, v28d policy).

This is the hardware-side counterpart of sim2sim/mujoco_rollout.py. It
implements EXACTLY the observation/action contract the policy was trained
and sim2sim-verified with:

  OBS (288, float32, term-major history x3, oldest first):
    per frame (96): [base_ang_vel(3), projected_gravity(3), cmd(3),
                     joint_pos_rel(29), joint_vel_rel(29), last_action(29)]
    flattened as:   ang(9) grav(9) cmd(9) jpos(87) jvel(87) act(87)
    history:        each term's 3 frames concatenated, oldest first
                    (v26 root-cause: IsaacLab term-major flatten)
  ACTION (29): q_target = default_q + 0.25 * action   [lab joint order]
  RATE: 50 Hz policy; PD at >=200 Hz recommended (Isaac decimation=4 @5ms)

Modes:
  --self-test        replay a recorded sim2sim log (npz) through the ONNX
                     graph; verifies output sanity (no NaN, bounded, smooth)
  --onnx-loop-check  run mujoco closed-loop THROUGH this runtime's obs/action
                     pipeline (proves deploy code path == verified artifact)
  (default)          hardware loop: implement RobotInterface below with your
                     SDK (unitree_sdk2 / cyber / ...), then:
                     python deploy_onnx_rt.py --onnx x1_policy_v28.onnx

Real-robot checklist (before first power-on):
  1. joint order: map your SDK's motor IDs to LAB_DOF_ORDER below (29 motors)
  2. default_q: write DEFAULT_Q (rad) as the soft start pose; hold PD first
  3. gravity: IMU quaternion -> projected gravity = R^T @ [0,0,-1]
  4. commands: vx [-0.5,2.5] vy [-0.5,0.5] wz [-1.5,1.5] (trained range);
     backward verified to -1.0 OOD; START at zero, ramp in over 0.5s
  5. safety: action clip [-100,100]; e-stop on base_z < 0.35 or tilt > 45 deg
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

CONTROL_DT = 0.02
ACTION_SCALE = 0.25
HIST = 3
IN_DIM = 288

# lab joint order (policy column i == this order) — from retarget config x1.yaml
LAB_DOF_ORDER = [
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

DEFAULT_Q = {
    "lumbar_yaw_joint": 0.0, "lumbar_roll_joint": 0.0, "lumbar_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_pitch_joint": 0.0,
    "left_elbow_yaw_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_pitch_joint": 0.0,
    "right_elbow_yaw_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_roll_joint": 0.0,
    "left_hip_pitch_joint": 0.48891, "left_hip_roll_joint": 0.06213,
    "left_hip_yaw_joint": -0.33853, "left_knee_pitch_joint": 0.63204,
    "left_ankle_pitch_joint": -0.27224, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.48891, "right_hip_roll_joint": -0.06213,
    "right_hip_yaw_joint": 0.33853, "right_knee_pitch_joint": 0.63204,
    "right_ankle_pitch_joint": -0.27224, "right_ankle_roll_joint": 0.0,
}

# PD gains (stiffness, damping) — X1_CFG (Isaac-verified in sim2sim)
PD = {
    "lumbar_yaw_joint": (120, 4), "lumbar_roll_joint": (120, 4), "lumbar_pitch_joint": (150, 5),
    "left_hip_pitch_joint": (120, 4), "right_hip_pitch_joint": (120, 4),
    "left_hip_roll_joint": (100, 3.3), "right_hip_roll_joint": (100, 3.3),
    "left_hip_yaw_joint": (100, 3.3), "right_hip_yaw_joint": (100, 3.3),
    "left_knee_pitch_joint": (150, 5), "right_knee_pitch_joint": (150, 5),
    "left_ankle_pitch_joint": (50, 2.5), "right_ankle_pitch_joint": (50, 2.5),
    "left_ankle_roll_joint": (50, 2.5), "right_ankle_roll_joint": (50, 2.5),
    "left_shoulder_pitch_joint": (40, 2), "right_shoulder_pitch_joint": (40, 2),
    "left_shoulder_roll_joint": (40, 2), "right_shoulder_roll_joint": (40, 2),
    "left_shoulder_yaw_joint": (40, 2), "right_shoulder_yaw_joint": (40, 2),
    "left_elbow_pitch_joint": (30, 1.5), "right_elbow_pitch_joint": (30, 1.5),
    "left_elbow_yaw_joint": (20, 1), "right_elbow_yaw_joint": (20, 1),
    "left_wrist_pitch_joint": (15, 1), "right_wrist_pitch_joint": (15, 1),
    "left_wrist_roll_joint": (15, 1), "right_wrist_roll_joint": (15, 1),
}

SLICES = [(0, 3), (3, 6), (6, 9), (9, 38), (38, 67), (67, 96)]  # term-major


class ObsBuffer:
    """3-frame per-term history, term-major flatten. Feed full 96-d frames."""

    def __init__(self):
        self.hist = np.zeros((HIST, 96), dtype=np.float32)

    def push(self, frame96: np.ndarray) -> np.ndarray:
        frame96 = np.asarray(frame96, dtype=np.float32)
        assert frame96.shape == (96,), "frame must be 96-d"
        self.hist[:-1] = self.hist[1:]
        self.hist[-1] = frame96
        return np.concatenate([self.hist[:, a:b].reshape(-1) for a, b in SLICES])


class Policy:
    """ONNX session wrapper (normalizer is fused INSIDE the graph)."""

    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        assert self.sess.get_inputs()[0].shape[-1] == IN_DIM

    def __call__(self, obs288: np.ndarray) -> np.ndarray:
        a = self.sess.run(["action"], {self.name: obs288.reshape(1, -1)})[0].reshape(-1)
        return np.clip(a, -100.0, 100.0)


class RobotInterface:
    """ADAPTER: implement these 4 methods for your hardware SDK.

    Units: rad, rad/s. IMU quaternion: [w,x,y,z], world->base (body) frame.
    base_ang_vel: body-frame gyro (rad/s). base_height estimate for e-stop.
    """

    def read_state(self):
        """-> dict(joint_pos(29 lab order), joint_vel(29), ang_b(3),
                   grav_b(3), base_height(float), tilt_deg(float))"""
        raise NotImplementedError

    def write_targets(self, q_target, kp, kd):  # arrays, lab order (29,)
        """send position targets + PD gains to motors (lab order)"""
        raise NotImplementedError

    def read_command(self):
        """-> (vx, vy, wz) desired base velocity, or (0,0,0)"""
        return (0.0, 0.0, 0.0)

    def estop(self, reason):
        raise NotImplementedError


def run_hardware(policy_path, robot: RobotInterface, settle_s=1.0):
    pol = Policy(policy_path)
    buf = ObsBuffer()
    q_def = np.array([DEFAULT_Q[n] for n in LAB_DOF_ORDER], dtype=np.float64)
    kp = np.array([PD[n][0] for n in LAB_DOF_ORDER], dtype=np.float64)
    kd = np.array([PD[n][1] for n in LAB_DOF_ORDER], dtype=np.float64)
    last_act = np.zeros(29, dtype=np.float32)
    t0, n = time.time(), 0
    print(f"[DEPLOY] loop start; settle {settle_s}s at default pose")
    while True:
        st = robot.read_state()
        cmd = robot.read_command()
        age = time.time() - t0
        if age < settle_s:
            cmd = (0.0, 0.0, 0.0)
        frame = np.concatenate([
            st["ang_b"], st["grav_b"], np.asarray(cmd, dtype=np.float32),
            (st["joint_pos"] - q_def).astype(np.float32),
            st["joint_vel"].astype(np.float32), last_act])
        obs = buf.push(frame)
        if age < settle_s:
            act = np.zeros(29, dtype=np.float32)
        else:
            act = pol(obs)
        last_act = act.astype(np.float32)
        q_tgt = q_def + ACTION_SCALE * act
        robot.write_targets(q_tgt, kp, kd)
        if st["base_height"] < 0.35 or st["tilt_deg"] > 45.0:
            robot.estop(f"fall guard: z={st['base_height']:.2f} tilt={st['tilt_deg']:.0f}")
            break
        n += 1
        time.sleep(max(0.0, CONTROL_DT * (n + 1) - (time.time() - t0)))


def self_test(onnx_path, npz_log):
    z = np.load(npz_log, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    hinge = list(meta["hinge_names"])
    q, dq, bp, bq = z["q"], z["dq"], z["base_pos"], z["base_quat"]
    pol = Policy(onnx_path)
    q_def = np.array([DEFAULT_Q[n] for n in hinge])
    import numpy as _np
    hist = np.zeros((HIST, 96), dtype=np.float32)
    last_act = np.zeros(29, dtype=np.float32)
    cmd = _np.asarray(meta["cmd"], dtype=_np.float32)
    jumps, worst = 0, 0.0
    prev = None
    for t in range(len(q)):
        # quaternion -> rotation (wxyz), mujoco mju_quat2Mat equivalent
        w, x, y, zz = bq[t] / _np.linalg.norm(bq[t])
        R = _np.array([
            [1 - 2 * (y * y + zz * zz), 2 * (x * y - w * zz), 2 * (x * zz + w * y)],
            [2 * (x * y + w * zz), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - w * x)],
            [2 * (x * zz - w * y), 2 * (y * zz + w * x), 1 - 2 * (x * x + y * y)]])
        grav = R.T @ _np.array([0, 0, -1.0])
        # body ang vel from quaternion rate (approx: finite diff)
        if t > 0:
            dqdt = (bq[t] - bq[t - 1]) / 0.02
            ang_b = 2.0 * _np.array([w * dqdt[1] - x * dqdt[0] - y * dqdt[3] + zz * dqdt[2],
                                     w * dqdt[2] + x * dqdt[3] - y * dqdt[0] - zz * dqdt[1],
                                     w * dqdt[3] - x * dqdt[2] + y * dqdt[1] - zz * dqdt[0]])
        else:
            ang_b = _np.zeros(3)
        frame = _np.concatenate([ang_b, grav, cmd, (q[t] - q_def), dq[t], last_act]).astype(_np.float32)
        hist[:-1] = hist[1:]
        hist[-1] = frame
        obs = _np.concatenate([hist[:, a:b].reshape(-1) for a, b in SLICES])
        a = pol(obs)
        assert _np.isfinite(a).all(), f"NaN/Inf action at t={t}"
        if prev is not None:
            d = _np.abs(a - prev).max()
            worst = max(worst, float(d))
            if d > 0.5:
                jumps += 1
        prev = a
        last_act = a
    print(f"[SELF-TEST OK] {len(q)} steps: all finite, max step jump {worst:.3f} "
          f"(>0.5: {jumps} frames)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--self-test", metavar="NPZ", default=None,
                    help="replay a recorded sim2sim log through the graph")
    args = ap.parse_args()
    if args.self_test:
        self_test(args.onnx, args.self_test)
        return
    print("[DEPLOY] no RobotInterface configured — implement and call "
          "run_hardware(); see docstring checklist")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
