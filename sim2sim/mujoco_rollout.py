#!/usr/bin/env python3
"""
X1 sim2sim: Isaac-trained AMP policy -> MuJoCo (gmr_x1_assets/x1.xml) rollout.

Faithfulness notes (deltas vs Isaac documented in the report):
- Control loop: 50 Hz policy, PD torques applied every physics substep with a
  held target — matches Isaac decimation=4 @ sim.dt 0.005. MuJoCo xml runs at
  0.001 s (GMR vendor model), so we take 20 substeps per control step.
- PD gains / effort limits: exactly X1_CFG (robolab assets/robots/x1.py).
- Joint damping/frictionloss present in x1.xml (vendor values) ADD to the PD;
  Isaac had URDF damping=1.0 on all joints instead. Documented approximation.
- Observations replicate the policy obs group exactly (no noise, PLAY cfg):
    [base_ang_vel(3), projected_gravity(3), cmd(3), joint_pos_rel(29),
     joint_vel_rel(29), last_action(29)] x history 3 (oldest-first).
- Action: q_target = default_q + 0.25 * a  (JointPositionActionCfg scale).
- MuJoCo free-joint qvel[3:6] is body-frame angular velocity == Isaac
  base_ang_vel (verified empirically 2026-08-17).

Usage:
  PYTHONPATH=pylibs python sim2sim/mujoco_rollout.py \
      --ckpt model_4000.pt --repo-root . \
      [--cmd 1.0 0.0 0.0] [--duration 12] [--video out.mp4]
"""

import argparse
import functools
import json
import re
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

CONTROL_DT = 0.02
HIST = 3
ACTION_SCALE = 0.25
FALL_Z = 0.30          # base height below which we count a fall (MuJoCo frame)
FALL_TILT = 60.0       # deg

# X1_CFG PD gains / effort limits (Isaac -> MuJoCo by joint name)
PD = {
    "lumbar_yaw_joint": (120.0, 4.0, 180.0),
    "lumbar_roll_joint": (120.0, 4.0, 180.0),
    "lumbar_pitch_joint": (150.0, 5.0, 180.0),
    "left_hip_pitch_joint": (120.0, 4.0, 180.0),
    "right_hip_pitch_joint": (120.0, 4.0, 180.0),
    "left_hip_roll_joint": (100.0, 3.3, 180.0),
    "right_hip_roll_joint": (100.0, 3.3, 180.0),
    "left_hip_yaw_joint": (100.0, 3.3, 180.0),
    "right_hip_yaw_joint": (100.0, 3.3, 180.0),
    "left_knee_pitch_joint": (150.0, 5.0, 180.0),
    "right_knee_pitch_joint": (150.0, 5.0, 180.0),
    "left_ankle_pitch_joint": (50.0, 2.5, 80.0),
    "right_ankle_pitch_joint": (50.0, 2.5, 80.0),
    "left_ankle_roll_joint": (50.0, 2.5, 80.0),
    "right_ankle_roll_joint": (50.0, 2.5, 80.0),
    "left_shoulder_pitch_joint": (40.0, 2.0, 27.0),
    "right_shoulder_pitch_joint": (40.0, 2.0, 27.0),
    "left_shoulder_roll_joint": (40.0, 2.0, 27.0),
    "right_shoulder_roll_joint": (40.0, 2.0, 27.0),
    "left_shoulder_yaw_joint": (40.0, 2.0, 27.0),
    "right_shoulder_yaw_joint": (40.0, 2.0, 27.0),
    "left_elbow_pitch_joint": (30.0, 1.5, 27.0),
    "right_elbow_pitch_joint": (30.0, 1.5, 27.0),
    "left_elbow_yaw_joint": (20.0, 1.0, 27.0),
    "right_elbow_yaw_joint": (20.0, 1.0, 27.0),
    "left_wrist_pitch_joint": (15.0, 1.0, 10.0),
    "right_wrist_pitch_joint": (15.0, 1.0, 10.0),
    "left_wrist_roll_joint": (15.0, 1.0, 10.0),
    "right_wrist_roll_joint": (15.0, 1.0, 10.0),
}

# Isaac X1_CFG init joint_pos == x1.xml keyframe (verified identical)
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


def parse_yaml_list(path, key):
    out, cur = [], False
    for line in Path(path).read_text().splitlines():
        if re.match(rf"^\s*{key}:\s*$", line):
            cur = True
            continue
        if cur:
            m = re.match(r"^\s*-\s*(\S+)\s*$", line)
            if m:
                out.append(m.group(1))
            elif line.strip() and not line.strip().startswith("#"):
                cur = False
    return out


def quat2mat(q):
    import mujoco
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.ascontiguousarray(q, dtype=np.float64))
    return out.reshape(3, 3)


def build_model(xml_path: Path):
    """Load x1.xml, add floor, enable foot collisions, disable the unused
    vendor <motor> actuators (we drive torques via qfrc_applied)."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(xml_path))
    # floor
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[20, 20, 0.1], friction=[1.0, 0.005, 0.0001],
                   contype=1, conaffinity=1, rgba=[0.35, 0.45, 0.5, 1])
    # enable collisions only on foot geoms (ankle_roll bodies)
    n_feet = 0
    for body in spec.bodies:
        if body.name.endswith("ankle_roll_link"):
            for geom in body.geoms:
                geom.contype = 1
                geom.conaffinity = 1
                n_feet += 1
    # drop vendor actuators (we drive torques via qfrc_applied)
    for act in list(spec.actuators):
        try:
            spec.delete(act)
        except Exception:
            act.forcerange = [0.0, 0.0]
            act.ctrlrange = [0.0, 0.0]
    model = spec.compile()
    return model, n_feet


def load_policy(ckpt_path: Path):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    layers, i = [], 1
    while f"{i}.weight" in actor_sd:
        w = actor_sd[f"{i}.weight"].float().numpy()
        b = actor_sd.get(f"{i}.bias")
        b = b.float().numpy() if b is not None else None
        layers.append((w, b))
        i += 1
    mean = sd["actor_obs_normalizer._mean"].float().numpy().squeeze(0)
    std = sd["actor_obs_normalizer._std"].float().numpy().squeeze(0)
    meta = {k: ck.get(k) for k in ("it", "infos")}
    return layers, mean, std, meta


def run_policy(layers, mean, std, obs):
    """rsl_rl MLP: Linear -> ELU (between layers), last layer linear, raw mean
    action out (no tanh — IsaacLab clips at clip_actions=100, effectively raw)."""
    x = (np.asarray(obs, dtype=np.float32) - mean) / std
    for w, b in layers[:-1]:
        x = x @ w.T + (b if b is not None else 0)
        x = np.where(x < 0, np.expm1(np.minimum(x, 20)), x)  # ELU(alpha=1)
    w, b = layers[-1]
    return x @ w.T + (b if b is not None else 0)


class SoftRenderer:
    """Pure-CPU stick-figure renderer (matplotlib Agg, no GL at all).

    v22 evidence: the container has NO usable GL stack — system PyOpenGL
    (isaac_sim site-packages), pip PyOpenGL, EGL and OSMesa all fail, so
    mujoco.Renderer can never initialize. This renderer draws the mj
    skeleton (body world positions + parent edges) with matplotlib 3D and
    feeds frames to imageio — guaranteed to work headless."""

    def __init__(self, model, height=480, width=840):
        import os as _os
        import matplotlib
        if not _os.environ.get("MPLCONFIGDIR"):
            # containers may have no writable home; font cache needs a home
            try:
                d = Path.cwd() / "mplcache"
                d.mkdir(exist_ok=True)
                _os.environ["MPLCONFIGDIR"] = str(d)
            except Exception:
                pass
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        self._plt = plt
        self.model = model
        try:
            self.parent = [int(model.body_parentid[i]) for i in range(model.nbody)]
        except Exception:
            self.parent = [int(model.body(i).parentid) for i in range(model.nbody)]
        self.foot_ids = [i for i in range(model.nbody)
                         if model.body(i).name.endswith("ankle_roll_link")]
        self.trail = []
        dpi = 100
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")

    def capture(self, data) -> np.ndarray:
        import numpy as _np
        ax, plt = self.ax, self._plt
        P = data.xpos.copy()
        root = P[1]  # body 1 = torso (child of world)
        self.trail.append(root.copy())
        if len(self.trail) > 100:
            self.trail.pop(0)
        ax.clear()
        for i in range(1, self.model.nbody):
            a, b = P[self.parent[i]], P[i]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="0.25", lw=1.8)
        for fi in self.foot_ids:
            f = P[fi]
            ax.scatter([f[0]], [f[1]], [f[2]], color="crimson", s=42)
        tr = _np.array(self.trail)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="steelblue", lw=1.2, alpha=0.8)
        ax.scatter([root[0]], [root[1]], [root[2]], color="navy", s=60)
        # camera follows root; fixed 3m box
        ax.set_xlim(root[0] - 1.5, root[0] + 1.5)
        ax.set_ylim(root[1] - 1.5, root[1] + 1.5)
        ax.set_zlim(0.0, 1.6)
        ax.view_init(elev=-18, azim=-75)
        ax.set_box_aspect((1, 1, 0.53))
        ax.set_axis_off()
        self.fig.canvas.draw()
        buf = _np.asarray(self.fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    def close(self):
        try:
            self._plt.close(self.fig)
        except Exception:
            pass


def main():
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--cmd", nargs=3, type=float, default=[1.0, 0.0, 0.0])
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--settle", type=float, default=0.5,
                    help="hold default pose before applying command (s)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--render", choices=["gl", "soft"], default="gl",
                    help="soft = matplotlib stick figure (no GL needed)")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    lab_dof = parse_yaml_list(root / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml",
                              "lab_dof_names")
    assert len(lab_dof) == 29

    model, n_feet = build_model(root / "gmr_x1_assets" / "x1.xml")
    data = mujoco.MjData(model)

    # joint indices (MuJoCo) and lab->mj mapping
    mj_names = [model.joint(i).name for i in range(model.njnt)]
    hinge = [n for n in mj_names if n in DEFAULT_Q]
    assert len(hinge) == 29, f"expected 29 hinges, got {len(hinge)}"
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])
    vadr = np.array([model.jnt_dofadr[jid[n]] for n in hinge])
    # policy column (lab order) -> mj hinge index
    lab2mj = np.array([hinge.index(n) for n in lab_dof])
    kp = np.array([PD[n][0] for n in hinge])
    kd = np.array([PD[n][1] for n in hinge])
    eff = np.array([PD[n][2] for n in hinge])
    q_default = np.array([DEFAULT_Q[n] for n in hinge])

    layers, mean, std, meta = load_policy(Path(args.ckpt))
    in_dim = layers[0][0].shape[1]
    print(f"[INFO] policy input dim = {in_dim} (expect {3*96}); feet colliders = {n_feet}")
    assert in_dim == 3 * 96, f"unexpected obs dim {in_dim}"

    # initial state: keyframe standing
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 0.62]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[qadr] = q_default
    mujoco.mj_forward(model, data)

    cmd = np.array(args.cmd, dtype=np.float64)
    substeps = max(1, round(CONTROL_DT / model.opt.timestep))
    n_steps = int(args.duration / CONTROL_DT)
    settle_steps = int(args.settle / CONTROL_DT)

    hist = np.zeros((HIST, 96), dtype=np.float32)
    last_act = np.zeros(29, dtype=np.float64)
    frames = []
    vxy_err, yaw_err, alive_steps = [], [], 0
    video = None
    soft = None
    if args.video:
        if args.render == "soft":
            soft = SoftRenderer(model, height=480, width=840)
        else:
            video = mujoco.Renderer(model, height=480, width=840)

    rng = np.random.default_rng(0)
    for step in range(n_steps):
        q = data.qpos[qadr]
        dq = data.qvel[vadr]
        R = quat2mat(data.qpos[3:7])
        ang_b = data.qvel[3:6].copy()                     # body frame (verified)
        grav_b = R.T @ np.array([0, 0, -1.0])
        v_w = data.qvel[0:3].copy()
        v_b = R.T @ v_w
        obs = np.concatenate([ang_b, grav_b, cmd,
                              (q - q_default)[lab2mj], dq[lab2mj], last_act]).astype(np.float32)
        hist[:-1] = hist[1:]
        hist[-1] = obs

        if step < settle_steps:
            act = np.zeros(29)
        else:
            act = run_policy(layers, mean, std, hist.reshape(-1))
        last_act = act
        # act[i] is for lab_dof[i]; target = default + 0.25*act; place into
        # MuJoCo hinge order via lab2mj (lab2mj[i] = hinge index of lab_dof[i])
        q_tgt_full = q_default.copy()
        q_tgt_full[lab2mj] = q_default[lab2mj] + ACTION_SCALE * act

        for _ in range(substeps):
            tau = kp * (q_tgt_full - data.qpos[qadr]) - kd * data.qvel[vadr]
            tau = np.clip(tau, -eff, eff)
            data.qfrc_applied[vadr] = tau
            mujoco.mj_step(model, data)

        # metrics + termination
        base_z = data.qpos[2]
        tilt = np.degrees(np.arccos(np.clip(quat2mat(data.qpos[3:7])[2, 2], -1, 1)))
        R = quat2mat(data.qpos[3:7])
        v_b = R.T @ data.qvel[0:3]
        if step >= settle_steps:
            alive_steps += 1
            vxy_err.append(np.linalg.norm(v_b[:2] - cmd[:2]))
            yaw_err.append(abs(data.qvel[5] - cmd[2]))
            if base_z < FALL_Z or tilt > FALL_TILT:
                print(f"[FALL] step {step} t={step*CONTROL_DT:.2f}s base_z={base_z:.3f} tilt={tilt:.1f}")
                break
        if video or soft:
            if soft is not None:
                frames.append(soft.capture(data))
            else:
                cam = video.camera
                lookat = data.qpos[0:3].copy()
                cam.lookat[:] = lookat + [0, 0, 0.1]
                cam.distance, cam.azimuth, cam.elevation = 3.2, 90.0, -12.0
                video.update_scene(data)
                frames.append(video.render())

    print(f"[INFO] survived {alive_steps} steps = {alive_steps*CONTROL_DT:.2f}s of "
          f"{n_steps - settle_steps} commanded steps")
    if vxy_err:
        print(f"[INFO] mean |v_xy error| = {np.mean(vxy_err):.3f} m/s, "
              f"mean |yaw error| = {np.mean(yaw_err):.3f} rad/s, "
              f"distance = {np.linalg.norm(data.qpos[:2]):.2f} m")

    if args.video and frames:
        import imageio.v2 as imageio
        imageio.mimwrite(args.video, frames, fps=int(1 / CONTROL_DT), quality=8)
        print(f"[VIDEO] {args.video} ({len(frames)} frames, "
              f"{Path(args.video).stat().st_size // 1024}KB)")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "ckpt": str(args.ckpt), "cmd": list(args.cmd),
            "duration_s": args.duration, "settle_s": args.settle,
            "survived_steps": alive_steps,
            "survived_s": alive_steps * CONTROL_DT,
            "mean_vxy_err": float(np.mean(vxy_err)) if vxy_err else None,
            "mean_yaw_err": float(np.mean(yaw_err)) if yaw_err else None,
            "distance_m": float(np.linalg.norm(data.qpos[:2])),
            "final_base_z": float(data.qpos[2]),
            "fell": bool(alive_steps < n_steps - settle_steps),
        }, indent=1))
    if soft:
        soft.close()
    if video:
        video.close()


if __name__ == "__main__":
    main()
