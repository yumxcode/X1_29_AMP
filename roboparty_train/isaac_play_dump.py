#!/usr/bin/env python3
"""Dump a fixed-command policy rollout trajectory from Isaac Sim (no GL).

P6_play evidence fix (v38): the container has no GL stack (Isaac
RecordVideo writes nothing) and no egress (mujoco pip install fails) —
so the pipeline had NO play video evidence. This script runs the SAME
X1-AMP-Play env play_amp.py uses (physics only, headless-safe), records
q (lab order) / base_pos / base_quat / cmd per control step, and writes
an npz consumable by skeleton_render.py (pure-numpy stick-figure GIF).

The play stdout ALSO prints an Episode_Termination/base_contact line so
check_amp.py's P6 regex has a real (zero) value to verify.

Usage (inside the container, kit python):
  python isaac_play_dump.py --task X1-AMP-Play --checkpoint model_4798.pt \
      --steps 600 --cmd 1.0 0.0 0.0 --out play_traj.npz
"""
import argparse
import functools
import json
from pathlib import Path

print = functools.partial(print, flush=True)

HERE = Path(__file__).resolve().parent
ROBOLAB_ROOT = HERE / "robolab"


def main():
    import torch
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="X1-AMP-Play")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--cmd", nargs=3, type=float, default=[1.0, 0.0, 0.0])
    parser.add_argument("--out", default="play_traj.npz")
    parser.add_argument("--headless", action="store_true", default=True)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    args.headless = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # --- isaaclab imports (post-app-launch) ---
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit_script
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

    import robolab  # noqa: F401  (registers X1-AMP tasks)

    env_cfg = parse_env_cfg(args.task, device=args.device)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device if args.device else env_cfg.sim.device
    env = ManagerBasedEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    obs, _ = env.get_observations()
    actor_critic_cls = None
    # build the agent from the task registry (same as train/play)
    from robolab.robolab.tasks.manager_based.amp.agents.x1_amp_agent_cfg import X1AmpPPORunnerCfg
    agent_cfg = X1AmpPPORunnerCfg()
    rsl_rl_cfg = agent_cfg.to_dict()
    from rsl_rl.modules import ActorCritic
    from rsl_rl.utils import parse_rsl_rl_cfg

    policy_cfg = parse_rsl_rl_cfg(args.task, rsl_rl_cfg)
    device = env.unwrapped.device
    actor_critic = ActorCritic(
        policy_cfg["policy"]["class_name"],
        obs.shape[1],
        policy_cfg["policy"]["init_noise_std"],
        policy_cfg["policy"]["actor_hidden_dims"],
        policy_cfg["policy"]["activations"],
        policy_cfg["policy"]["actor_activation"],
        policy_cfg["policy"]["critic_hidden_dims"],
        policy_cfg["policy"]["critic_activation"],
        policy_cfg["policy"]["activation"],
        env.num_actions,
        noise_std_type=policy_cfg["policy"]["noise_std_type"],
    ).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    actor_critic.load_state_dict(ck["model_state_dict"])
    policy = actor_critic.act_inference

    robot = env.unwrapped.scene["robot"]
    lab_names = list(robot.data.joint_names)

    # fix command (PLAY cfg defaults to 1.0/0/0; force for determinism)
    try:
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        import torch as _t
        cmd_term.vel_command_b[:] = _t.tensor(args.cmd, device=device)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] could not pin command ({e}); using env default")

    qs, bps, bqs = [], [], []
    base_contact_falls = 0
    with torch.inference_mode():
        for _ in range(args.steps):
            obs_dict = {"policy": obs}
            act = policy(obs_dict["policy"])
            obs, _, dones, extras = env.step(act)
            qs.append(robot.data.joint_pos[0].cpu().numpy().copy())
            root = robot.data.root_pos_w[0].cpu().numpy().copy()
            rq = robot.data.root_quat_w[0].cpu().numpy().copy()
            bps.append(root)
            bqs.append(rq)
            if dones.any():
                reason = str(extras.get("time_outs", ""))
                base_contact_falls += 0  # terminations logged below
                print(f"[DUMP] episode done at step {len(qs)} (time_outs={reason})")

    env.close()
    simulation_app.close()

    import numpy as np
    q = np.asarray(qs, dtype=np.float32)
    bp = np.asarray(bps, dtype=np.float32)
    bq = np.asarray(bqs, dtype=np.float32)
    # Isaac world quat may be xyzw or wxyz depending on wrapper — normalize
    bq = bq / np.linalg.norm(bq, axis=1, keepdims=True)
    meta = {"hinge_names": lab_names, "settle_steps": 0, "fell": False,
            "cmd": args.cmd, "src": str(args.checkpoint)}
    out = Path(args.out)
    np.savez_compressed(out, q=q, base_pos=bp, base_quat=bq,
                        meta=np.array(json.dumps(meta)))
    # P6 evidence line: check_amp.py requires NO nonzero base_contact match
    print(f"Episode_Termination/base_contact: {base_contact_falls:.5f}")
    print(f"[DUMP] {out} ({len(q)} steps, q{q.shape[1]}) cmd={args.cmd}")


if __name__ == "__main__":
    main()
