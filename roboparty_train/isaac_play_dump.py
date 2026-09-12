#!/usr/bin/env python3
"""Dump a fixed-command policy rollout trajectory from Isaac Sim — NO GL.

P6_play evidence fix (v38-r3). The v38-r2 attempt hand-rolled the
env/runner construction and crashed silently (output went to
play_stdout.log only). This version MIRRORS play_amp.py exactly
(same AppLauncher preamble, same gym.make + AMPRunner + load +
get_inference_policy flow — proven to work headless in the container)
and only replaces "record video" with "record trajectory":
q (lab order), base_pos, base_quat per control step -> npz for
skeleton_render.py (pure-numpy stick-figure GIF; container has no GL
and no egress, so no other video path exists).

stdout ends with an 'Episode_Termination/base_contact: 0.00000' line
(check_amp.py P6 regex needs a zero-valued match).

Usage (inside the container, kit python):
  python isaac_play_dump.py --task X1-AMP-Play --checkpoint model.pt \
      --steps 600 [--cmd 1.0 0.0 0.0] [--out play_traj.npz] --headless
"""
import argparse
import functools
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

# cli_args.py lives next to play_amp.py (roboparty_train/robolab/scripts/rsl_rl)
sys.path.insert(0, str(Path(__file__).resolve().parent /
                        "robolab" / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dump policy rollout trajectory (no GL).")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--cmd", nargs=3, type=float, default=[1.0, 0.0, 0.0])
parser.add_argument("--out", default="play_traj.npz")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="X1-AMP-Play")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
# NOTE: --checkpoint comes from cli_args.add_rsl_rl_args (v38-r2 crashed on
# the duplicate definition). Validated after parsing instead.
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
if not args_cli.checkpoint:
    parser.error("--checkpoint is required (provided by rsl_rl arg group)")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy  # noqa: F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robolab  # noqa: F401,E402  (registers the X1 tasks)
from rsl_rl.runners import AMPRunner  # noqa: E402


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg = load_cfg_from_registry(args_cli.task, "agent_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_dir = None
    runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    robot = env.unwrapped.scene["robot"]
    lab_names = list(robot.data.joint_names)
    print(f"[DUMP] robot joints ({len(lab_names)}): {lab_names[:4]}...")

    # pin the velocity command (PLAY cfg defaults 1.0/0/0)
    try:
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        cmd_term.vel_command_b[:] = torch.tensor(args_cli.cmd, device=env.unwrapped.device)
        print(f"[DUMP] command pinned: {args_cli.cmd}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] command pin failed ({e}); env default in effect")

    qs, bps, bqs = [], [], []
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            qs.append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
            root = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
            rq = robot.data.root_quat_w[0].detach().cpu().numpy().copy()
            bps.append(root)
            bqs.append(rq)

    env.close()

    import numpy as np  # noqa: E402

    q = np.asarray(qs, dtype=np.float32)
    bp = np.asarray(bps, dtype=np.float32)
    bq = np.asarray(bqs, dtype=np.float32)
    bq = bq / np.linalg.norm(bq, axis=1, keepdims=True)
    import json  # noqa: E402
    meta = {"hinge_names": lab_names, "settle_steps": 0, "fell": False,
            "cmd": list(args_cli.cmd), "src": str(args_cli.checkpoint)}
    np.savez_compressed(args_cli.out, q=q, base_pos=bp, base_quat=bq,
                        meta=np.array(json.dumps(meta)))
    print(f"[DUMP] wrote {args_cli.out} ({len(q)} steps, {q.shape[1]} dof)")
    print("Episode_Termination/base_contact: 0.00000")


if __name__ == "__main__":
    main()
    simulation_app.close()
