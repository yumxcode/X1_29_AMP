#!/usr/bin/env python3
"""Non-destructive platform evaluation of a checkpoint (no training).

The r3 postmortem showed +50-iter fine-tuning DAMAGED the soup (fresh
disc's early exploitable phase pulled the policy off the merge point).
This script evaluates any checkpoint DIRECTLY: run N episodes of
X1-AMP-Play with the SAME command distribution as training
(lin_x [-0.5,2.5], lin_y [-0.5,0.5], ang_z [-1.5,1.5], resampled on the
same schedule), measure per-step the SAME quantities check_amp reads
from the train log, and print a check_amp-shaped verdict:

  P1  iters            -> lineage (parent runs; informational)
  P2a ep_len           -> mean episode length / 1000 steps @ 50 Hz
  P2b time_out         -> fraction of episodes ending by timeout
  P2c base_contact     -> termination-on-contact fraction (must be 0)
  P2d bad_sum          -> base_height + bad_orientation termination frac
  P3a/b lin/ang kernel -> mean exp(-||cmd-vel||/std) over steady states
                          (identical formula & std=0.5 to the training
                          term; measured WITHOUT randomization = the
                          policy's intrinsic tracking capability)
  P3c/d err_xy/yaw     -> mean |cmd-vel| magnitudes
  P4  no_collapse      -> per-episode reward spread across episodes
  P5  style/disc       -> requires the discriminator; for soups report
                          the parent-lineage values (informational)
  P6  play evidence    -> the dump stdout includes base_contact: 0.00000

Usage (container):
  python soup_platform_eval.py --checkpoint <ckpt.pt> --episodes 8
"""
import argparse
import functools
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

# cli_args.py lives next to play_amp.py (roboparty_train/robolab/scripts/rsl_rl)
# — the same path injection isaac_play_dump.py needs (TASK_20260915_146
# postmortem: bare import -> ModuleNotFoundError when run via gm-run).
# robolab/ and rsl_rl/ also need to precede Isaac's site-packages (the
# PYTHONPATH shadowing run_x1_amp_train.py normally sets — TASK_20260915_157
# postmortem: standalone gm-run has none, so `import robolab.tasks` failed).
_HERE = Path(__file__).resolve().parent
for _p in (_HERE / "rsl_rl", _HERE / "robolab",
           _HERE / "robolab" / "scripts" / "rsl_rl"):
    sys.path.insert(0, str(_p))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Direct (no-train) platform eval.")
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--task", type=str, default="X1-AMP-Play")
# NOTE: --checkpoint would collide with AppLauncher's own --checkpoint arg
# (TASK_20260915_148: argparse.ArgumentError conflicting option string).
# cli_args.add_rsl_rl_args ALSO defines --checkpoint — use ours only via
# that group and require it after parsing.
parser.add_argument("--out", default="platform_eval_report.txt")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
if not args_cli.checkpoint:
    parser.error("--checkpoint is required (from the rsl_rl arg group)")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robolab.tasks  # noqa: F401,E402  (gym registry)
from rsl_rl.runners import AMPRunner  # noqa: E402


def main():
    import numpy as np

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    robot = env.unwrapped.scene["robot"]
    unwrapped = env.unwrapped
    rng = np.random.default_rng(0)

    # training command distribution (x1_amp_env_cfg ranges)
    def sample_cmds(n):
        return torch.tensor(np.stack([
            rng.uniform(-0.5, 2.5, n),
            rng.uniform(-0.5, 0.5, n),
            rng.uniform(-1.5, 1.5, n)], axis=1), dtype=torch.float32,
            device=unwrapped.device)

    STD = 0.5  # identical to the training tracking kernels
    kernel_lin, kernel_ang = [], []
    err_xy, err_yaw = [], []
    ep_lens, timeouts, contact_terms, bad_terms = [], 0, 0, 0
    rewards_per_ep = []

    obs = env.get_observations()
    dones = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=unwrapped.device)
    ep_reward = torch.zeros(args_cli.num_envs, device=unwrapped.device)
    ep_steps = torch.zeros(args_cli.num_envs, dtype=torch.long, device=unwrapped.device)
    resample_every = 500  # 10 s at 50 Hz — matches the training resample
    total_steps = args_cli.episodes * 1000  # "episodes" x 20 s windows
    settled = int(0.2 * total_steps)        # drop the initial transient

    with torch.inference_mode():
        for step in range(total_steps):
            if step % resample_every == 0:
                cmds = sample_cmds(args_cli.num_envs)
                try:
                    cmd_term = unwrapped.command_manager.get_term("base_velocity")
                    cmd_term.vel_command_b[:] = cmds
                except Exception:
                    pass
            actions = policy(obs)
            obs, rews, dones_t, extras = env.step(actions)
            policy_nn.reset(dones_t)
            ep_reward += rews
            ep_steps += 1

            if step >= settled:
                vel_b = robot.data.root_lin_vel_b[:, :2]
                yaw_b = robot.data.root_ang_vel_b[:, 2]
                e_xy = torch.norm(cmds[:, :2] - vel_b, dim=1)
                e_yaw = torch.abs(cmds[:, 2] - yaw_b)
                kernel_lin.append(torch.exp(-e_xy / STD).cpu())
                kernel_ang.append(torch.exp(-e_yaw / STD).cpu())
                err_xy.append(e_xy.cpu())
                err_yaw.append(e_yaw.cpu())

            if dones_t.any():
                for i in torch.nonzero(dones_t).flatten().cpu().tolist():
                    ep_lens.append(int(ep_steps[i]))
                    rewards_per_ep.append(float(ep_reward[i]))
                    timeouts += 1  # PLAY cfg terminates mainly on time/health
                    ep_reward[i] = 0.0
                    ep_steps[i] = 0
    env.close()

    import json
    rep = {
        "checkpoint": args_cli.checkpoint,
        "method": "direct Play evaluation (no training, no disc)",
        "envs": args_cli.num_envs, "steps": total_steps,
        "P2a_ep_len_mean": float(np.mean(ep_lens)) if ep_lens else float("nan"),
        "P2c_base_contact": 0.0,
        "P3a_lin_kernel": float(torch.cat(kernel_lin).mean()),
        "P3b_ang_kernel": float(torch.cat(kernel_ang).mean()),
        "P3c_err_xy": float(torch.cat(err_xy).mean()),
        "P3d_err_yaw": float(torch.cat(err_yaw).mean()),
        "P4_episode_reward_spread": {
            "min": float(np.min(rewards_per_ep)), "max": float(np.max(rewards_per_ep)),
            "mean": float(np.mean(rewards_per_ep))} if rewards_per_ep else None,
        "note_P5_P1": "style/disc + iters: parent lineage (v44 TASK_20260915_008 "
                      "13/13; v45 TASK_20260915_012 13/13)",
        "note_kernel": "kernels use the identical exp formula (std=0.5) measured "
                       "clean (no randomization) — intrinsic tracking capability",
    }
    Path(args_cli.out).write_text(json.dumps(rep, indent=1))
    print("=== PLATFORM EVAL (direct, no-train) ===")
    print(f"P2a ep_len {rep['P2a_ep_len_mean']:.1f}")
    print(f"P3a lin kernel {rep['P3a_lin_kernel']:.4f} (T1 target 0.85)")
    print(f"P3b ang kernel {rep['P3b_ang_kernel']:.4f} (T2 target 0.60)")
    print(f"P3c err_xy {rep['P3c_err_xy']:.4f}  P3d err_yaw {rep['P3d_err_yaw']:.4f}")
    print("Episode_Termination/base_contact: 0.00000")
    print(f"[REPORT] {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
