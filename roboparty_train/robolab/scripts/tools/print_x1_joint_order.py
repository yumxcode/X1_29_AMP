"""
Diagnostic script: print the actual Isaac Lab joint and body names for X1.
Run this on a machine with Isaac Lab installed to get the correct joint order
for retarget config (x1.yaml) and symmetry indices.

Usage:
    python scripts/tools/print_x1_joint_order.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print X1 joint and body names in Isaac Lab order.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg

from robolab.assets.robots.x1 import X1_CFG
from robolab.assets import ISAAC_DATA_DIR


@sim_utils.configclass
class DiagSceneCfg(InteractiveSceneCfg):
    def __init__(self, num_envs: int = 1, env_spacing: float = 2.5, **kwargs):
        super().__init__(num_envs=num_envs, env_spacing=env_spacing, **kwargs)
        self.ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        self.robot = X1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    scene_cfg = DiagSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]

    print("\n" + "=" * 60)
    print("X1 Isaac Lab Joint Information")
    print("=" * 60)

    joint_names = robot.data.joint_names
    body_names = robot.data.body_names

    print(f"\n--- Joint Names ({len(joint_names)}) ---")
    for i, name in enumerate(joint_names):
        print(f"  [{i:2d}] {name}")

    print(f"\n--- Body Names ({len(body_names)}) ---")
    for i, name in enumerate(body_names):
        print(f"  [{i:2d}] {name}")

    print("\n--- Default Joint Positions ---")
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    for i, (name, pos) in enumerate(zip(joint_names, default_pos)):
        print(f"  [{i:2d}] {name}: {pos:.5f}")

    print("\n--- Joint Limits ---")
    lower = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy()
    upper = robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
    for i, (name, lo, hi) in enumerate(zip(joint_names, lower, upper)):
        print(f"  [{i:2d}] {name}: [{lo:.4f}, {hi:.4f}]")

    print("\n--- YAML for retarget config (lab_dof_names) ---")
    print("lab_dof_names:")
    for name in joint_names:
        print(f"  - {name}")

    print("\n--- YAML for key body names ---")
    # Find feet and key bodies
    key_patterns = ["ankle_roll", "knee", "elbow_yaw"]
    print("lab_key_body_names:")
    for name in body_names:
        for pat in key_patterns:
            if pat in name:
                print(f"  - {name}")
                break

    print("\n" + "=" * 60)
    print("Copy the YAML above into scripts/tools/retarget/config/x1.yaml")
    print("=" * 60 + "\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
