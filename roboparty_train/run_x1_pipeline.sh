#!/bin/bash
# ============================================================
# Step 1: Verify Isaac Lab joint order (run first)
# ============================================================
# python robolab/scripts/tools/print_x1_joint_order.py --headless
# → Copy output to robolab/scripts/tools/retarget/config/x1.yaml

# ============================================================
# Step 2: Run GMR retargeting (SMPLX → X1)
# ============================================================
# Assumes GMR repo is cloned and installed at /workspace/GMR
# Assumes AMASS data is at /workspace/AMASS (mounted personal storage)

# --- Single test motion ---
# python GMR/scripts/smplx_to_robot.py \
#     --smplx_file /workspace/AMASS/BMLrub_stageii/rub001/0005_normal_walk1_stageii.npz \
#     --robot x1 \
#     --save_path /workspace/output/x1_gmr/normal_walk1.pkl \
#     --rate_limit \
#     --save_as_pkl

# --- Batch: BMLrub walking/jogging motions ---
# python GMR/scripts/smplx_to_robot_dataset.py \
#     --src_folder /workspace/AMASS/BMLrub_stageii/rub001 \
#     --tgt_folder /workspace/output/x1_gmr \
#     --robot x1 \
#     --save_as_pkl

# --- Batch: CMU key subjects ---
# python GMR/scripts/smplx_to_robot_dataset.py \
#     --src_folder /workspace/AMASS/CMU/127 \
#     --tgt_folder /workspace/output/x1_gmr \
#     --robot x1 \
#     --save_as_pkl

# ============================================================
# Step 3: Retarget GMR → Isaac Lab format
# ============================================================
# After updating x1.yaml with correct lab_dof_names:

# python robolab/scripts/tools/retarget/dataset_retarget.py \
#     --robot x1 \
#     --input_dir robolab/data/motions/x1_gmr \
#     --output_dir robolab/data/motions/x1_lab \
#     --config_file robolab/scripts/tools/retarget/config/x1.yaml \
#     --loop clamp --headless

# ============================================================
# Step 4: Train
# ============================================================
# Flat (no motion data needed):
# python robolab/scripts/rsl_rl/train.py --task=X1-Flat --headless --logger=tensorboard --num_envs=8192

# AMP (needs x1_lab motion data):
# python robolab/scripts/rsl_rl/train.py --task=X1-AMP --headless --logger=tensorboard --num_envs=8192
