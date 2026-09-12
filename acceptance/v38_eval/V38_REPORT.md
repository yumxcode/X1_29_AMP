# v38 验收报告（TASK_20260912_135，2026-09-12 判别器观测扩展）

**变更**（commits e1790b0/624c120/e6314f7）：disc obs 6→10 body（+L/R shoulder/wrist，v32 数据集 FK≤1.2mm）+ X1_ROBUST_TRAIN=3（push±0.5@5-10s）+ P6 骨架 GIF 链 + mirror dof 序 bug 修复。基座 v37 model_4798 +1500 iters（disc 维度 237→273 触发 re-init，属预期）。

## 结论：**v37 维持发布候选**。v38 平台 12/13 保持，但本地腿对称回退——判别器"看见了臂"（style 分与臂形质建立相关性的首个证据），但形式收益为负。

## 全门矩阵（v37 → v38, model_6000, -298 iter）

| 门 | v37 | v38 | 判定 |
|---|---|---|---|
| 平台判定 | 12/13 | **12/13** | 持平 |
| P3a/P3b kernel | 0.843/0.522 | **0.842/0.539** | ✓（ang +0.017） |
| P5a style | 0.796 | **0.562**（仍 HIT ≥0.5） | 新 disc 打分更严 |
| P7 形式门 | 8/8 | **7/8**（P7e 腿对称 0.70 边界败） | ✗ |
| G2 walk10 髋/膝 | 0.823/0.837 | **0.694/0.959** | 髋 ✗ 恶化、膝 ✓ 修复 |
| G2 walk05 髋/膝 | (膝 0.927 PASS) | 0.755 ✗ / 0.937 ✓ | 髋族全面回退 |
| walk10 漂移 | +0.94 m | **−1.47 m**（反向） | ✗ |
| 鲁棒性 | 48/60（push1.0 0/5） | **48/60**（push1.0 1/5, worst 0/5） | 持平 |
| 臂摆幅 joint° | 29.7/26.8 | **21.7/23.2** | ✗ 反而下降 |
| 臂反相/耦合 | −0.98/+0.79 | **−1.00/+0.95** | ✓ 更优 |
| P6_play | FAIL | FAIL（dump 崩，r3 修复待验） | 容器契约 |

## 判读

1. **判别器观测扩展的方向性证据**：style 0.796→0.562 与臂摆幅 29→22 同向变化——旧 disc 对臂"失明"时给高分；新 disc 看见 wrist/shoulder 后对当前策略的臂运动打分显著变低（demo 的 wrist 侧向摆幅 137-150mm vs 策略的塌缩摆动）。**style 分首次成为臂形质的有效信号**。
2. **但形式未受益反受损**：髋对称族全面回退（0.69-0.82），漂移反向 −1.47m。假设：disc 输入扩容后 style 梯度重新分配，30% style 权重下最易"取悦"disc 的方向不是放大臂摆（action_rate/joint_vel/arm 守卫惩罚大摆动），而是腿部风格细节的过拟合；1.0 m/s 下 L 侧髋摆幅 26° vs R 38° 的不对称是新吸引子。
3. kernel/ang 反而 +0.017（0.539），P3 家族对 disc 扩容不敏感。
4. **容器内两个工程修复已落地待验**：final-sweep 晚到 checkpoint 镜像（v38 的 model_6297 因竞态未注册，只拿到 6000）、isaac_play_dump r3（完全镜像 play_amp.py 结构 + dump 失败诊断到任务日志 + argparse --checkpoint 冲突修复）。

## v39 方向（若继续 disc-led 路线）

- style_reward_scale 1.5→2.5 或 task_style_lerp 0.7→0.65：放大已建立相关性的 style 通道
- 臂部专项：wrist demo 幅度大 vs 策略小——考虑给 style 加臂通道 dropout 课程，或先冻结 disc 只训 policy 数百 iter 观察臂响应
- 髋对称回退需对冲：paired_leg_deviation 守卫（镜像数据已对称，当前无腿部对称惩罚项）
- 每步都有代价：v37 是已验证的平衡点，v39 若不顺应回退到 v37 发布

## 证据

- 平台：TASK_20260912_135（12/13；model_5000/5500/6000 已注册；model_6297 因 monitor 竞态未注册——已修 final-sweep）
- `acceptance/v38_eval/`：model_6000 policy.npz + 5 场景 npz + gait/robustness 报告
- `acceptance/evidence/v38_p7_gate.json`（7/8）、term audit 在 V38 数据内
- `acceptance/videos_v31/v38_policy_skeleton.gif`（本地渲染验证链）
- kernel 轨迹：argo main.log 1500 iters（P3 于 ~5000 达标后稳定至 6297）
