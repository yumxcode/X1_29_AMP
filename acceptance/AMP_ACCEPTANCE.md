# X1 AMP 训练结果验收标准

版本: v1.0 (2026-08-17)
适用对象: 任务 `X1-AMP`（robolab manager_based/amp，rsl_rl AMPRunner/PPOAMP）的训练产出。
执行器: `acceptance/check_amp.py`（解析训练 stdout 日志 → JSON 报告 + PASS/FAIL 判定）。

## 0. 指标语义（全部读码确认，勿凭直觉）

来源: `x1_amp_env_cfg.py`、`amp_env_cfg.py`、`rsl_rl/utils/amp_logger.py`、
`isaaclab_tasks .../velocity/mdp/commands.py`。

| 日志指标 | 语义 | 换算 |
|---|---|---|
| `Mean episode length` | 每 episode 步数（上限 = 20 s × 50 Hz = **1000**） | ≥950 ⇒ 95% 时长存活 |
| `Episode_Termination/time_out`` | episode 以超时结束的占比 | 高 = 不跌倒 |
| `Episode_Termination/base_contact` | 非脚部碰撞终止占比 | 0 = 不跌倒 |
| `Episode_Termination/base_height` / `bad_orientation` | 过低/过倾终止占比 | 低 = 姿态稳定 |
| `Episode_Reward/track_lin_vel_xy_exp` | **整 episode 逐帧 reward×权重 的均值**（amp_logger 除以 max_episode_length） | 线速度跟踪核 = 该值 / 权重 **1.25**；核=exp(−e²/σ²), σ=0.5 |
| `Episode_Reward/track_ang_vel_z_exp` | 同上，角速度 | 角速度跟踪核 = / 1.25 |
| `Metrics/.../error_vel_xy` | IsaacLab 命令指标：逐帧 ‖Δv‖ 累计 ÷ (10 s/0.02 s) | 满 episode(1000 帧) ≈ **2 × 平均速度误差**(m/s) |
| `Mean AMP style reward` | LSGAN style 奖励 episode 总和，上限 = 1000×0.02×1.5 = **30** | v16/v17 实测 ≈0.33（判别器近饱和）|
| `Mean reward` | 任务奖励 episode 均值×权重（含惩罚项） | 收敛与稳定性参考 |

命令分布: lin_vel_x ∈ U(−0.5, 2.5), lin_vel_y ∈ U(−0.5, 0.5), ang_vel_z ∈ U(−1.5, 1.5)
（heading 模式），10 s 重采样；2% 站立 env；push 扰动每 5–10 s 一次（±0.5 m/s）。
**跟踪指标在含扰动条件下测得**，阈值已按此校准。

## 1. 硬性验收（全部满足 = PASS）

评估窗口：最后 100 个 iteration 的均值（除非另注）。

| # | 指标 | 阈值 | 依据 |
|---|------|------|------|
| P1 | 训练完整性 | 达到配置 max_iterations（rsl_rl 迭代号 0-indexed：4000 次 = 最后日志行 "iteration 3999"，v20 曾因 off-by-one 误判）；最终 checkpoint 文件存在 | 流程完备 |
| P2a | Mean episode length | ≥ 950 / 1000 | 不跌倒 |
| P2b | time_out 占比 | ≥ 0.95 | 不跌倒 |
| P2c | base_contact 占比 | = 0.000 | 不跌倒（非脚部触地）|
| P2d | base_height + bad_orientation | ≤ 0.03（合计） | 姿态稳定 |
| P3a | 线速度跟踪核 | ≥ 0.82（= reward 1.025） | v16/v17 两次复现基线 0.835+0.005 余量 |
| P3b | 角速度跟踪核 | ≥ 0.50（= reward 0.625） | v16/v17 基线 0.54 |
| P3c | error_vel_xy | ≤ 0.44（≈平均误差 0.22 m/s） | 与 P3a 一致性互证 |
| P3d | error_vel_yaw | ≤ 0.95 | 同上 |
| P4 | 无后期崩溃 | 末 100 iter mean_reward ≥ 0.9 × 全程最佳 rolling(100) 均值 | 收敛性 |
| P5a | AMP style reward | ≥ 0.15 | 判别器未完全压死风格信号（基线 0.33）|
| P5b | disc_loss | ≤ 0.05 且有限 | 判别器数值稳定 |
| P6 | 固定指令行走验证 | X1-AMP-Play, 指令 (1.0, 0, 0)：≥10 s 视频且 >100KB，期间无 base_contact 终止；视频文件产出。视频来源允许 Isaac RecordVideo 或 MuJoCo sim2sim 兜底渲染（v20 中 Isaac RecordVideo 未产出 mp4，play exit=0）| 行走证据（可视化验收）|

## 2. 目标线（TARGET，非阻断；达标即宣告"85% 速度跟踪"达成）

| # | 指标 | 目标 | 说明 |
|---|------|------|------|
| T1 | 线速度跟踪核 | ≥ 0.85（= reward 1.0625） | 对应平均速度误差 ≤ 0.19 m/s；用户目标"速度跟踪 85%" |
| T2 | 角速度跟踪核 | ≥ 0.60 | 转向质量 |
| T3 | AMP style reward | ≥ 0.5（上限 30 的 1.7%）| 风格自然度改善 |

v18 相对 v16/v17 的两处改动（用于逼近 TARGET）: `max_iterations` 3000→4000；
`task_style_lerp` 0.6→0.75（style 已塌缩至 ~1%，把有效梯度让给任务项）。
若 v18 仍只过 P3a 不过 T1，如实报告差距（预计 0.83–0.84），不粉饰。

## 3. 执行与证据链

**验证策略：所有验收一律在 Gradmotion 训练容器内执行（Linux/GPU）；本地 Mac 仅做代码静态检查，不跑任何验证。**

1. **容器内**（v18 pipeline 自动执行）: 训练 stdout 全量落盘 → check_amp.py →
   `amp_acceptance_report.json` 随产物上传；play 视频存 `logs/x1_amp/`。
2. **sim2sim**（独立轻量任务）: 挂载最终 checkpoint → MuJoCo (x1.xml) rollout →
   行走视频 + 存活/跟踪指标（`sim2sim/run_sim2sim_task.py`）。
3. **平台侧复核**: `gm task data get` 拉训练曲线，与容器内报告交叉核对（防日志裁剪）。
4. 判定输出: PASS/FAIL 逐项清单 + TARGET 达成表。

## 决策记录（阈值与超参的实证依据）

- **v20（2026-08-17，TASK_20260817_083，lerp 0.75 / 4000 iter）实测**：P2/P3/P4 全过
  （lin kernel 0.8422、ang 0.5540、ep_len 991.7、timeout 0.984、无后期崩溃）；
  P5a style 0.146 < 0.15（FAIL，边缘）；P1 因迭代号 0-indexed 语义误判（已修）；
  P6 因 Isaac RecordVideo 未产出 mp4 而无证据（v21 增加 MuJoCo 兜底）。
- **task_style_lerp 终选 0.6**：lerp 0.75 相对 0.6 的 lin kernel 收益 +0.003（噪声级），
  而 style 0.335→0.146 腰斩跌破 P5a。0.6 在 P3a/P5a 双侧均有大余量（0.839/0.335），
  4000 iter 预计 lin ~0.84。
- **TARGET（用户 85% 跟踪）现状**：v16 0.8391、v20 0.8422，两次独立配置均未达 0.85；
  属平台期结构性差距，验收以 P3a=0.82 为准并如实报告 TARGET 差距。
- **checkpoint 上传机制（v16-v20 三连败根因，v21 修复）**：SDK 仅注册**仓库工作树外**
  的 .pt（唯一实证=gvhmr_pt 垃圾文件成功注册 4 次）；`logs/` 被 .gitignore 挡住、
  `model_upload/` 无 exported_data 模式从未注册。v21 起主镜像路径 =
  `/workspace/isaaclab/x1_upload/{tag}/`（树外），报告/视频/导出模型同路径镜像。
