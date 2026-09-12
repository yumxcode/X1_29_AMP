# v35 验收报告（TASK_20260912_075，2026-09-12）

**变更**（commit 8550211）：yaw DC-bias guard（`|EMA_0.005(ang_vel_z − cmd_z)|`，w=−0.5）+ task_style_lerp 0.6→0.7。
**基线对比**：v34（TASK_20260911_136，P7 ALL-PASS 首次 / drift +3.49 m / yaw 偏置 +4.7–5.0 deg/s）。

## 结论速览

| 维度 | v34 | v35 | 判定 |
|---|---|---|---|
| P7 步态形式门（8 项） | 8/8 PASS | **8/8 PASS** | 保持 ✓ |
| 横向漂移（walk10, 12s） | +3.49 m | **+0.29 m** | 12× 改善 ✓ |
| yaw DC 偏置 | +4.7–5.0 deg/s | **+0.05 deg/s** | ~100× 改善 ✓ |
| P7d 臂腿耦合 | +0.81 | **+0.93** | 更接近 ref（+0.8..0.99）✓ |
| G1/G2/G3（walk10/walk05/walkturn） | —（未存档） | **全 PASS** | 达 v29f 同级 ✓ |
| 鲁棒性矩阵（60 cells） | 未跑 | **55/60**（历史最好：v28d 49 / v29e4b 45 / v29f 43） | ✓ |
| 平台 P5a style | 1.109 | 0.728（≥0.3 revert 线） | lerp 0.7 代价可接受 ✓ |
| 平台判定 | 8/13（P3×4 + P6_play） | 8/13（同四项） | 无回归，未提升 |
| 臂摆幅（joint°） | 30.8 / 39.8 | 28.5 / 29.6 | 下降（~28% of ref 92/83）⚠ v36 目标 |

**v35 两大变更均达成设计目标且无形式回归**：yaw 守卫把漂移压回 v32b 水平（+0.29 vs +0.30 m）且 heading 漂移近零；lerp 0.7 未伤 P7（耦合反而升至 +0.93），style 0.728 远高于 revert 线 0.3。**无需回退。**

## 证据清单

- 平台：TASK_20260912_075（13 个模型已注册，model_3999 13:06:10）
- `acceptance/evidence/v35_p7_gate.json` — P7 8/8 PASS（anti −0.99 / elbow 37.5 / lumY 10.1 / coupling +0.93 / ratio 0.89 / lean asym 2.5 / shared 9.0；TARGET 双命中）
- `acceptance/evidence/v35_term_audit.txt` — 逐项审计：terms −0.015/−0.036 ≈ 参考水平（ref −0.024/−0.031），lumP +14.5 vs ref +14.9，无统计失真
- `acceptance/v35_eval/v35_gait_report.json` — G1 稳定 / G2 对称 / G3 落地全过（walk10：42 落地事件 0 toe_first 0 scuff，penetration ≤1.3mm；walk05 同级；walkturn G2 EXEMPT:turn 但对称比 0.966）
- `acceptance/v35_eval/v35_robustness_report.json` — 55/60：nominal/noise1x/noise2x/lat1/lat2/lag1/lag2/mass±15% 全 5/5；push0.75 4/5、push1.0 3/5、worst 3/5
- `acceptance/videos_v31/v35_policy_walk_1.0.mp4` — 12s 行走视频
- `acceptance/diag_drift.py`（已扩展 v35）输出：world y drift +0.29 m (+0.03 m/s)，final heading +1.1°，heading drift +0.05 deg/s，body-frame vx +0.87 / vy +0.01

## 平台门控 8/13 的既有缺口（v34 同款，非 v35 引入）

| 门 | v34 | v35 | 差距 |
|---|---|---|---|
| P3a lin kernel | 0.7917 | 0.7913 | 阈值 0.82（差 0.029） |
| P3b ang kernel | 0.4487 | 0.4503 | 阈值 0.5（差 0.050） |
| P3c err_xy | 0.4756 | 0.4765 | 阈值 0.44 |
| P3d err_yaw | 1.1311 | 1.1243 | 阈值 0.95 |
| P6_play | FAIL | FAIL | play 证据契约缺口（容器内无 MuJoCo rollout 日志 + log 缺 base_contact 行，已知） |

lerp 0.7 未提升 P3（0.7913 vs 0.7917）——tracking 卡在 0.79 平台不是 style/task 权重问题，疑似 kernel 型奖励的饱和结构（exp kernel 对小误差不敏感）或命令分布覆盖问题。

## v36 方向建议

1. **P3b ang kernel 是最大缺口**（0.45 vs 0.5，v35 略升）。err_yaw 1.12 主要来自转向跟踪；可考虑 ang kernel 改用更陡的 σ 或对 yaw cmd 区间加密重采样。
2. **臂摆幅回升**：v35 joint 28.5/29.6°（ref 92/83）。v34 曾达 ~45%。候选：把 ref 幅幅分布作为 style 侧引导（AMP 辨别器主导步态的路线），或在 P7 上加幅度下限 TARGET。
3. P6_play：平台端修 play 证据契约（MuJoCo rollout 日志 + base_contact 指标行）——纯工程项，不阻塞本地结论。
