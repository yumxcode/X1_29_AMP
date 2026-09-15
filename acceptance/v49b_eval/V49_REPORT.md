# v49 验收报告（v49a TASK_20260915_205 + v49b TASK_20260916_002，2026-09-15/16）

## v49a：v48 守卫配方回归 regime 3（审核项 3 的直接尝试）

平台 12/13（P3 全过 0.838/0.546——kernel 回 regime 3 域 ✓；仅 P4 89% 窄败）。本地：walk05 比值实际双过
（0.973/0.934，VERDICT FAIL 为步态分割计数假象）+ walk10 PASS，但 P7g 16.4 ✗ / 漂移 -1.77 ✗ / back05
0.848 边界 + 1 toe_first / 鲁棒 44/60。**被现有冠军支配，弃用**（第三次打地鼠确认：guard 剂量 × regime
的二维网格无单点全过）。

## v49b：disc 幅度通道（wrist 速度观测——审核项 1/2 路线的直接验证）

**变更**（commits 403f1d3+16375b6）：disc 观测新增 key_body_VELOCITY（10 body × 3/steps）：
- ref 侧：motion_data_manager 有限差分派生（无需数据集重生成）+ interpolation + get_motion_state
- policy 侧：EMA 平滑有限差分（clamp ±10 m/s）
- demo 侧：ref_key_body_vel_b obs + AnimationTerm getter + DiscriminatorDemoCfg term（r1 崩于此缺口）
- X1_DISC_VEL A/B 开关；disc re-init 走已验证路径（style 瞬态正常：0→0.89→0.84 稳定）

**结果**（model_8394，+1000 iters from v45 disc-intact）：

| 轴 | v49b | 判读 |
|---|---|---|
| **主读出：臂摆幅** | **22.2/23.4°（walk10）/ 22.7/25.5°（walk05）** | **未达 35° 线**——vel 通道未显著超越位置通道（v47 24.6/26.0） |
| 平台 | 12/13（P3 全过 0.833/0.537；P4 84% 尾落） | disc 重置 + 1000 iter 的尾落 |
| P7 | **8/8**（耦合 +0.90 HIT） | 形式保持 |
| G3 | 全场景 0 toe_first | ✓ |
| 漂移 | **-0.12 / -0.43 m 双优** | 历代最佳之一 |
| G2 | walk10 髋 0.791 / walk05 膝 0.835 / back05 0.761 | ✗（对称回退） |
| 鲁棒 | 49/60（lat2 4/5 lag2 4/5 push0.75 4/5） | 中游 |

**判读**：速度通道方向正确但**单通道剂量不足**——幅度信息现在可见（corr -1.00 保持、漂移最优证明形式
没有崩），但 disc 梯度仍被动作平滑惩罚抵消（v40 已证平滑是主抑制项）。臂摆幅 22-25° 平台在三代
（v47/v49a/v49b）稳定出现 = **当前架构的幅度平衡点**；突破 35° 需 vel 通道 × 平滑 1/8 或 ref 侧幅度
加权（下一步杠杆，均已就绪）。

## 终版发布矩阵 v4（v36-v49b 十七代后定稿）

| 用途 | 发布点 | 凭证 |
|---|---|---|
| 平台门冠军 | **v47 model_7794** | 原生 13/13 + T1/T2 双 HIT + P7 8/8 + 双 walk G2 + 臂 24.6/26.0 |
| 本地 sim2sim 冠军 | **soup_a50** | G 全绿 + 漂移 0.01 + 鲁棒 51 + 直接计量 8/8 |
| 鲁棒性记录 | v48 model_8194 | 52/60 + 漂移/back05 解决配方（yaw -1.0 + 髋膝 -1.0） |
| disc 幅度路线验证 | v49b model_8394 | vel 通道端到端打通 + 12/13 + P7 8/8 + 漂移最优（22-25° 平台确认） |

## 证据

- v49a：TASK_20260915_205（12/13 归档）；v49b：TASK_20260916_002（12/13，r1 维度缺口修复归档）
- `acceptance/v49a_eval/`、`acceptance/v49b_eval/`：5 场景 npz + gait/robustness 报告
- `acceptance/evidence/v49a_platform_verdict.txt`、`v49{a,b}_p7_gate.json`、`v49b_term_audit`（term audit 数据）
