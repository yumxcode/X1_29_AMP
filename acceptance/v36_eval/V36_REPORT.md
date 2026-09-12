# v36 验收报告（TASK_20260912_111，2026-09-12 clean 末段微调）

**变更**（commit 1dd387b）：v35 model_3999 resume +600 iters，X1_ROBUST_TRAIN=0（v29e 配方）。
**结论：平台 P3 四项全过（12/13），但本地形式门回退——不满足 v36 accept 门，v35 仍为发布候选。**

## 核心结果（与 v35 对照）

| 维度 | v35（随机化训练终点） | v36（+600 clean） | 门限 | 判定 |
|---|---|---|---|---|
| 平台 P3a lin kernel | 0.7913 FAIL | **0.8640 PASS** | ≥0.82 | ✓ |
| 平台 P3b ang kernel | 0.4503 FAIL | **0.5925 PASS** | ≥0.5 | ✓ |
| 平台 P3c err_xy | 0.4765 FAIL | **0.3684 PASS** | ≤0.44 | ✓ |
| 平台 P3d err_yaw | 1.1243 FAIL | **0.7675 PASS** | ≤0.95 | ✓ |
| 平台判定 | 8/13 | **12/13**（仅 P6_play 环境契约 FAIL） | 13/13 | ✓ |
| 平台 style | 0.728 | **0.947**（T3 HIT） | ≥0.15 | ✓ |
| 本地 P7 形式门 | 8/8 | **8/8**（anti −1.00 / coupling +0.92） | 8/8 | ✓ |
| 本地 G1 稳定 | 5/5 | 5/5 | 不摔 | ✓ |
| 本地 G3 落地 | walk 全过（0 toe_first） | walk 全过（0 toe_first，≤1.9mm） | 全过 | ✓ |
| **本地 G2 walk05 膝对称** | **0.935 PASS** | **0.678 FAIL**（L 34.7 / R 24.6°） | ≥0.85 | **✗** |
| 本地 G2 walk10 | 0.894/0.991 PASS | 0.850 边界 FAIL（膝）+相位 0.444 | ≥0.85 | **✗** |
| walk10 横向漂移 | +0.29 m | **+1.83 m** | ≤1.0 | **✗** |
| 臂摆幅 walk05（joint°） | 27.3/29.2 | 36.3/37.1（回升 25%） | ref 92/83 | 改善 |
| 鲁棒性 | 55/60 | 见 v36_robustness_report.json | ≥48 | 待查 |

## 关键测量

- **Kernel 轨迹**（argo 完整日志解析，100-iter 滑窗）：P3 全部达标于 **iter ~4125（+125 clean）**，4200 后平台（lin 0.862-0.864 / ang 0.570-0.593）。clean 微调对 kernel 见效极快。
- **侵蚀同步发生**：model_4500（+501）walk10 膝 0.921 PASS 但 walk05 膝 0.708 FAIL——形式侵蚀与 kernel 攀升在相同时间尺度上发生，无"先达标再侵蚀"的可分离窗口。
- **漂移悖论**：训练期 yaw_bias 罚金 v36（−0.021/step）反而低于 v35（−0.035），但 MuJoCo 漂移 +1.83m——干净域重新涌现的不对称步态在 sim 差异下表现为漂移；奖励守卫在训练域内被"付掉了"而不是被消除。
- P7 仍 8/8（臂反相 −1.00、耦合 +0.92、腿对称 0.97）；term audit 干净（−0.012，lumP +15.0=ref）。

## 判定与去向

v36 accept 门（P3a≥0.82 ✓ + P7 8/8 ✓ + 鲁棒性≥48 待查 + **漂移≤1.0 ✗** + G2 ✗）→ **FAIL**。
发布候选维持 **v35**（本地 sim2sim 全绿）；v36 证明平台 P3 的可达性（12/13）。
**v37**（commit 3321c92）：联合调度——半强度扰动（push ±0.4 @6-12s + 1-step delay）下从 v35 +800 iters，两族门同时优化，直接针对本报告量化的反相关。

## 证据

- 平台：TASK_20260912_111（VERDICT 12/13；model_4000/4500/4598 已注册）
- `acceptance/evidence/v36_p7_gate.json`、`acceptance/evidence/v36_term_audit.txt`
- `acceptance/v36_eval/`：model_4598/4500 policy.npz、5 场景 rollout npz、v36_gait_report.json、v36_robustness_report.json
- kernel 轨迹解析：argo main.log（/tmp/v36_main.log，600 iters 全量）
