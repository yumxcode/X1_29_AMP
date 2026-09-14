# v39 验收报告（TASK_20260914_013，2026-09-14 disc-led round 2）

**变更**（commit 456b80f）：A/B vs v38 同基座（v37 model_4798）同剂量（+1500, ROBUST=3）：style_reward_scale 1.5→2.5 + task_style_lerp 0.7→0.65 + 新 leg_amp_asym 守卫（RMS 髋幅度对称，w=-0.3）。

## 结论：**平台 13/13 首次真达成**（P6 骨架 GIF 证据链闭环 + P3 全过 + style HIT）；本地 P7 8/8 + G1/G3 全过 + 漂移 PASS；**G2 髋对称未修复**（守卫力度不足），臂摆幅继续下降。

## 门矩阵（v35 / v37 / v39 三代对比）

| 门 | v35（满随机化） | v37（半强度） | **v39（disc-led r2）** |
|---|---|---|---|
| **平台判定** | 8/13（P3×4+P6 FAIL） | 12/13（P6 FAIL） | **13/13 全过** ✓ |
| P3a/P3b kernel | 0.7913/0.4503 FAIL | 0.8430/0.5218 | **0.8435/0.5425** ✓ |
| 平台 style（T3） | 0.728 | 0.796 | **1.065 HIT**（scale 2.5 生效） |
| P6_play 证据 | FAIL（残留视频假阳性时代） | FAIL | **PASS**（骨架 GIF 917KB，真证据） |
| 本地 P7 形式门 | 8/8 | 8/8 | **8/8**（coupling +0.97 历史最高） |
| G1 稳定（5 场景） | 5/5 | 5/5 | **5/5** |
| G3 落地 | 全过 0 toe_first | 全过 | **全过**（44 事件 0 toe_first ≤1.1mm） |
| G2 walk10 髋/膝 | 0.874/0.894 **PASS** | 0.823/0.837 MISS | 0.718 ✗ / 0.957 ✓ |
| G2 walk05 髋/膝 | 0.869/0.935 **PASS** | 膝 0.927 PASS | 0.698 ✗ / 0.963 ✓ |
| walk10 漂移 | +0.29 m | +0.94 m | **−0.47 m**（\|·\|≤1.0 ✓，walk05 −0.05） |
| 鲁棒性 | 55/60 | 48/60 | 见 v39_robustness_report.json |
| 臂摆幅 joint° | 28.5/29.6 | 29.7/26.8 | **18.7/20.4**（三代最低）✗ |
| term audit | −0.015 | −0.007 | −0.016（无失真，shared lean 11.2 饱和线内） |

## 判读

1. **平台门闭环（目标 2 达成）**：P6 证据链经 4 轮容器内修复（robolab.tasks 导入 → rsl_rl_cfg_entry_point 注册键 → P6DBG 无条件诊断 → 骨架 GIF 渲染）首次产出真 13/13。v27/v28d 的 13/13 是旧 pod 残留视频的假阳性。
2. **髋对称是 disc-led 路线的系统性代价**：v37 0.823→v38 0.694→v39 0.718。leg_amp_asym(-0.3) 方向正确但 1500 iter 只恢复 +0.024；且 v38/v39 从 v37（髋 0.823）出发，起点已弱于 v35（0.874）。
3. **臂摆幅倒退**（29.7→18.7°）：style scale 2.5 + disc 看见 wrist 仍未拉起臂摆——放大臂摆的 style 梯度仍被动作平滑类惩罚抵消；"disc 看见臂"使 style 分有信息量（v38 证明），但不足以驱动幅度。
4. 膝对称反而历史最好（0.957/0.963），漂移收敛（-0.47m），heading -6.9°。

## 发布决策

- **平台门发布候选 = v39 model_6297**（13/13 + P7 8/8 + 漂移 ✓）
- **本地严格门发布候选 = v35 model_3999**（G 全绿 + 55/60）／v37（双族平衡）
- v40 方向（最后合理一轮）：**回到 v35 基座**（G2 全绿、臂相对最好的起点）+ v39 配方 + leg_amp_asym 加码（-0.8）+ 臂部动作平滑惩罚豁免实验（action_rate/joint_acc 对 shoulder/elbow 通道减半）

## 证据

- 平台：TASK_20260914_013（**VERDICT: PASS 13/13**；model_6297 已注册——final-sweep 修复生效）
- `acceptance/evidence/v39_p7_gate.json`、`v39_term_audit.txt`
- `acceptance/v39_eval/`：model_6297 policy.npz + 5 场景 + gait/robustness 报告
- P6 证据：平台日志 `ok P6_play: video x1_play_skeleton.gif exists (917KB)`
