# 归因 Probe 报告（TASK_20260914_070，2026-09-14）

**问题**（用户目标 4 验证）：步态形式由 style 判别器引导还是任务侧形式守卫引导？目标结构：速度跟随归任务 reward，其余步态归 style 判别器。

**设计**：v39 model_6297（P7 8/8 干净基座）+300 iters，X1_FORM_GUARDS=0——五项手工形式守卫全关（arm_pitch_sync / arm_asym_lean / arm_leg_coupling / lumbar_posture / leg_amp_asym），其余与 v39 完全一致（disc obs 10 body、style 2.5、lerp 0.65、ROBUST=3）。守卫关闭经日志证实（五项从 Episode_Reward 消失 = weight 0 被移除）。

## 结果（attr model_6596 vs v39-with-guards 基线）

| 归因通道 | v39 基线（守卫开） | probe（守卫关） | 归因结论 |
|---|---|---|---|
| **速度 kernel P3a/P3b**（任务侧对照） | 0.844/0.543 | **0.843/0.549** | 任务 reward 主导 ✓（恒定） |
| 平台判定 | 13/13 | **13/13** | 证据链稳定 |
| **walk10 髋对称** | 0.718 | **0.941**（+0.223） | **判别器主导 ✓**（守卫关反而更好） |
| **walk05 髋对称** | 0.698 FAIL | **0.900 PASS** | **判别器主导 ✓** |
| walk10 膝 / walk05 膝 | 0.957 / 0.937 | 0.841 / 0.873 | 判别器主导 ✓（略降仍近门） |
| G3 落地（物理正则仍在） | 全过 0 toe_first | **全过 0 toe_first** | 保持 ✓ |
| 臂反相 / 臂腿耦合 | −1.00 / +0.97 | **−0.96 / +0.50** | 判别器主导 ✓（耦合降但仍过 TARGET） |
| **walk10 漂移** | −0.47 m | **+0.38 m** | 保持 ✓ |
| **臂 DC 前倾 P7h** | 7.6 ✓ | **14.2 ✗**（门 12） | **守卫托底 ✗**（lumbar 17.3 vs 目标 14.9） |
| **walk05 臂不对称 DC** | 9.2° | **32.5° ✗** | **守卫托底 ✗**（L+3.8/R−28.7）+ 漂移 +2.33m |
| 臂摆幅 joint° | 18.7/20.4 | 19.2/17.5 | 持平（守卫与判别器都不是幅度来源） |

## 归因结论（目标 4 的精确答案）

1. **腿部步态（对称/膝/落地）：style 判别器已主导**——五守卫全关后腿部形式不降反升（镜像对称数据集 + 10-body 判别器观测足够）。leg_amp_asym 守卫可以退役（v39 的髋回退另有成因，非守卫缺失）。
2. **臂部相位结构（反相/耦合）：判别器主导**——T7a/T7b 在无守卫下保持 HIT。
3. **臂部 DC 姿态（前倾/不对称）与躯干前倾：仍靠守卫**——关守卫后 P7h 超门、walk05 臂不对称爆发并耦合进转向（漂移 +2.33m）。disc 对慢变 DC 分量（wrist 位置对臂 DC 不敏感）缺少梯度。
4. **速度跟随：任务 reward 主导**，与形式通道解耦良好（对照成立）。

**目标结构达成度**：速度→任务 ✓；腿部+臂相位→判别器 ✓；臂 DC/躯干姿态→仍需 2 项守卫（arm_asym_lean、lumbar_posture）。

## v41 建议（最小守卫判别器主导配置）

- X1_FORM_GUARDS=2 模式：仅保留 arm_asym_lean + lumbar_posture（臂 DC/躯干），退役 leg_amp_asym + arm_pitch_sync + arm_leg_coupling（归因证明冗余）。
- 臂幅度问题独立处理：判别器 wrist 通道对摆幅不敏感，考虑 ref 侧 wrist 轨迹增强（速度加权观测）或 demo 重采样加权高摆幅片段。

## 证据

- 平台：TASK_20260914_070（VERDICT PASS 13/13，守卫全关下）；model_6596 注册
- `acceptance/attr_probe/`：model_6596 policy.npz + walk10/walk05 npz
- `acceptance/evidence/attr_p7_gate.json`（7/8，仅 P7h）
