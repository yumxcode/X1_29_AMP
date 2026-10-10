# F1_29DOF AMP 行走 v1 交付报告

日期：2026-10-11 ｜ 账户：gekorej721（余额尽）→ xiroyen240（用户指令：与 X1 分账户）
仓库：github.com/yumxcode/X1_29_AMP（main）
交付策略：**model_5498**（v1h，TASK_20261011_002，F1-AMP @50Hz，5498 iters）

## 1. 任务契约完成度

| # | 要求 | 状态 | 证据 |
|---|------|------|------|
| 1 | F1_29DOF URDF（参照 X1，仅下肢 12DOF 差异） | ✅ | `F1_29DOF/urdf/f1_29dof.urdf`（29 revolute，32.2kg，树校验 PASS；正交髋重设计+右腿完美镜像） |
| 2 | 动作重定向（X1 重定向参考指标） | ✅ | `f1_lab_v67` 20 clips；`check_f1_retarget.py` **PASS 0 FAIL / 0 WARN**（膝相关 0.970、摆幅保留 1.003、A–E 全门） |
| 3 | 50Hz 行走算法 + mujoco sim2sim（X1 标准） | ✅栈/⚠️门 | 稳定行走达成（12s×15 场景 15/15 不摔、9.94m@1.0m/s 命令）；X1 门 6/9 类通过（见 §3，节奏/跟先/臂摆未达 X1 champion 13/14 水平） |

## 2. 模型与管线（全新资产，全部入库）

- **URDF**：`build_f1_29dof_urdf.py` 合并 X1 上身 + F1 下肢。FK 实测：F1 全伸展髋→踝 534.2mm（X1 574.1），髋宽 290.4mm（X1 267.9），左右镜像 x-asym 0.00mm。
- **重定向**：`retarget_f1_from_x1lab.py`——上身/根轨迹逐帧复制，下肢 6-DOF DLS IK 逐帧重解至 X1 踝世界位姿；**垂直分量按腿长比缩放 0.926**（可达性可证：‖t−hip‖²=‖d‖²−(1−s²)dz²≤s²·574²=534²；水平不缩放 → stance 脚世界系不滑步）。多起点 IK，p99 残差 1.0–6.7mm。
- **训练**：`F1-AMP` 环境子类化 X1AmpEnvCfg（继承 v67 全部奖励结构），F1_CFG 站立位形为 FK 验证 IK 解（残差 0.0mm），4096 envs @50Hz。
- **sim2sim**：`gmr_f1_assets/f1.xml`（URDF 派生 MJCF，与 URDF FK 交叉验证 0.000mm）+ `mujoco_rollout.py --robot f1`；本地 Mac mujoco 3.1.1 shim（`run_f1_sim2sim_local.py`，字符串注入地板 + v26 动力学对齐 damping1.0/armature0.01/frictionloss0）。

## 3. sim2sim 验收（v1h model_5498，X1 gait_metrics 门）

walk10（1.0 m/s 前进，12s，seed0）：

| 门 | 结果 | 数值 | X1 champion 参考 |
|----|------|------|------------------|
| G1 稳定 | **PASS** | 9.94m 不摔，\|v误差\|0.153，base_z cv 0.0056 | 13/14 门全过基线 |
| G2 对称 | FAIL | L/R 摆幅比 0.77–0.79（门≥0.85）；step_len 0.40m（R2 门≥0.78m） | hip 0.94/knee 0.94 |
| G3 落地 | **PASS** | 25 事件全平掌（pitch@TD 2.7°）、穿透 1.7mm、scuff 0 | 全过 |
| K1 直膝 | **PASS** | stance 中期膝屈 12.1/10.7°（门≤18°） | 10.5–12.9° |
| K2 膝节律 | 边缘 | 21.3/24.4°（门≥25°） | ~26° |
| H1 跟先 | FAIL | 0%（X1 champion 也仅 14–41%，此门 X1 自身未过） | 14–41% |
| H2 蹬离 | **PASS** | toe-off 100% | 100% |
| H3 趾先 | FAIL | L 12.5%（门≤10%） | 0–9.5% |
| R 节奏 | FAIL | 周期 0.46s 规律（cv 0.021✓）但步幅/节奏门未达 | R1–R4 过 |
| 鲁棒性 | **PASS**(紧凑) | 15/15 alive（lat1/noise1x/mass±15%/push0.75 × fwd10/stand/turn） | 48–54/60 |

**真实步态形态**（视频 `v1h_5498_walk10.mp4` + Gemini 审读）：双脚**正常交替**（各 26 次摆动/12s，反相）、摆脚离地清晰、无跛行/膝塌陷；缺陷：趾先着地、手臂近僵直（肩摆 ptp 21.8/15.9° vs X1 冠军 ~28°+）。
**检测器注**：右脚仅 1 触地事件是 Schmitt 阈值边缘（峰值 p50 12.9mm vs off 门 12mm、持续时间不足 80ms 去抖）而非不迈步——视频与 lift-segment 分析（两脚各 26 次抬脚）证实交替存在。

## 4. 训练弧线（4 轮有效 + 3 轮基础设施）

| 轮 | iters | 配方 | reward | sim2sim 结论 |
|----|-------|------|--------|--------------|
| v1 (091) | 0→4000 | v67 recipe | −1.7→8.4 | 稳定拖步：L 脚摆高 11.4mm<12mm 门（0 事件），R 19.6mm 正常 |
| v1d/v1e | →5499 | +SWING_CLEAR | ~14.3 | checkpoint 丢失（SDK 不上传，见 §5） |
| **v1h (002)** | 3999→5498 | +SWING_CLEAR | ~14.3 | **交付**：L 17.1mm✓ 24 事件；R 12.9mm 边缘；H3 12.5% |
| v1i (009) | 5498→6997 | 同上 | 平台 | 回退：趾先 73.9%/100%（SWING_CLEAR 学会跖屈够摆高——X1 v57d ankle-flick 病理复现） |

**结论**：SWING_CLEAR −0.5 有效（11→17mm）但后续 iters 走向跖屈作弊而非屈膝抬脚；demo 参考摆高 45–58mm，policy 峰值仅 ~30%。与 X1 弧线最终结论一致：此 AMP 栈在节奏/风格维度有结构性上限（X1 用 15 轮杠杆堆叠才达 13/14）。

## 5. 基础设施发现（gradmotion SDK，已修复入库）

1. **PT 上传单目录锁存**：SDK 每任务只认领第一个含 .pt 的目录；launcher 自建 `_resume_src`（含 .pt）会饿死训练 run dir 的全部上传（v1d/v1e/v1f 三轮丢失 model_4000–5498）。
2. **删除已见目录会崩扫描器**：v1g 删 _resume_src 触发 SDK Errno2 → 任务被杀。
3. **修复（v1h 验证有效）**：resume 源改名 `.bin`（SDK 只扫 .pt/.pth/.ckpt；train.py --checkpoint 是文件名正则）+ run dir 正常认领上传（"PT file uploaded successfully" ×4）。
4. 其他：`--resume` 是布尔标志需配 `--load_run/--checkpoint`；任务 JSON `personalDataPath` 与 ESKU000001 不兼容（err 601）；`gm task logs` 混 SDK 噪声（`filter_task_log.py`）。

## 6. 下一步杠杆（供续作）

1. **K2/摆高**：demo 摆高 45–58mm vs policy 30% —— 提高 SWING_CLEAR 剂量有 v52@0.3 过量先例，建议配 ankle_flick 加倍防跖屈作弊（v1i 教训）。
2. **H1 跟先**：X1 也未过；heel_first 0.40 已在配方中。
3. **臂摆**：X1 的 arm_amp_prior 0.06 剂量弧线（0.08→29–32°）；F1 可从 0.08 起步。
4. **节奏 R2 步幅 0.4 vs 0.78**：周期 0.46s 已对（人类 1.0m/s ~0.55s 单侧），步幅不足一半——cadence/stride 杠杆（X1_CADENCE_PRIOR/X1_SWING_PRIOR）未启用。
5. 训练预算：v1 曲线 4000 iters 处 reward 8.4 仍陡升，X1 收敛 ~20 —— F1 或需 8–10k iters 才到平台。

## 7. 交付物清单

- 模型：`F1_29DOF/urdf/f1_29dof.urdf`、`F1_29DOF/mjcf/`、`gmr_f1_assets/`、`roboparty_train/robolab/data/robots/f1/`
- 数据：`roboparty_train/robolab/data/motions/f1_lab_v67/`（20 clips + 报告）
- 代码：`build_f1_29dof_urdf.py`、`fk_lib.py`、`retarget_f1_from_x1lab.py`、`f1.py`、`f1_amp_env_cfg.py`、`f1_amp_agent_cfg.py`、`run_f1_amp_train.py`、`build_f1_mjcf.py`、`run_f1_sim2sim_local.py`、`run_f1_robustness_local.py`、`acceptance/check_f1_retarget.py`
- 策略：`acceptance/f1_v1/model_5498.pt` / `policy_5498.npz`（+3999/6997 对照）
- 证据：`acceptance/f1_v1/*.npz`、`*_gait_report.json`、`v1h_5498_walk10.mp4`、`v1h_5498_robustness_compact.json`
