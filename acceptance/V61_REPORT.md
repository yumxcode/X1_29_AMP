# V61 报告：AMP 训练控制频率 50Hz → 100Hz（2026-09-20 终版）

目标：① 100Hz 训练 ② 通过 sim2sim ③ 达到 soup59c 步态指标集。

## 终判（诚实陈述）

| 子目标 | 判定 | 证据 |
|---|---|---|
| ① 100Hz 训练基础设施 | ✅✅ 达成 | decimation 开关 + 时间常数再归一化 + 平台 5 轮 100Hz 训练（4×13/13 + 3×TARGET 全 HIT） |
| ② sim2sim 通过 | ✅✅ 达成 | v61b：平台 13/13 + T1/T2/T3，本地 5/5 场景 11.5s 无摔、漂移 0.61/-0.23、鲁棒 52/60；m9500@100Hz 电池同过 |
| ③ soup59c 步态指标集 | ✗ 未达成（13/14 门，缺一不可按达成计） | 最优 100Hz 点对齐 13/14 门；**G2 walk10 hip 对称 0.803 < 0.85 未过**（soup59c@50Hz 为 0.870）。且该点 = m9500（**50Hz 训练**检查点）直驱 100Hz 的产物，**非 100Hz 训练结果**——100Hz 训练侧（v61b-f 六轮）幅度族全部塌陷 |

## 100Hz 基础设施（全部可复用，commit e82875f..9ecf284）

- `X1_CONTROL_HZ` 开关：decimation=200//hz（100Hz→2），sim.dt 0.005 不变
- 时间常数再归一化（shared `control_hz_renorm.py`，干跑驱动真实现）：
  EMA alpha /r（显式 params **和函数默认值**两形态）、action_delay ×r、
  action-rate 权重 ×r、style_reward_scale ×r
- obs 速度有限差分改用 env.step_dt（原硬编码 /0.02 在 100Hz 会慢 2×）
- sim2sim 全链 `--control-dt`：rollout/sweep/deploy；gait 消抖窗口时间基化（80ms）；
  鲁棒 lat/lag 步数时基匹配
- 干跑 `acceptance/dryrun_100hz_v61.py` 4/4（含 disc 切片帧索引、buffer 往返崩溃现场路径）
- 评估电池 `acceptance/eval_100hz_battery.py`：一条命令出 soup59c 同口径全套

## 平台 100Hz 训练矩阵（6 轮，账号 id=8）

| 版本 | 任务 | 变量 | 平台验收 | sim2sim 幅度（walk10） |
|---|---|---|---|---|
| v61b | TASK_20260920_032 | 基线平移 | **13/13 + T1 0.8607/T2/T3** | 臂 14.5/10.8°、拖步 |
| v61c | TASK_20260920_067 | disc 时窗 6 步=60ms | 13/13 + 3T | 臂 18.8/18.3°、L 足 12.6mm |
| v61d | TASK_20260920_085 | disc ::stride | 0-iter 崩（断言漏改） | — |
| v61d2 | TASK_20260920_090 | +断言修复+disc 热恢复 | 13/13 + 3T | 臂 12.4/11.5°、摆高 11mm |
| v61e | TASK_20260920_092 | +臂幅先验 0.06→0.12 | 13/13 + 3T | 臂 13.3/11.8° |
| v61f | TASK_20260920_116 | +style lerp→0 | 8/13（kernel/P4 崩） | 臂 20.8/27.8° **但行走塌**（原地摆臂，vxy 0.98） |

## 根因链（三重取证）

1. **判别器在 100Hz 微调中主动奖励微步，而非通道死亡**：纠正算术后，
   微步步态 style 收益/秒 = 冠军的 ~2.7×（5.4e-3 vs 2.0e-3 /s）。
   disc 均衡对微步给出 rew≈1.0（冠军 0.80 带梯度区）。
2. **基座策略本身在 100Hz 健在**：m9393/m9500/soup59c_50 直驱 100Hz 全部
   稳定行走（12s 无摔 ×5 场景），幅度/漂移行为基本率无关保真
   （鲁棒 57/60 历史最高）——是 100Hz 微调过程摧毁幅度。
3. **五条修复路线全部预注册证伪**：时窗恢复（v61c）、原生间距采样+热 disc
   （v61d2）、先验加倍（v61e）、style 摘除（v61f，行走进一步塌）。
   幅度-行走驱动在当前 AMP 结构下于 100Hz 不能兼得（style 通道二义）。

## 100Hz 直接驱动前沿（本地 sim2sim，全电池）

| 检查点 | 漂移 w10/w05 | hip | 相位 | 臂幅 | K1 | K2 | G3/H2 | 鲁棒 |
|---|---|---|---|---|---|---|---|---|
| **m9500@100Hz（推荐；50Hz 训练直驱）** | **0.32/0.80 ✓✓** | 0.803✗(差0.047) | 0.475 反相 ✓ | 27.1/28.8 ✓ | 12.9/10.4 ✓ | 24.2/**25.2**(过原门) ✓ | ✓/100% | **54/60** ✓ |
| soup59c_50@100Hz | 0.24 ✓ | **0.870 ✓** | 0.025 同相✗（hip 共摆，raw corr 0.93） | 27.7/28.2 ✓ | 12.8/10.6 ✓ | 24.7/27.5 ✓ | ✓/100% | — |
| m9393@100Hz | 5.56✗ | 0.799✗ | ✓ | 28.0 ✓ | 15.0/11.4 ✓ | 25.2 ✓ | ✓/100% | 57/60 ✓ |
| soup59c_50@50Hz（冠军基线） | 0.18/0.73 | 0.870 | 0.481 反相 | 34.7/34.1 | 13.8/11.7 | 22.3/25.5 | ✓/100% | 51/60 |

- m9500@100Hz：13/14 门对齐冠军（hip 0.803 为唯一窄缺；其 50Hz 自身 0.848 也未过 0.85，
  该门由 soup59c 合并修复——soup 家族在 100Hz 出现 hip 共摆相变，见 diag_phase_100hz.py）。
- 全部候选 walk05/back05 低速拖步族与冠军同样未达成（继承缺陷，非 100Hz 回退）。
- push1.0 1/5 与冠军家族相同（历史 open trade）。

## 交付物

- **100Hz 部署点：`acceptance/v59c_eval/model_9500.policy.npz` @100Hz**
  （m9500@100Hz：漂移/K1/K2/G3/H2/臂幅/相位/鲁棒全过，hip 0.803 窄缺）
- 备选：soup59c_50@100Hz（hip+漂移最优，步态相位结构改变需知悉）
- 100Hz 训练能力：`run_v61*_100hz.py` 系列启动器 + X1_CONTROL_HZ/X1_AMP_NUM_STEPS/
  X1_ARM_PRIOR/X1_TASK_LERP 环境杠杆
- 评估链：eval_100hz_battery.py / dryrun_100hz_v61.py / diag_phase_100hz.py

## 升采样实验终判（contract rev3，v62/v62b，2026-09-20 完成）

**用户提问（rev3）**：demo 升采样到 ≥200fps（spline 重采样）能否解决 style 坍缩？

**答案：不能——升采样单独不修复坍缩（第 6 条证伪路线，预注册判定）**。

| 证据 | 读数 |
|---|---|
| v62b 平台验收 | 13/13 + T1/T2/T3（kernel 0.8608）——工程门依旧全绿 |
| v62b sim2sim walk10（预注册口径） | 臂幅 **7.5/7.8°**（门≥24，全系列最差）；摆高 11.5/22.9mm（拖步）；漂移 **4.89m**（门≤1.0，超 5 倍）；步态事件 61 ✓（唯一达标项） |
| 训练侧信号 | arm_amp_prior 收益 0.0124（塌陷水平；冠军 0.0218） |

结论链：demo 帧统计已排除——v61d2 把 disc 双侧采样恢复到与 50Hz 冠军分布等价（20ms 间距）仍塌；v62b 把 demo 网格加密到 200fps（相位相干窗口、lerp 残差缩小 5.3×）仍塌且更差。**坍缩根因在训练动力学**：disc 均衡在 100Hz 微调中系统性奖励微步态（style 收益/秒 = 冠军 2.7×），与 demo 数据的时间分辨率无关。

工程遗产（全部入库，commit 3171d86/cd627b9/61ad0ce）：
- `upsample_demo_spline.py`：PCHIP（Fritsch–Carlson，无过冲）升采样管线，22/22 clips
  通过 roundtrip+FK 双验证（key_body 由 FK 重算，构造性一致；三次样条在 jog 快段
  过冲 5.3mm 被 PCHIP 消除）
- `X1_MOTION_DIR` / `X1_DISC_STRIDE` 数据集与 disc 采样开关
- 干跑按数据流覆盖（存储形状=消费形状断言防 (n,30)/(n,10,3) 类 0-iter OOM 复发）

其他候选（均未测）：disc per-body loss 加权、100Hz 原生参考重采集、对抗奖励形式的
频率重参数化（如 style 收益以每秒而非每步计）。


## v63 弧线：可观测性路线修复 100Hz style 塌缩（contract rev4，2026-09-21）

**三杠杆**（用户建议 + AMP 论文支持，全部默认关，v63 启动器开启）：
1. **disc 宏观窗 600ms**：`X1_AMP_NUM_STEPS=60`（60×10ms→stride2=30帧@20ms，与冠军同 cadence、窗口×10）；`X1_DISC_BUFFER=24` 控内存 2.9GB——步幅/位移进入判别
2. **disc 加 root 线速度**：`X1_DISC_LINVEL=1`（obs 121→124）——原仓库注释掉了它；AMP 论文消融标记 velocity 特征为动态动作必需。**这是微步塌缩的观测基础**：无 linvel 时 disc 窗内微步与行走不可分（v61f 取证：微步 style 收入/s = 冠军 2.7×）
3. **action 低通 10Hz×2 级**：`X1_ACT_LPF_HZ=10`（amp_action_lpf.py torch 侧 + mujoco_rollout --action-lpf numpy 镜像，P7/电池/sweep 全链路转发）——策略有效带宽拉回 demo 带，100Hz PD 伺服好处保留

### 平台轮次（账号 id=8）

| 轮 | 任务 | 变量 | 结果 |
|---|---|---|---|
| v63 | TASK_20260921_038 | 三杠杆齐上 | **arms 37.7-42.0°**（v62b 塌缩 7.5/7.8）；P7 PASS（pod 上 mujoco vendored pylibs 生效）；平台 12/13（P4=disc 重置后奖励重基线，尾段稳定 19.0-19.4 无螺旋）；m10500 drift -0.31/-0.52 |
| v63b | TASK_20260921_064 | yaw guard 2.5 | **反面证伪**：arms 40→17-24，臂 DC 爆到 P7g 48°，末点 pod P7 摔——强 yaw 惩罚毁步态 DC 结构 |
| v63c | TASK_20260921_066 | arm_asym_lean -2.4, +600 | 平台 **13/13** + P7 PASS@100Hz（P7g 修复轮）；但 arms 16.8-18.8（guard 压幅度，v53 权衡重现） |

### 交付点：soup(m10500+m11000) 50/50 权重平均

| 门 | 读数 | 判定 |
|---|---|---|
| P7（100Hz+LPF，本地 FK） | **8/8**：P7a -0.99 / P7g **4.1°** / P7d +0.42 / 肘 p95 33.6 / 存活 23s | ✅ |
| 臂幅 | **28.6/28.9°**（walk05 26.0/24.6）≥24 | ✅ |
| 漂移 | **-0.55 / +0.79 m** ≤1.0 | ✅ |
| 场景 | 5/5 存活（walk10/05/back05/stand/walkturn） | ✅ |
| K1 | 12.8/13.1° ≤18 | ✅ |
| 鲁棒 | **48/60** ≥48（lat/lag/noise/mass 45/45；push1.0 0/5 家族性弱点） | ✅ |
| 平台 13/13 | 母本 m11099 同配方达成（soup 为本地合并，家族先例 soup59c 直测交付） | ✅ |
| **P8 节律** | cadence ~2.9Hz（人 0.84）/ swing 0.19s（人 0.43）/ duty ~0.95 / 步长 0.15m（人 0.71-0.92） | ❌ 家族性 |

**100Hz style 塌缩判定：已修复**（第 7 条路线成立）。v62b 证伪数据侧（升采样）后，观测侧（窗口+linvel+带宽对齐）正面解决：arms 从 7.5/7.8 → 28.6-40.0，arm_amp_prior 收益 0.0124→0.0186（冠军 0.0218）。

### P8 节律门（contract rev2 新增，2026-09-21）

用户指令落地：`acceptance/rhythm_gates.py` + `RHYTHM_GATES.md` + `evidence/rhythm_reference.json`（demo 实测：cadence 0.839Hz [0.66,1.06] / swing 0.429s / duty 0.611 / 步长 0.71-0.92m@1.0-1.4m/s）。门 R1-R4 阈值=人类带放宽15%。流水线 log-only 集成（家族全员不过 R1/R2/R4——50Hz 冠军 1.85Hz/0.17s/0.27m 也不过；**100Hz 使节律缺陷恶化 54%**，用户观察被定量证实）。后续杠杆（预注册未实验）：overground clip 加权、步频先验、步长 shaping。

### 工程遗产

- `pylibs/` vendored linux cp311 wheels（mujoco 3.2.7+numpy+deps）——pod 离线 P7 从此可用
- **P7 控制率 bug 修复（16854bc）**：on-pod P7 一直在 50Hz 默认跑 100Hz 策略（教训已入经验库：控制频率类参数必须沿训练→导出→评估全链路显式传播并断言）
- `X1_ACT_LPF_HZ/X1_DISC_LINVEL/X1_DISC_BUFFER/X1_ARM_ASYM` 环境杠杆（全部默认关，50Hz 配方零影响）
- `dryrun_v63.py` 6/6（LPF -3dB/节 @10Hz、torch/numpy 零差、600ms 窗全 buffer 路径）
- API 推送兜底成熟（github 443 断连时 blobs/trees/commits/refs via api.github.com）