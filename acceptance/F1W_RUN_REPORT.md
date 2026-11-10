# F1w AMP 跑步 v1 训练与 sim2sim 验收报告（负结果）

日期：2026-11-11 ｜ 判定：**FAIL**（预注册门不降低；按 X1 弧负结果协议登记）
对象：F1W-AMP v1（f1w_lab_v1 v7 演示库 52 片，F1 weilai 29DOF）

## 1. 执行记录（gradmotion 任务链）

| 段 | 任务 | 账户 | 迭代 | checkpoint | 结束原因 |
|---|---|---|---|---|---|
| v1e | TASK_20261110_162 | gifahim182(63) | 0→1644 | model_1500 上传 | 余额耗尽（18:57） |
| v1g | TASK_20261110_210 | hataxi6084(64) | 1500→2999 | model_2999 上传 | +1500 预算完（env 顺序 bug，见 §5.4） |
| v1h | TASK_20261110_233 | pepam51928(65) | 2999→5998 | **model_5998 上传** | +3000 预算完（6000 计划达成） |

最终策略：`roboparty_train/checkpoints/f1w_v1h_model_5998.pt`（iter 5998，59.8MB，
actor 300→512→256→128→29 + AMP 判别器）。发射前失败链（全部止损未烧卡/秒退）：
v1b(103) 门计数 54→52；v1c(107) 权重键双射崩溃；v1d(121) 平台资源竞争挂死 95min（用户确认）；
诊断 134 同因 60min 0 字节；v1f(206) 挂载名解析 bug 发射前停止。

## 2. 训练侧 P1-P6（预注册门，check_amp 阈值逐字不变）

测量通道：rsl_rl TensorBoard 标量（`gm task data get`，v1e+v1g+v1h 三段拼接 0→5998；
容器 stdout 文件平台不导出，标量与 stdout 是同一 runner 的同一指标，通道差异已注明）。
检查器：`acceptance/check_amp_chart.py`（阈值=acceptance/check_amp.py TH 逐字复制）。
明细 JSON：`acceptance/F1W_V1_TRAIN_P1P6.json`；原始序列 `sim2sim/out_f1w_v1/train_chart_series.json`。

| 门 | 阈值 | 实测（末100迭代均值） | 判定 |
|---|---|---|---|
| P1 迭代完整性 | 6000 计划 | 5998 | ok |
| P2a ep_len | ≥950 | 978.2 | ok |
| P2b timeout | ≥0.95 | 0.952 | ok |
| P2c base_contact | =0 | 0.0012 | **FAIL** |
| P2d bad_orient+base_h | ≤0.03 | 0.0483（0.0276+0.0207） | **FAIL** |
| P3a 线速度核 | ≥0.82 | 0.2632 | **FAIL** |
| P3b 角速度核 | ≥0.50 | 0.9706 | ok |
| P3c err_xy | ≤0.44 | 2.2215 | **FAIL** |
| P3d err_yaw | ≤0.95 | 0.4805 | ok |
| P4 无后期崩溃 | ≥0.90×best | 26.66 ≥ 0.975×27.31 | ok |
| P5a style | ≥0.15 | 0.0061 | **FAIL** |
| P5b disc_loss | ≤0.05 | 0.0001 | ok（数值过门，语义=判别器完胜） |

**P1-P6 = FAIL（5/11）**。策略收敛到"存活但不跟踪指令、无风格"的解：
ep_len 978/timeout 0.952（学会不摔）但速度跟踪核 0.26、style 0.006。
Isaac play 视频（F1W-AMP-Play @1.6）未产出 mp4（launcher mp4 搜索 0 命中），
如实登记；训练域证据以上表 6000 迭代曲线为准。

曲线关键形态（stitched）：
- mean_reward 0.76→26.7 单调升（P4 健康）；
- **disc_loss 在 ~q1（≈iter 1500）即 ≈0.000**，style 同步归零 —— 判别器从头完胜，
  风格先验全程无有效梯度；
- error_vel_xy 随训练**上升**（0.20→2.22）：奖励增长来自存活/姿态项而非跟踪。

## 3. sim2sim（P7 + RN1-RN6 预注册门）

MuJoCo `sim2sim/mujoco_rollout.py --robot f1w`（gmr_f1w_assets/f1w.xml 质量合并版；
PD_F1W/ARMATURE_F1W/passive 镜像 weilai URDF；时钟 env X1_CLOCK_T_A=1.265 等已带）。

| 场景 | cmd (m/s) | 存活 | 距离 | 判定 |
|---|---|---|---|---|
| run16 | 1.6 | **0.38 s**（FALL t=0.86s 含 settle） | 0.08 m | RN1 FAIL（门 ≥16.8 m/12s） |
| run10 | 1.0 | 0.50 s | 0.05 m | RN5 FAIL |
| walk05 | 0.5 | 1.26 s | 0.50 m | RN5 FAIL |
| v03 探针 | 0.3 | 0.74 s | 0.40 m | — |
| stand（cmd=0，**带外 OOD**，仅参考） | 0 | 0.60 s | 0.37 m | 不作判据 |

RN2/RN3/RN4（腾空/步频/步幅）：策略未存活 ≥2s，无稳态步态可测 —— FAIL（不可测即不过）。
RN6 鲁棒性：同上。P7 步态质量（check_policy_gait）：同上。
产物：`sim2sim/out_f1w_v1/run16.mp4`（摔倒过程视频）+ 各场景 json/npz。

**sim2sim = FAIL（全场景摔倒）**，与训练侧 P3/P5 失败一致（策略从未学成走/跑）。

## 4. 归因分析

### 4.1 主因（机制链，有数据支撑）

**演示库"根速-步频解耦" → 判别器平凡可分 → 风格先验失效 → 存活不跟踪解**：

1. 跑类演示根轨迹被重建到机器人可行速（0.41-0.63× 源速，fnd_73421d5e），
   但关节轨迹保留源步频/摆幅（时间轴未重采样）→ 演示内 (根速, 步频) 对物理不自洽：
   部分片段隐含步幅 ≥0.95-1.02 m（几何上限 ~0.95 m，实测 3-5/26 片触顶/超限，
   run_09 族步频检测本身因时标错位失败）。真机器人无法产生"慢根速+全速摆腿"组合。
2. 判别器观测含根速度通道（f1w_env.txt X1_DISC_VEL=1/X1_DISC_LINVEL=1，
   X1_DISC_VMATCH=1 速度匹配采样——与 X1 交付弧同配置），演示类的 (根速, 关节) 分布
   与策略可达分布几乎不重叠 → disc_loss ≈0 自始成立（q1 即 0.000）→ style≈0.006。
   X1 行走弧成功因走库根锚定后自洽（速度改动小）；跑库改动 0.37-0.59× 夸大十倍。
3. 风格与跟踪同时不可达时，alive/姿态/时钟项给出"原地微动不摔"的廉价最优
   （ep_len 978、err_xy 2.22、yaw 跟踪正常 0.97——纯扭转类指令可以糊弄）。

### 4.2 已排除（均有对照证据）

- **sim2sim 接线**：X1 y8397 经同一（本轮修改后）rollout 脚本 walk10 存活 7.5s、
  v_xy 误差 0.120、6.74 m —— 脚本/obs 装配/PD/动作缩放路径无恙。
- **obs 布局/归一化**：checkpoint normalizer term-major 统计（grav_z=-0.992、cmd_x
  0.981±0.87）与 rollout 装配一致；站立 obs 各通道 |z|≤3。
- **PD/动力学保持**：settle 段（25 步零动作 PD 保持）稳定 base_z 0.570。
- **时钟 env**：带/不带 X1_CLOCK_* 跑带 env 均在 <3s 摔（非时钟相位问题）。

### 4.3 平台与发射侧缺陷（已修复入库，不改变判定）

资源竞争挂死（pit_6da2477a：30min 0 日志→止损）；余额耗尽续训链
（checkpoint 周期上传+挂载名正则 358f1969）；env 文件加载顺序（361e1793）；
发射器 stdout-tail 可见性（05e22b55）；数据集门 52+权重双射（99bd452f）。

## 5. 预注册决策树命中与下一步

F1W_RUN_ACCEPTANCE §3"任何 P/RN FAIL：按 X1 弧负结果协议登记"——本报告即登记。
门未降低；本次无任何阈值修订。

v2 配方建议（按优先级，供下一轮预注册）：
1. **演示时间重标定（治本）**：对每条跑片按"重建根速/源根速"比例重采样时间轴
   （步频随速度同缩），使 (根速, 步频, 步幅) 自洽落在机器人可达包络内；
   重过 R1-R3 门（源域区分度重测）。
2. 保守起步：先用行走锚库+已自洽的慢跑片（08_01/03/06 类，步幅 0.45-0.58m）训练
   0.5-1.3 m/s 里程碑，再扩跑速带（比从 2.2 顶格命令直接学更符合 X1 弧渐进史）。
3. 训练带与演示质心对齐复核：命令范围收至演示实际覆盖（0.3-2.0），VMATCH 容差
   与覆盖联合校验（防空批）。
4. 判别器通道消融定位：X1_DISC_VEL=0（关节位姿-only 判别）对照一轮，
   隔离根速通道的贡献（X1 弧未在"重建速度差>2×"的库上开过根速通道）。

## 6. 交付物清单

- 数据量评估：acceptance/F1W_RUN_DATA_ASSESSMENT.md（fnd_6cf7cc71 外部基准）
- 数据清单与过门记录：roboparty_train/robolab/data/motions/f1w_lab_v1/
  （52 片 + _acceptance_report.json PASS fails=0 + _retarget_report.json 真残差）
- 速度目标依据：acceptance/F1W_RUN_SPEED_TARGET.md（[0.2,2.2] 训练带、≥1.6 验收）
- 训练任务与结果：§1 表 + acceptance/F1W_V1_TRAIN_P1P6.json + train_chart_series.json
- sim2sim 验收：本报告 §3 + sim2sim/out_f1w_v1/（run16.mp4 摔倒视频、json/npz）
- 策略入库：roboparty_train/checkpoints/f1w_v1h_model_5998.pt（负结果工件，
  供 v2 对照）+ 发射/检查/适配脚本全部入库
