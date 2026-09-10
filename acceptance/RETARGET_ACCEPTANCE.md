# X1 GMR 重定向结果验收标准（严格版）

版本: v1.0 (2026-08-17)
适用对象: `roboparty_train/robolab/data/motions/x1_gmr/*.pkl`（GMR/MuJoCo 域）与
`roboparty_train/robolab/data/motions/x1_lab/*.pkl`（Isaac Lab 域，dataset_retarget 产物）。
执行器: `acceptance/check_retarget.py`（纯 numpy，可选 mujoco FK；容器内 GMR venv 与本地均可运行）。

## 0. 背景与数据链路

```
AMASS_minimal/{CMU,BMLrub_stageii}/*.npz  (14 个, 与 env cfg motion_data_weights 一一对应)
  → GMR IK (smplx→x1, MuJoCo x1.xml, auto-IK 校准)        → x1_gmr/*.pkl
  → dataset_retarget.py (纯关节重排 gmr→lab + Isaac FK)     → x1_lab/*.pkl
  → X1-AMP 训练 (discriminator demo 数据)
```

关键事实（作为阈值依据）:
- X1: 29 DOF；MuJoCo 模型 `gmr_x1_assets/x1.xml` 站姿 keyframe root_z = 0.61 m（弯膝）；
  Isaac `X1_CFG` 站姿 init z = 0.75 m（不同基座约定，两域各自校验）。
- GMR 产物 pkl 字段: `fps, root_pos, root_rot(wxyz), dof_names(MJ序), body_names,
  dof_positions/dof_pos, body_positions, body_rotations`；root 已做贴地校正
  （全体 body 最低点 = 0.04 m，即踝原点到脚底 ~4 cm）。
- lab 产物 pkl 字段: `fps, root_pos, root_rot, dof_pos(lab序), loop_mode, key_body_pos`，
  由 `extract_gmr_data` **纯重排**得到（无重采样、无 root 修改），关节角应与 gmr 域按名逐位相等。
- 帧率: GMR 保留 AMASS stageii 源采样率（本数据集实测 120 fps）。下游 `motion_data_manager` 按 `dt=1/fps` 逐动作取时间戳采样，任何物理合理的采集帧率均合法；允许 30–250（拦截 fps=0/1/NaN 类坏值）。
- 验收分为**硬门（FAIL → 阻断训练）**与**警示门（WARN → 记录但不阻断）**。
  结构/运动学/物理约束全部为硬门；运动语义（F 组）为警示门。

## A. 结构完整性（硬门）

| # | 检查 | 判据 |
|---|------|------|
| A1 | x1_gmr 每个 pkl 可加载且字段齐全 | 缺任一字段 FAIL |
| A2 | DOF 数量与命名 | `dof_names` 逐项等于 `config/x1.yaml:gmr_dof_names`（29 项，顺序一致）|
| A3 | 文件集合完整性 | `motion_data_weights` 14 项全部存在（缺失 FAIL）；多余文件 WARN |
| A4 | 帧率与时长 | fps ∈ [30, 250]；帧数 ≥ 100（≥2 s），否则 FAIL |
| A5 | 四元数良构 | 每帧 ‖q‖₂ 与 1 偏差 < 1e-3，否则 FAIL |
| A6 | x1_lab 结构 | 14 个 pkl；字段 = {fps, root_pos, root_rot, dof_pos, loop_mode, key_body_pos}；`dof_pos.shape[1]==29`；`key_body_pos.shape[1]==6`；与 gmr 同名文件帧数一致 |

## B. 关节限位（硬门，依据 x1.xml / f1.urdf hinge range）

| # | 检查 | 判据 |
|---|------|------|
| B1 | 不超硬限位 | 任一关节任一帧超出 MJCF range 的幅度 > 0.02 rad，或超限帧占比 > 0.1% → FAIL（允许 IK 数值毛刺）|
| B2 | 软限位占比 | 超出"限位内缩 5%"区间的帧占比 ≤ 5%（按关节），超出仅 WARN | 部分关节自然姿态即贴限位边界（如 shoulder_roll (-2,0) 垂臂≈0），故降为警示 |

## C. 根轨迹物理性（硬门）

| # | 检查 | 判据 | 依据 |
|---|------|------|------|
| C1 | 高度均值 | root_z mean ∈ [0.30, 1.20] m | root_z = GMR IK 的 SMPL 骨盆轨迹：treadmill 文件归一化到机器人尺度（实测均值 0.45–0.75），地面走/跑文件保留源骨盆高度（实测 0.79–0.95）；阈值只拦病理数据（NaN/量级错误），v19 实测全 14 文件覆盖 |
| C2 | 高度极值 | min ≥ 0.20 m，max ≤ 1.50 m | 实测极值带 [0.60, 1.38]（jog 弹跳）；下限拦跪倒/深蹲失控，上限拦离地乱飞 |
| C3 | 垂直速度 | 帧间 \|Δz\| > 0.05 m 的帧占比 > 2% FAIL（@120fps → 6 m/s）| 单帧 IK 尖峰（实测 <0.6%）容忍，系统性跳变拦截 |
| C4 | 水平速度 | 帧间水平位移 > 0.08 m 的帧占比 > 2% FAIL（@120fps → 9.6 m/s）| 同上；jog ~3 m/s 正常 |

## D. 关节运动平滑性（硬门）

| # | 检查 | 判据 | 依据 |
|---|------|------|------|
| D1 | 速度上限 | 任一帧 \|Δq\| > 3.0 rad（@120fps → 360 rad/s，物理不可能）→ FAIL；\|Δq\| > 0.5 rad 的帧占比 > 1% → FAIL；零星单帧 > 0.5 rad → WARN | v19 实测：单帧尖峰 1.75 rad（210 rad/s，IK 偶发）与起始帧瞬态 0.56 rad，均不构成系统性污染；系统性抖动必拦 |
| D2 | 平滑度 | 全关节中位帧速 ≤ 0.15 rad，P99 ≤ 0.35 rad，否则 FAIL | 正常步态 <10 rad/s |

## E. FK 一致性与地面约束（硬门；需 mujoco，缺失则该项跳过并标注）

| # | 检查 | 判据 |
|---|------|------|
| E1 | 无穿地 | 用 x1.xml 重算 FK：所有 body z ≥ −0.02 m（任意帧），否则 FAIL |
| E2 | 脚部离地 | 双脚（ankle_roll link）z < −0.03 m → FAIL；< 0.01 m → WARN | v19 实测最低 0.008 m（jog 冲击帧）；3 cm 以上穿地=数据损坏，1 cm 内贴地=边缘取证 |
| E3 | 存档 FK 一致 | 重算 FK body 位置 vs pkl `body_positions` 平均误差 ≤ 5e-3 m，超出仅 WARN | 该字段是 GMR 内部量（世界系/局部系未文档化），且不进入下游（lab 的 key_body_pos 由 Isaac FK 重算），仅作约定取证 |

## F. 运动语义（警示门 WARN——不阻断，但逐条记录进报告）

| # | 检查 | 判据 | 备注 |
|---|------|------|------|
| F1 | 前进速度 | 非 treadmill/stand 文件：平均水平速度 ∈ [0.1, 3.0] m/s | treadmill 原地走豁免；AMASS normal_walk 实测 0.14–0.17 m/s 属合法慢走 |
| F2 | 步态相位 | 文件名含 walk/jog/run/treadmill：L/R hip_pitch 关节角相关系数 > +0.5 | X1 URDF 右髋符号镜像（限位 L=(-1,2) vs R=(-2,1)）：关节角同相 = 物理摆腿反相 = 正常步态（v16/v17 实测 +0.98~0.99）；低相关提示步态不对称 |
| F3 | 双支撑 | walk 类文件 ≥10% 帧双脚 z < 0.06 m | 步行应有双支撑相 |

## G. lab 转换保真（硬门）

| # | 检查 | 判据 |
|---|------|------|
| G1 | 角度逐位一致 | x1_lab.dof_pos 按名字映射回 gmr 序后，与 x1_gmr.dof_pos 的 max\|Δq\| ≤ 0.005 rad（转换是纯重排，理论上=0）|
| G2 | 根轨迹一致 | root_pos max\|Δ\| ≤ 1e-6（`extract_gmr_data` 全量切片拷贝，`run_simulator` 不回写 root_pos）；root_rot 语义比较：gmr 四元数（xyzw）经 `convert_quat(wxyz)+quat_unique+normalize` 变换后与 lab 的 min\|dot\| ≥ 1 − 1e-3（浮点噪声余量）| gmr_to_lab 对 root_rot 做合法变换（读码确认，v19 实测差异 1.1–1.4 纯为 wxyz↔xyzw 分量错位）|
| G3 | key_body 合法 | key_body_pos 有限值（无 NaN/Inf），z 分量 ≥ −0.05 m |

## 判定

- **通过（PASS）**: 无任何 FAIL。
- **系统性 WARN 升级策略（v1.2 修订）**: 仅当 WARN 属于"系统性出现会污染训练"的检查（白名单 F1 / F2 / G0）且在 ≥3 个文件触发时升级为 FAIL。B2（软限位）与 E3（存档 FK 约定取证）不参与升级——两者是本机器人+GMR IK 的固有数据特征，同特性数据已在 v16/v17 训练中实证可用。
- **阈值校准记录（v1.2，2026-08-17）**: v18/v19 两次门控运行用真实管线产物校准——A4 帧率 30–250（GMR 保留 AMASS 120fps）、F2 右髋符号镜像、C1/C2 骨盆轨迹带、C3/C4 系统性占比、D1 双阈值、E2 双阈值、G2 四元数语义比较。所有 FAIL 判定均与下游代码（motion_data_manager 按 dt=1/fps 采样、gmr_to_lab 根变换）交叉验证；v16/v17 同管线训练成功作为数据可用性实证。
- **未通过（FAIL）**: 任一 FAIL，或白名单 WARN 系统性触发。
- 容器内执行点：GMR retarget + dataset_retarget 完成后、AMP 训练启动前；FAIL 则**阻断训练**并上传报告。
- 本地复验点：下载 `model_retarget_data.pt`（含 x1_gmr+x1_lab 全部 pkl）后独立重跑本 checker。

## H 组 — 上肢关节分解门（v30 新增，2026-09-10）

v29 复盘发现 GMR IK 在任务空间正确、但关节空间病态：肘 pitch 105–113° 顶满限位
（114.6°）+ 肩 yaw −32° 内旋拼出"前臂朝下"，腰 yaw 摆幅 46–56°（人类 spine 链
14–18° 的 3 倍）。AMP 判别器观测 joint_pos，把该失真学成了风格先验 → v29 策略
"靠腰摆上身、小臂抬起"。B2 软限位 WARN 当年被豁免正是因为缺这组检查。

执行器：`roboparty_train/fix_arm_decomposition.py`（管线 Phase 2.5，产出
`x1_lab_v30/`，之后 `mirror_lab_motions.py` 生成镜像；训练 env 读 x1_lab_v30）。

| # | 检查 | 判据 | 说明 |
|---|------|------|------|
| H1 | 肘铰链轨迹保持 | 解算后 elbow_pitch_link（解剖肘，GMR 对齐 SMPLX elbow）世界轨迹前向分量漂移 p95 ≤ 15 mm | 前向分量=摆臂信号本身；垂直/侧向放松（各向异性权重 250/60/60）以换取腰压缩，属不可见自由度 |
| H2 | 肘 pitch 健康位 | p95 ≤ 65°（先验目标 20°=人类走路屈曲；源 SMPLX ~20°） | 旧数据 105–113° 顶限位 |
| H3 | 肩/肘 yaw 补偿清除 | \|shoulder_yaw\| 均值 ≤ 15° | 旧数据 −32° 恒定内旋 |
| H4 | 腰 yaw 摆幅 | p95−p5 ≤ 32°（目标压缩到原摆幅 35%；t1 达标 19–25°） | 肩 roll 外展限位阻塞时分级回退 t2/t3 并标 PRTL（114/127 四片段因此被移出 v30 训练权重） |
| H5 | 存档 FK 一致性 | 修复前 MuJoCo FK vs pkl 存档 key_body_pos p95 ≤ 15 mm | 验证脚本 FK 复刻的正确性 |

方法：逐帧阻尼 Gauss-Newton 重解 11 关节（lumY + 双臂 5×2），保持肘铰链轨迹 +
姿态先验（elbP→20°、shoY/elbY→0、lumY→中位数+35% 原摆幅），回溯线搜索。
腕位置/前臂方向**故意不跟踪**——GMR 正是靠扭曲解才够到 SMPLX 腕点（X1 臂按
0.75 缩放），任何腕匹配都会把解拉回 106° 扭曲。
诊断工具：`acceptance/diag_arm_swing.py`（关节空间）、`diag_arm_swing_smplx.py`
（AMASS 源对照）、`probe_arm_joints.py`（X1 关节语义 FK 探针）。

## J 组 — 步态质量总门（v31，必过，2026-09-10）

执行器：`acceptance/check_retarget_gait.py`（管线 Phase 2.6，FAIL 阻断训练）。
整合 H/I 组全部指标为可执行门 + 结构/协调/镜像检查，阈值按 x1_lab_v31 实测校准：

| 组 | 检查 | 判据 |
|---|------|------|
| S | 结构 | N 源 + N 镜像成对；pkl schema/四元数范数/有限值；env 权重表↔文件集双向 1:1 |
| F | FK 保真 | FK(dof,root) vs 存档 key_body_pos p95 ≤ 15mm（列序错位/静默 fancy-index 免疫门） |
| A | 上肢 | elbP p95≤65°、\|shoY\|≤15°、\|elbY\|≤20°、几何肘弯≤45°、肩反相≤−0.45 |
| T | 躯干 | lumY 摆幅分级：PRIMARY/CIRCLE≤32°、JOG≤65°、CMU_OLD≤55°、CMU_NEW≤75° |
| G | 地面 | 穿模≥−3mm 全帧、支撑 pitch \|L−R\|≤3.5/6°、\|pitch\|中位分级≤15/25° |
| C | 协调 | 髋 L/R corr≥+0.3、臂-对侧腿耦合≥+0.45、髋摆幅比分级（PRIMARY [0.7,1.4]，CIRCLE/CMU 放宽） |
| M | 镜像 | 每 源有 mirror 且 key_body = y-flip+swap(p95≤20mm) |

分级设计依据：CIRCLE 内外腿天然不对称（0026 实测 1.44）；CMU_OLD 源固有不对称由镜像对补偿（36_01 实测 2.05，WARN-only）。
**门的战果**：首跑即拒 138_18（GMR 将其源对称摆臂翻转为同相，anti +0.47/耦合 −0.20）——视觉 4s 视频无法察觉、判别器可学的风格缺陷。

## I 组 — 地面接触与对地约束门（v31 新增，2026-09-10）

v30 复盘（diag_gait_plausibility.py / diag_wholebody.py）：GMR IK 无对地约束——
脚底穿模 7–10.6% 接触帧（最深 33.6mm），触地脚底 pitch 左右不对称（中位
+13.9° vs +4.6°）；且 BMLrub 全库为跑步机/原地协议（净位移 0.01–0.03m），
AMASS_minimal 的走路片段实为原地踏步。判别器观测全在 root 系，穿模与原地性
不影响 v16–v30 训练，但参考携带物理不可能性与错误的步幅-速度耦合。

执行器：`roboparty_train/fix_ground_root.py`（管线 Phase 2.5 第二级，
v30 → v31；之后 mirror；训练 env 读 x1_lab_v31）。

| # | 检查 | 判据 | 说明 |
|---|------|------|------|
| I1 | 穿模清零 | 全帧任意脚最低点 ≥ −3mm | 原 −33.6mm；root_z 支撑锚定（σ=6 帧平滑 + 全局不穿模钳位） |
| I2 | 踝对称 | 支撑相脚底 pitch 中位 \|L−R\| ≤ 3.5°（jog 放宽 6°） | 相位归一化双侧滚动剖面均值化 ×2 迭代，踝 pitch 单自由度修正（只旋转踝下足部，不影响腿臂） |
| I3 | 上肢不变 | elbP p95 ≤ 66°（应仍为 20°） | 地面修复不得触碰臂部修复成果 |
| I4 | 输入 FK | v30 存档 key_body_pos 复刻 p95 ≤ 15mm | 映射正确性自检 |

设计决策记录：
- **root_xy 不做原地→行进转换**——BMLrub 源是真原地走（支撑脚相对 root 不后
  移，锚定积分测得 ≈0 m/s），刚体平移只会造成支撑脚前滑。真实行进步态由
  v31 新增 CMU 片段（103_07 / 138_18，净位移 3.3–4.6m，1.0–1.2 m/s）提供，
  本地 GMR 链重定向（`setup_gmr_local.py` + `retarget_new_clips_local.py` +
  `gmr_to_lab_local.py`，复用远程 auto-IK 配置 acceptance/v25_unpacked/
  smplx_to_x1_auto.json）。
- 新片段选片门（scan_amass_walking.py + diag_cmu_candidates.py，全库 5044 扫
  116 过初筛）：源躯干反旋（spine−pelvis yaw 去趋势摆幅）<30°、摆臂对称
  corr>0.6、肘 −40~0°、步频 60–140、髋摆幅比 0.75–1.3。138_01/03/04 因源
  躯干反旋 60°+ 撤选。
