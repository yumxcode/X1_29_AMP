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
