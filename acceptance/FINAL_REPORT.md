# X1 AMP 训练——阶段交付报告
日期：2026-08-18 ｜ 账号：xetot43672@rpaintel.com（余额耗尽）→ coref25034@rpaintel.com
仓库：github.com/yumxcode/X1_29_AMP（main@2558f24）

## 原始目标与完成度

| 子目标 | 状态 | 证据 |
|---|---|---|
| 1. GMR 重定向严格验收标准 | ✅ 完成 | `acceptance/RETARGET_ACCEPTANCE.md` + `check_retarget.py` v1.2 |
| 2. 重定向验收**通过** | ✅ 完成 | v20（TASK_20260817_083）门控 VERDICT: PASS（训练启动=通过，FAIL 即 sys.exit）|
| 3. AMP 训练验收标准 | ✅ 完成 | `acceptance/AMP_ACCEPTANCE.md` + `check_amp.py`（P1-P6+TARGET）|
| 4. AMP 验收**通过** | ⏳ 差一项 | v20 = 11/13（P1 off-by-one 已修并实测 4000/4000；P5a/P6 已修待 v21 重跑）|
| 5. sim2sim | ⏳ 代码就绪未出产物 | `sim2sim/mujoco_rollout.py` + `run_sim2sim_task.py`；v21 起内联为视频兜底 |
| 6. 视频行走结果 | ⏳ 待 v21b | v20 Isaac RecordVideo 未产出 mp4（exit=0 但无文件）；v21 已加 MuJoCo 兜底渲染 |

## 关键实证结果（v20，TASK_20260817_083，lerp0.75/4000iter）

- retarget 门控：**PASS**（checker v1.2，14 文件，0 FAIL）
- AMP 验收 11/13（P1 修复后复核）：
  - PASS：ep_len 991.7/1000、timeout 0.984、base_contact=0、
    **lin kernel 0.8422**（≥0.82）、ang kernel 0.5540（≥0.50）、
    err_xy 0.398（≤0.44）、err_yaw 0.841（≤0.95）、无后期崩溃（20.79≈best 20.82）、
    disc_loss 0.0006
  - FAIL：P5a style 0.146<0.15（lerp 0.75 所致，v21 回 0.6，预期 ~0.335）；
    P6 无视频（v21 已修）
- TARGET（用户 85% 跟踪）：**未达**。lin kernel v16=0.8391、v20=0.8422，两次独立
  配置均 ~0.84 平台期，属结构性差距，非迭代数问题（末段 3% 仅 +0.4%）。

## 过程中发现并修复的系统性问题（本次主要工程价值）

1. **验收门三连误杀→v1.2 校准**（v18: A4 fps 120≠50 假设×14；v19: C/D/E/G 共 27 项
   fixture 假设 vs 真实产物）。修正均以读码+真实产物实证：fps 由 motion_data_manager
   按 dt=1/fps 消费、右髋 URDF 符号镜像、root_z 是 SMPL 骨盆轨迹、gmr_to_lab 对
   root_rot 做 convert_quat+quat_unique+normalize（G2 改语义比较 |dot|≥1-1e-3）。
2. **checkpoint 上传根因（v16-v20 四连败）**：SDK 只注册**仓库工作树外** .pt（唯一
   正例=gvhmr 垃圾注册 4 次）；`logs/` 被 .gitignore 挡、`model_upload/` 从未注册。
   v21 主镜像改 `/workspace/isaaclab/x1_upload/{tag}/`（树外）。
3. **P1 off-by-one**：rsl_rl 迭代号 0-indexed，4000 次=最后日志 "iteration 3999"。
   已用 v20 真实日志干跑验证修复。
4. **lerp 实证决策**：0.75 相对 0.6 的 lin kernel 收益 +0.003（噪声级）但 style
   0.335→0.146 腰斩。终选 0.6（P3a/P5a 双侧大余量）。

## 未完成与缺口（如实）

- **v21b（TASK_20260818_010）在途**：xetot43672 余额耗尽致 v21 秒败（3min 无日志）；
  已换 coref25034（同 GitHub 凭证）重启。预期产出：AMP 验收 13/13 + checkpoint
  真正注册 + MuJoCo/Isaac 行走视频。
- 最终 checkpoint 从未成功上传平台（v21b 验证树外镜像方案是否成立）。
- 无最终行走视频产物。
- TARGET 0.85 跟踪未达成（预计仍 ~0.84；如需突破需算法侧改动而非调参）。

## 关键文件

- 验收：`acceptance/RETARGET_ACCEPTANCE.md`、`check_retarget.py`（v1.2）、
  `acceptance/AMP_ACCEPTANCE.md`、`check_amp.py`
- 管线：`roboparty_train/run_x1_amp_train.py`（v21：树外镜像+视频兜底）
- sim2sim：`sim2sim/mujoco_rollout.py`、`sim2sim/run_sim2sim_task.py`
- 超参：`x1_amp_agent_cfg.py`（4000 iter / save 500 / lerp 0.6）

## v21b 最新状态（2026-08-18 收尾快照，最终核验）

- TASK_20260818_010（coref25034/PRO_20260818_002）status 3 运行中，09:12:11 启动。
- **重定向门控已通过**（管线设计：gate FAIL 即 sys.exit，训练不可能启动）。
- 训练进度：**iteration 1686/4000**（~42%，日志拉取时点）。按此速率预计
  ~13:20-13:50 训练完成，之后自动执行 play+视频（Isaac→MuJoCo 兜底）→
  AMP 验收 → 420s 上传尾巴，全程无人值守自动完成。
- 收尾核验清单（自动化已内置于管线，人工只需确认）：

## v21b 完成后的收尾清单（自动化已内置于管线）

1. `gm task model list`：确认 model_*.pt（loadRun={tag}）注册成功
2. 日志尾部：AMP 验收 VERDICT（预期 13/13）
3. 视频证据（Isaac mp4 或 MuJoCo 兜底）→ 交付
4. `gm task data get` 交叉复核曲线

## v21b 最终结果（TASK_20260818_010，coref25034，12:35 完成）

- retarget 门控：PASS（第二次真实产物实证通过）
- 训练：4000/4000 迭代完成（P1 修复验证 ✅）
- **AMP 验收 12/13**（唯一 FAIL = P6 视频）：
  - P1 ✅ 4000/4000 ｜ P5a ✅ style **0.333**（lerp 0.6 修复验证）
  - P3a lin kernel **0.8469**（≥0.82，较 v20 +0.005）｜ ang 0.5672 ｜ err_xy 0.389 ｜ err_yaw 0.812
  - P2a ep_len 992.7 ｜ timeout 0.9826 ｜ base_contact=0 ｜ P4 无崩溃（22.79≈22.80）｜ disc 0.0007
  - P6 ❌：Isaac play exit=0 仍无 mp4；MuJoCo 兜底死于 pip PyOpenGL 损坏（已定位+修复）
- checkpoint 注册：**0 个**（终态复查）。gvhmr 垃圾曾在 09:20 占据 5 个注册位又被平台清除——
  支持"每任务 5 个配额被垃圾吃满"假设；x1_upload 树外镜像 3h 未注册。
- 证据：`acceptance/evidence/v21b_amp_acceptance_12of13.txt`、`v21b_phase5_video_attempt.txt`、
  `v21b_task_log_tail.txt`

## v22 修复（已实现，待探针结果定稿后启动）

1. MuJoCo 渲染矩阵：系统 Python 优先（容器自带 mujoco 3.6 + imageio）→ pylibs（剥离损坏
   PyOpenGL）→ apt 装 osmesa 兜底；3 组 rollout（1.0/1.5/转弯）+ 指标 JSON 包裹上传
2. gvhmr 垃圾 .pt 在训练开始前清除（v21b 证明它们从树外也被注册）
3. 训练期不再镜像中间 checkpoint；final sweep 只镜像最终 ckpt；save_interval 500→4000
   （磁盘只剩 model_0 + model_3999，配额假设下 final+报告恰好 5 个）
4. 探针任务 TASK_20260818_075（10 个标记 .pt × 7 位置 × 早晚时间窗）实证注册规则

## 复核命令（下次会话直接执行）

```bash
K=$(account-pool list | grep "id=24" | sed 's/.*api_key=//' | tr -d ' ')
gm --api-key "$K" task info --task-id TASK_20260818_010    # status 5 = 完成
gm --api-key "$K" task model list --task-id TASK_20260818_010   # 应见 model_*.pt
gm --api-key "$K" task logs --task-id TASK_20260818_010   # 尾部 AMP VERDICT
```

若 v21b 失败：诊断顺序为账号余额（试 id=22 misino1603 / id=23 mevesa9407）→
git clone 权限（coref25034 同 349588189@qq.com GitHub 凭证）→ 日志定位。
