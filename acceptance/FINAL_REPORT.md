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

## SDK 注册根因破解（探针 TASK_20260818_075，2026-08-18 13:45）

- **规则**：SDK 扫描**非递归**——只有平铺在仓库根 `/workspace/isaaclab/X1_29_AMP/*.pt`
  的文件被上传+注册（探针 loc1/late1 均 "uploaded successfully"，loadRun=X1_29_AMP）。
  model_upload/、ckpt_reg/、logs/、x1_upload/、/workspace 根的文件仅 "detected
  globally"，从未上传。时间与大小无关（t+15min 的 1.3KB 晚期文件照常上传）。
- v16-v21b checkpoint 四连败完全解释：所有镜像布局都在子目录，仓库根从未平铺过任何 .pt。
- v22 修复：最终 checkpoint + 全部报告 .pt 平铺仓库根（commit d589c86）。
- 证据：`acceptance/evidence/v22_probe_sdk_rules_log.txt`

## v22（TASK_20260818_124，2026-08-18 14:00 启动）

预期 13/13 + 视频 + sim2sim 指标 + checkpoint/报告平台注册。完成窗口 ~17:30-18:00。

## 复核命令（下次会话直接执行）

```bash
K=$(account-pool list | grep "id=24" | sed 's/.*api_key=//' | tr -d ' ')
gm --api-key "$K" task info --task-id TASK_20260818_010    # status 5 = 完成
gm --api-key "$K" task model list --task-id TASK_20260818_010   # 应见 model_*.pt
gm --api-key "$K" task logs --task-id TASK_20260818_010   # 尾部 AMP VERDICT
```

若 v21b 失败：诊断顺序为账号余额（试 id=22 misino1603 / id=23 mevesa9407）→
git clone 权限（coref25034 同 349588189@qq.com GitHub 凭证）→ 日志定位。

---

# v23 → v25 最终后验（2026-08-25，账号 misino1603 id=22）

## v23（TASK_20260824_018）：注册通道实证 + 12/13

- **发现 `model_upload/` 注册通道**：SDK 任务早期（~t+5min）枚举该目录，此后持续监听。
  v23 中 model_3999.pt + 视频（logs/{exp}/ 下 mp4 自动生成 videoUrl）**首次成功注册**。
- AMP 验收 12/13，唯一 FAIL 仍为 P6（MuJoCo 渲染矩阵全灭，无诊断输出）。

## v24（TASK_20260824_120，commit c5062af）：12/13 + **0 产物注册**

- 训练指标与 v23 持平（P2-P5 稳定通过）。
- 注册失败根因（后验）：① GMR 克隆带入 gvhmr_pt/ 下 5 个垃圾参考 .pt，t+6min
  偷光 5 槽配额；② model_upload/ 建晚了（错过 t0 枚举窗口，整任务期不被监听）。
- MuJoCo 矩阵再次全灭且无诊断（诊断写本地文件+复用已关句柄，教训入经验库）。

## v25（TASK_20260825_019，commit 438b295）：**被余额耗尽终止于 86.5%**

### 修复项全部生效（早期验证 ✓）

| 修复 | 验证结果 |
|---|---|
| t0 锚点 pipeline_meta.pt | 08:49:04 注册 ✓（证明 model_upload/ 在枚举窗口内）|
| gvhmr_pt 前移隐藏 | 垃圾 .pt 注册数 = **0** ✓（5 槽全保）|
| 重定向门 | **VERDICT: PASS fails=0**（14 文件，warns=184 均为 B2 软限/E3 取证类非阻断）|
| 产物注册 | 3/5 槽：pipeline_meta + model_retarget_report + model_retarget_data（38MB）✓ |

### 训练：健康且已收敛，但被外力终止

- 终止点 **iteration 3458/4000（86.5%）**，endTime 2026-08-25 10:51:36，status 6，
  runtime 7509s（≈2.09h × 5.4 元/h ≈ **11.3 元**——账号余额就此耗尽）。
- 终止前最后日志块：**ep_len 995.5/1000、mean reward 21.33、base_contact=0.0000、
  timeout 0.98、err_xy 0.377、err_yaw 0.833**——策略已收敛到接近满长不摔（对照
  v23/v24 4000 iter 终点：ep_len ~993、err_xy ~0.39、base_contact=0，12/13 中 P2-P5 全过，
  本曲线已在同一水平）。
- 日志尾部无任何报错，argo 主日志同样截断于训练块——外部强制停止，非代码崩溃。

### 损失与残留

- **唯一磁盘 checkpoint = model_0.pt（随机初始）**：save_interval=4000，最终保存点
  在 4000 iter，未到达。训练成果（3458 iter）随 pod 销毁丢失，npz 导出、MuJoCo
  视频、AMP VERDICT 三阶段均未执行。
- 已救回（本地 `acceptance/v25_artifacts/`，平台网页同样可下载）：
  model_retarget_data.pt（38MB 重定向动作数据）、model_retarget_report.pt、
  pipeline_meta.pt。

## 终局结论

| 子目标 | 状态 | 证据 |
|---|---|---|
| 重定向精准 + 严格指标 + 通过 | ✅ | 门控 PASS fails=0（v23/v24/v25 三连）；数据+报告平台可下载 |
| 训练不摔/速度跟随 | ✅（以 v23/v24 4000-iter 12/13 为准）| v25 同配置 86.5% 处已同水平收敛（ep_len 995、err_xy 0.38）|
| 最终策略 checkpoint 可下载 | ❌ | v25 死于余额耗尽，仅存 model_0（随机）|
| 行走视频（P6）| ❌ | 从未有任何一次成功出片（v23-v25 渲染链未在真实任务中验证成功）|
| AMP 13/13 | ❌ | v23/v24=12/13；v25 未跑完验收 |

**若续跑（需充值 ≥15 元）**：v25 配置已被证明三段全绿（注册链、门控、训练健康），
唯一未验证的是结尾三阶段（导出→软渲染→VERDICT，本地 Mac 已对渲染管线端到端
验证出片）。全流程 ≈2h30m ≈ 13.5 元；可复用已下载的 38MB 重定向数据跳过 GMR
重定向（省 ~20min ≈ 1.8 元）。
