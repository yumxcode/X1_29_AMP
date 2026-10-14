# V68 报告：sim2real 辨识动力学重训（X1 29DOF，50Hz，soup59c 配方）

日期：2026-10-14 ｜ 账号：jebexi1407@leafflip.com（id=45）｜ 仓库 main@bb56008
任务：TASK_20261014_099（round 1，完成）+ TASK_20261014_114（round 2，进行中）

## 契约（contract rev4）

1. 按 `x1_armature_pd.md`（sim2real 参数辨识定稿，方案 B）的新惯量与 kp/kd 训练；
2. 50Hz 控制频率；不追求极致 heel-to-toe；sim2sim 标准 = **soup59c 步态标准**
   （臂腿协调、交叉摆臂、稳定行走）；行走速度 **0.5–1.0 m/s**；
3. 训练配方 = soup59c 配方（v59c cell）；基座 = 实际部署的 soup59c 模型；
4. 训练对象 **X1**（非 F1）；训练走 gradmotion；长周期观测 timer park。

## 动机（A/B 实证，acceptance/v68_eval/baseline_*）

| 策略 | 动力学 | 1.0 m/s | 0.5 m/s |
|---|---|---|---|
| soup59c（部署冠军） | legacy（旧 PD/armature 0.01） | 11.5s 稳走 10.82m，v-err 0.079 | 11.5s 稳走 5.37m，v-err 0.038 |
| soup59c | **辨识后（kp 65-100/kd 8/armature 0.012-0.273）** | **1.66s 摔倒** | **1.72s 摔倒** |

→ 辨识动力学差（重腿 20×armature + kp -46%）使部署冠军即刻失稳，重训必要且充分。

## 配置变更（全部转录校验 PASS：roboparty_train/check_v68_params.py）

- `assets/robots/x1.py`：actuators 全表替换（md §1）；ζ 0.41–1.24、fn 1.1–7.8Hz、
  kd·dt/I_tot<0.8@200Hz 复算全对
- `x1_amp_env_cfg.py`：armature 随机化 (0.8,1.2)→**(0.8,1.3)**；damping →**(0.8,1.0)**
  （kd=8 是固件上限，1.2×8=9.6 不可达）；stiffness 保持 (0.8,1.2)
- `sim2sim/mujoco_rollout.py`：PD 表 + per-joint ARMATURE 表（md §3）+ `--pd-preset legacy`
  A/B 开关；md §3 提及的 F1 rollout_bisect.py 不在本工作区（另行套用同表）
- 部署 yaml 片段：`sim2sim/deploy_rl_walk_amp_pd_v68.yaml`（md §4 数组程序化生成并对表）

## 训练（round 1：TASK_20261014_099，15:11–16:03）

配方 = v59c cell 不变：`X1_ROBUST_TRAIN=3 / FORM_GUARDS=1 / DISC_VEL=1 / YAW_GUARD=1.0`；
resume `model_soup59c.pt`（iter 9500，含 disc/optimizer 全状态）+ **1200 iters** @50Hz；
镜像 BJX00000178/V000220（isaac-lab2.3.2，py3.11）。

平台验收：**AMP 13/13 PASS**、**P7 策略步态门 PASS**（rc=0）、P8 节律 FAIL（家族性
log-only，soup59c 本尊同样不过）。检查点梯子 10000/10500/10699 全部注册并下载。

## 本地 sim2sim 电池（新动力学 = 默认 preset，50Hz）

筛选（0.5/1.0 m/s × 全梯子 + soup）：**全部 11.5s 稳定行走**，v-err 0.08–0.14 m/s。

| 指标（soup59c 标准） | 门 | v68 最优点 | 冠军 soup59c_50 |
|---|---|---|---|
| 场景存活（walk10/05/back05/stand/walkturn） | 5/5 | **5/5** ✓ | 5/5 |
| 速度带 0.5–1.0 m/s | 契约 | **0.084/0.111 m/s v-err** ✓ | 0.038/0.079 |
| 臂摆幅 walk10（交叉摆臂） | ≥24° | **33.0/33.8°**（soup 端点 35.6/37.1°）✓ | 34.7/34.1° |
| 臂反相关（交叉相位） | ≤-0.45 | **-0.996～-0.998** ✓ | -0.995 |
| 鲁棒矩阵 | ≥48/60 | **52/60** ✓（超冠军） | 51/60 |
| 平台 AMP 验收 | 13/13 | **13/13** ✓ | 13/13 |
| 侧向漂移 walk10 | ≤1.0m | **1.47m**（soup(10000,10699)）✗ | 0.18m |
| 侧向漂移 walk05 | ≤1.0m | 2.91m ✗ | 0.73m |

- K1（膝中位）walk10 L=23.3° vs 冠军 13.8°（门≤18）✗；K2 16.8 vs 22.3。
- 漂移为**持续侧向蠕变**（soup 端点 146mm/s vs 冠军 32mm/s ≈ 8° 航向偏置），
  确定性（种子无关）；梯子内非单调（1.50/4.24/2.26m）→ 判定为**未完成的适应瞬态**
  而非收敛均衡 → round 2（TASK_20261014_114，+800 iters 同配方自 m10699）验证。

## 平台阻塞修复（本次工程副产物）

1. 镜像 V000124=py3.8：roboparty 管线 `Path|None` 注解必炸 + run_gmr_retarget amass
   守卫误杀 → 换 roboparty 已验证组合 BJX00000178/V000220（isaac-lab2.3.2/py3.11）。
2. `check_retarget_gait.py` S3 门控：unweighted 文件（103_07 对，冠军配方即如此、
   从不加载）被 v67 加固误杀 → 改 WARN-only（dangling 仍 FAIL），本地复跑 PASS。

## Round 2 结果

（待 TASK_20261014_114 完成后补：漂移是否 ≤1.0m、最终交付点、部署建议）
