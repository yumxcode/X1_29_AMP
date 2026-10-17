#!/usr/bin/env python3
"""v71 launcher: FROM-SCRATCH AMP 29DOF + EXOGENOUS GAIT CLOCK (rhythm arc).

用户指令（2026-10-17）：从头重训 AMP 29DOF；在现有 sim2sim 标准之上新增
步频+抬腿步幅要求（数值见 acceptance/X1_GAIT_SPEC_V71.md，基于 X1 物理
规格：L=0.70m、T*(v) Froude/demo 标定、抬腿 h*(1.0)=50mm）。

背景（18 轮证伪，V64-V70）：AMP+RL 内奖励侧锚定/数据侧/从头训/LPF/延迟/
demo 初始化/逐帧跟踪均无法翻转 2.2Hz 快踏吸引子 —— 策略自选"何时迈步"时
快微步恒为平衡易解，接触时序模仿信号无法传导。v71 的机制变量只有一个：
**外生步态时钟**（策略观测 sin/cos 相位 + 任务按 z*(φ)/接触窗打分），
"何时触地/抬多高"从涌现吸引子变为外生指令（V67 解锁方向② GaitGT 族）。

Package（vs v66 from-scratch 基线的全部差异）：
  [clock] X1_GAIT_CLOCK=1（obs +4 维、foot_clock w=1.0、contact_clock w=-0.5）
          X1_HIP_PHASE=0.4（时钟锁相髋摆参考——策略现在能观测相位）
          （X1_CADENCE_PRIOR/X1_SWING_PRIOR 保持 0：被时钟取代）
  [base]  v68 辨识动力学（assets/robots/x1.py，非本启动器变量）
          soup59c 配方：ROBUST=3 / FORM_GUARDS=1 / DISC_VEL=1 / YAW_GUARD=1.0
  [disc]  观测包：600ms 窗（NUM_STEPS=30）+ root linvel + BUFFER=24
          数据侧：X1_MOTION_DIR=x1_lab_v67（overground 主导）+ VMATCH=1
  [guards] v69/v70 交付剂量：TOE .6 / ABD .8 / SWAY .4 / STANCE_MAX .38 /
          SY .5 / LEAN .5（防外八/腰摆/小臂折叠/臂前倾的形态回归）

From-scratch guards:
  - X1_RESUME_CKPT / X1_FINE_TUNE_ITERS deliberately UNSET；
  - 预检 repo 根目录无 model_*.pt（run_x1_amp_train 会自动捡拾挂载点）；
  - X1_FRESH_ITERS=4000（v66 教训：pod 墙钟可能吃掉尾部 → 梯子 save 500）。

PRE-REGISTERED accept（acceptance/X1_GAIT_SPEC_V71.md §5，非回归+新门）：
  A. 非回归：平台 AMP 13/13（P1-P6）；本地 5 场景存活 11.5s；v-err walk10/05；
     P7 walk10/walk05 PASS；臂摆≥24°；漂移≤1.0m；鲁棒≥48/60。
  B. 节奏（binding）：R1 步频 walk10∈[0.74,1.18]Hz/足、walk05 周期[1.0,1.65]s；
     R2 摆动[0.28,0.60]s；R3 占空[0.45,0.75]；R4 步长≥0.78/0.42m；
     R5 事件≥4/足；R6 CV≤0.20；L1 抬腿 apex median≥35mm（p95≥25 由检测器
     隐含）—— 双场景（walk10+walk05），acceptance/rhythm_gates.py。
Decision tree:
  A+B 全过 → 交付（soup 候选按 v69 流程）；
  A 过 B 不过 → 时钟包在 from-scratch 下证伪（第 19 轮），如实报告；
  A 不过（未收敛/塌陷）→ v71b 延续轮补收敛后再判 B；
  A 不过且时钟项明显干扰（收入/存活与 v66 同期差>2σ）→ 剂量降档重跑。
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- from-scratch pre-flight: no resume checkpoint anywhere near -----------
os.environ.pop("X1_RESUME_CKPT", None)
os.environ.pop("X1_FINE_TUNE_ITERS", None)
_stray = sorted((_here.parent).glob("model_*.pt"))
if _stray:
    raise FileNotFoundError(
        f"FROM-SCRATCH pre-flight: repo root holds mounted checkpoints "
        f"{[p.name for p in _stray]} — run_x1_amp_train would auto-resume; "
        f"remove/rename them first")

# --- soup59c validated base -------------------------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"

# --- disc observability + data side (v63/v65-validated package) ------------
os.environ["X1_AMP_NUM_STEPS"] = "30"        # 600 ms disc window @50Hz
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v67"   # overground-dominant library
os.environ["X1_DISC_VMATCH"] = "1"           # speed-matched demo fetch

# --- v69/v70 delivery-dose posture guards ----------------------------------
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_LEAN_GUARD"] = "0.5"

# --- THE v71 mechanism: exogenous gait clock -------------------------------
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"
os.environ["X1_HIP_PHASE"] = "0.4"
os.environ["X1_CADENCE_PRIOR"] = "0"         # superseded by the clock
os.environ["X1_SWING_PRIOR"] = "0"

# --- from-scratch budget ----------------------------------------------------
os.environ["X1_FRESH_ITERS"] = "4000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
