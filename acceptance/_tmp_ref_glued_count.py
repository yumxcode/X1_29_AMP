import numpy as np, pickle, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'sim2sim'))
from sim2sim.mujoco_rollout import build_model, find_sole_geoms, DEFAULT_Q
from sim2sim.gait_metrics import schmitt_contact, stance_windows
import mujoco

model, _ = build_model(ROOT / "gmr_x1_assets" / "x1.xml")
data = mujoco.MjData(model)
soles, _ = find_sole_geoms(model)
feet_names = sorted(soles)
mj_names = [model.joint(i).name for i in range(model.njnt)]
hinge = [n for n in mj_names if n in DEFAULT_Q]
jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])

LAB = ['left_hip_pitch_joint','lumbar_yaw_joint','right_hip_pitch_joint','left_hip_roll_joint','lumbar_roll_joint','right_hip_roll_joint','left_hip_yaw_joint','lumbar_pitch_joint','right_hip_yaw_joint','left_knee_pitch_joint','left_shoulder_pitch_joint','right_shoulder_pitch_joint','right_knee_pitch_joint','left_ankle_pitch_joint','left_shoulder_roll_joint','right_shoulder_roll_joint','right_ankle_pitch_joint','left_ankle_roll_joint','left_shoulder_yaw_joint','right_shoulder_yaw_joint','right_ankle_roll_joint','left_elbow_pitch_joint','right_elbow_pitch_joint','left_elbow_yaw_joint','right_elbow_yaw_joint','left_wrist_pitch_joint','right_wrist_pitch_joint','left_wrist_roll_joint','right_wrist_roll_joint']

def hipdiff_cycle(q, fps):
    x = q[:, LAB.index('left_hip_pitch_joint')] - q[:, LAB.index('right_hip_pitch_joint')]
    x = x - x.mean()
    ac = np.correlate(x, x, 'full')[len(x)-1:]
    ac /= (ac[0]+1e-12)
    # strongest peak in physiological band 0.7-2.0 s
    best, bv = None, -1
    for k in range(int(0.7*fps), min(len(ac)-1, int(2.0*fps))):
        if ac[k-1] < ac[k] >= ac[k+1] and ac[k] > bv:
            best, bv = k/fps, ac[k]
    return best, bv

print(f'{"clip":24s} {"hipAc_s":>8s} {"cnt_s":>7s} {"spm":>6s} {"stride_m":>9s} {"v":>5s} {"vimpl":>6s} nTD')
out = {}
for name in ['0002_treadmill_slow','0000_treadmill_norm','0005_normal_walk1','0007_normal_walk3','0008_normal_walk4','36_01','36_11','0026_circle_walk','103_07']:
    clip = pickle.load(open(ROOT/f'roboparty_train/robolab/data/motions/x1_lab/{name}.pkl','rb'), encoding='latin1')
    q = np.asarray(clip['dof_pos']); rp = np.asarray(clip['root_pos']); rr = np.asarray(clip['root_rot'])
    T, fps = len(q), float(clip['fps']); dt = 1/fps
    zs = np.zeros((T,2)); mxy = np.zeros((T,2,2))
    for i in range(T):
        data.qpos[:] = 0
        data.qpos[0:3] = rp[i]; data.qpos[3:7] = rr[i]/np.linalg.norm(rr[i]); data.qpos[qadr] = q[i]
        mujoco.mj_forward(model, data)
        for f,fn in enumerate(feet_names):
            pts = np.array([data.geom_xpos[g] for g in soles[fn]])
            zs[i,f] = pts[:,2].min()-0.002; mxy[i,f] = pts[:,:2].mean(0)
    hipc, hipv = hipdiff_cycle(q, fps)
    # glued touchdown counting: glue airborne gaps <= 250 ms
    ntds, strides = {}, []
    for f,fn in enumerate(feet_names):
        on = zs[:,f] < 0.012
        gaps, i = [], 0
        # find off-gaps
        off = ~on
        j = 0
        while j < T:
            if off[j]:
                k = j
                while k < T and off[k]: k += 1
                if 0 < (k-j)*dt <= 0.25 and j > 0 and k < T:
                    pass  # candidate glue
                gaps.append((j, k))
                j = k
            else:
                j += 1
        on2 = on.copy()
        for a, b in gaps:
            if (b-a)*dt <= 0.25:
                on2[a:b] = True
        edges = np.diff(on2.astype(int), prepend=0)
        tds = np.where(edges == 1)[0]
        ntds[fn] = len(tds)
        for a, b in zip(tds[:-1], tds[1:]):
            strides.append(float(np.linalg.norm(mxy[b, f]-mxy[a, f])))
    n = sum(ntds.values())
    dur = T*dt
    cnt_cycle = 2*dur/n if n else float('nan')   # both feet -> TDs per cycle = 2
    sl = float(np.median(strides)) if strides else float('nan')
    v = float(np.median(np.linalg.norm(np.diff(rp[:,:2],axis=0),axis=1)/dt))
    print(f'{name:24s} {hipc if hipc else 0:8.3f} {cnt_cycle:7.3f} {120/cnt_cycle if cnt_cycle else 0:6.1f} {sl:9.3f} {v:5.2f} {sl/cnt_cycle if cnt_cycle else 0:6.2f} {n}')
    out[name] = dict(hip_cycle_s=hipc, count_cycle_s=cnt_cycle, spm=120/cnt_cycle, stride_m=sl, v=v)
json.dump(out, open(ROOT/'acceptance/_tmp_ref_cadence2.json','w'), indent=1)
