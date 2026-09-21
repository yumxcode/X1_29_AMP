import numpy as np, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'sim2sim'))
from sim2sim.mujoco_rollout import build_model, find_sole_geoms, DEFAULT_Q
import mujoco

model, _ = build_model(ROOT / "gmr_x1_assets" / "x1.xml")
data = mujoco.MjData(model)
soles, _ = find_sole_geoms(model)
feet_names = sorted(soles)
mj_names = [model.joint(i).name for i in range(model.njnt)]
hinge = [n for n in mj_names if n in DEFAULT_Q]
jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])

def fund(x, dt, lo=0.4, hi=2.5):
    t = np.arange(len(x))*dt
    c = np.polyfit(t, x, 2); x = x - np.polyval(c, t)
    x = (x-x.mean())/(x.std()+1e-12)
    w = np.hanning(len(x))
    nfft = 16*len(x)
    X = np.abs(np.fft.rfft(x*w, nfft))**2
    f = np.fft.rfftfreq(nfft, dt)
    band = (f >= 1/hi) & (f <= 1/lo)
    fd = f[band][np.argmax(X[band])]
    return fd

print('REF clips (foot-z fundamental = gait cycle):')
for name in ['0002_treadmill_slow','0000_treadmill_norm','0005_normal_walk1','0007_normal_walk3','0008_normal_walk4','36_01','36_11','0026_circle_walk']:
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
            zs[i,f] = pts[:,2].min(); mxy[i,f] = pts[:,:2].mean(0)
    cys = [1/fund(zs[:,f], dt) for f in range(2)]
    cyc = float(np.median(cys))
    # overground stride: displacement of foot midpoint between successive cycle-phase points
    # robust: median over cycle of |mxy[i+cycle]-mxy[i]|
    per_i = int(round(cyc/dt))
    strides = {0: [], 1: []}
    for f in range(2):
        for i in range(0, T-per_i-1, max(1,per_i//4)):
            strides[f].append(float(np.linalg.norm(mxy[i+per_i, f]-mxy[i, f])))
    sl = float(np.median(strides[0]+strides[1]))
    v = float(np.median(np.linalg.norm(np.diff(rp[:,:2],axis=0),axis=1)/dt))
    print(f'  {name:24s} cycle L={cys[0]:.3f} R={cys[1]:.3f} -> {120/cyc:6.1f} spm | stride~{sl:.3f} m | v={v:.2f} vimpl={sl/cyc:.2f}')
