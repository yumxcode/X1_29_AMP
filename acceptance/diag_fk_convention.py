"""Compare MuJoCo FK (from stored root/dof) against stored body_positions.

Root-relative comparison cancels the post-FK root shift done in
run_gmr_retarget.py (P6 fix) that made stored root_pos inconsistent with
stored body_positions.
"""
import pickle
import sys

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('X1_29DOF/mjcf/xyber_x1_flat.xml')
data = mujoco.MjData(model)
mj_bodies = [model.body(i).name for i in range(model.nbody)]
qadr = [model.jnt_qposadr[i] for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]


def fk_bodies(rp, rr, dof):
    data.qpos[:] = 0
    data.qpos[:3] = rp
    data.qpos[3:7] = rr
    for a, v in zip(qadr, dof):
        data.qpos[a] = v
    mujoco.mj_forward(model, data)
    return data.xpos.copy(), {model.body(i).name: data.xpos[i].copy() for i in range(model.nbody)}


CONVS = {
    'wxyz':      lambda q: q,
    'xyzw':      lambda q: q[[3, 0, 1, 2]],
    'wxyz_conj': lambda q: q * np.array([1, -1, -1, -1]),
    'xyzw_conj': lambda q: q[[3, 0, 1, 2]] * np.array([1, -1, -1, -1]),
    'z180_wxyz': lambda q: np.array([0, 0, 1, 0.]) * 0 + q,  # placeholder
}


def run(pkl_path, sample=8):
    m = pickle.load(open(pkl_path, 'rb'))
    bp = np.asarray(m['body_positions'])          # (T,30,3) truth
    names = list(m['body_names'])
    rp = np.asarray(m['root_pos'])
    rr = np.asarray(m['root_rot'])
    dof = np.asarray(m['dof_pos'])
    T = bp.shape[0]
    frames = np.linspace(0, T - 1, sample).astype(int)

    # body-name -> index map for truth, matched to mjcf body names
    idx = [names.index(n) for n in names if n in mj_bodies]
    root_name = names[0]
    print(f"\n=== {pkl_path.split('/')[-2]}/{pkl_path.split('/')[-1]}  "
          f"(T={T}, bodies matched={len(idx)}/{len(names)}, root={root_name})")

    for cname, fn in CONVS.items():
        if cname == 'z180_wxyz':
            continue
        errs, errs_root = [], []
        for t in frames:
            q = fn(rr[t] / np.linalg.norm(rr[t]))
            _, byname = fk_bodies(rp[t], q, dof[t])
            root_true = bp[t, 0]           # truth root body position
            root_mj = byname.get(root_name)
            if root_mj is None:
                continue
            # root-relative: subtract each side's own root
            rel_true = np.array([bp[t, names.index(n)] - root_true for n in names if n in mj_bodies])
            rel_mj = np.array([byname[n] - root_mj for n in names if n in mj_bodies])
            errs.append(np.linalg.norm(rel_true - rel_mj, axis=1).mean())
            errs_root.append(np.linalg.norm(root_mj - root_true))
        print(f"  {cname:10s} root-rel body err = {np.mean(errs):7.4f} m   "
              f"root abs err = {np.mean(errs_root):7.4f} m")


for name in ['0000_treadmill_norm', '0026_circle_walk', '114_08', '36_11']:
    run(f'acceptance/v25_unpacked/x1_gmr/{name}.pkl')
