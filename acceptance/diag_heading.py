import sys

sys.path.insert(0, '.')
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import play_motion_mujoco as pm  # noqa: E402

model = mujoco.MjModel.from_xml_path(pm.DEFAULT_MODEL)
data = mujoco.MjData(model)
fps, rp, rr, dof, names, fmt, m = pm.load_motion('v25_unpacked/x1_gmr/0026_circle_walk.pkl')
qpos = pm.build_qpos(model, rp, rr, dof, names)
angles = []
for t in range(0, 1990, 10):
    data.qpos[:] = qpos[t]
    mujoco.mj_forward(model, data)
    fwd = data.xmat[model.body('base_link').id].reshape(3, 3)[:, 0]  # robot +x
    vel = rp[t + 10] - rp[t]
    if np.linalg.norm(vel[:2]) < 1e-4:
        continue
    c = np.dot(fwd[:2], vel[:2]) / (np.linalg.norm(fwd[:2]) * np.linalg.norm(vel[:2]) + 1e-9)
    angles.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
a = np.array(angles)
print(f'朝向-速度夹角: 中位数 {np.median(a):5.1f}° | <30° 占比 {(a < 30).mean() * 100:.0f}% | 帧数 {len(a)}')
