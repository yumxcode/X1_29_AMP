#!/usr/bin/env python3
"""F1 weilai: merge the 23 fixed decorative links' inertia into the 30-link tree.

Defect (found 2026-11-10): the delivered f1_serial.xml / f1_isaac.urdf drop the
23 fixed links (waist motors, wrist mechanisms, neck/head/camera/radar) instead
of merging them -> 32.220 kg vs SYS-SPEC source 35.033 kg (2.813 kg lost, all
upper body), contradicting F1_AUDIT.md section 2 (35.03 kg).

Fold every fixed chain into its movable parent (parallel-axis + rotation) and
SURGICALLY rewrite only the 3 affected <inertial> entries (lumbar_pitch_link,
left/right_elbow_yaw_link) in:
  - F1_29DOF_weilai/mjcf/robot/f1/f1_serial.xml  (pos/quat/mass/diaginertia)
  - F1_29DOF_weilai/mjcf/robot/f1/f1_isaac.urdf  (mass/origin/inertia)
Everything else stays byte-identical. Run verify_f1w_mass.py afterwards.
"""
import re
import numpy as np
import xml.etree.ElementTree as ET

REPO = __file__.rsplit('/roboparty_train/', 1)[0]
SRC_URDF = f'{REPO}/F1_29DOF_weilai/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf'
ISAAC = f'{REPO}/F1_29DOF_weilai/mjcf/robot/f1/f1_isaac.urdf'
MJCF = f'{REPO}/F1_29DOF_weilai/mjcf/robot/f1/f1_serial.xml'
AFFECTED = ['lumbar_pitch_link', 'left_elbow_yaw_link', 'right_elbow_yaw_link']

def rpy2R(rpy):
    r, p, y = rpy
    cx, sx, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def parse_origin(el):
    if el is None:
        return np.zeros(3), np.eye(3)
    return (np.fromstring(el.get('xyz', '0 0 0'), sep=' '),
            rpy2R(np.fromstring(el.get('rpy', '0 0 0'), sep=' ')))

def get_inertial(link):
    ine = link.find('inertial')
    if ine is None or ine.find('mass') is None:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    m = float(ine.find('mass').get('value'))
    xyz, R = parse_origin(ine.find('origin'))
    Iel = ine.find('inertia')
    a = lambda k: float(Iel.get(k, '0'))
    I_loc = np.array([[a('ixx'), a('ixy'), a('ixz')],
                      [a('ixy'), a('iyy'), a('iyz')],
                      [a('ixz'), a('iyz'), a('izz')]])
    return m, xyz, R @ I_loc @ R.T

def combine(m1, c1, I1, m2, c2, I2):
    M = m1 + m2
    c = (m1 * c1 + m2 * c2) / M
    def shift(I, m, d):
        return I + m * (d @ d * np.eye(3) - np.outer(d, d))
    return M, c, shift(I1, m1, c1 - c) + shift(I2, m2, c2 - c)

def mat2quat_wxyz(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q

def fold_fixed(root):
    links = {l.get('name'): l for l in root.iter('link')}
    fixed = [j for j in root.iter('joint') if j.get('type') == 'fixed']
    merged = {}
    def contribution(name):
        return merged.get(name, (0.0, np.zeros(3), np.zeros((3, 3))))
    changed = True
    while changed and fixed:
        changed = False
        for j in fixed[:]:
            parent = j.find('parent').get('link')
            child = j.find('child').get('link')
            if any(j2.find('parent').get('link') == child for j2 in fixed if j2 is not j):
                continue
            xyz, R = parse_origin(j.find('origin'))
            mc, cc, Ic = get_inertial(links[child])
            me, ce, Ie = contribution(child)
            m, c, I = combine(mc, cc, Ic, me, ce, Ie)
            m0, c0, I0 = contribution(parent)
            merged[parent] = combine(m0, c0, I0, m, xyz + R @ c, R @ I @ R.T)
            fixed.remove(j)
            changed = True
    assert not fixed
    return merged, links

def main():
    src = ET.parse(SRC_URDF).getroot()
    merged, links = fold_fixed(src)

    out = {}
    for name in AFFECTED:
        m0, c0, I0 = get_inertial(links[name])
        me, ce, Ie = merged.get(name, (0.0, np.zeros(3), np.zeros((3, 3))))
        out[name] = combine(m0, c0, I0, me, ce, Ie)
        print(f'{name}: {m0:.4f} -> {out[name][0]:.4f} kg')

    # --- MJCF: replace the 3 inertial lines in-place (regex, byte-surgical) ---
    s = open(MJCF).read()
    for name, (m, c, I) in out.items():
        w, V = np.linalg.eigh(I)
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        q = mat2quat_wxyz(V)
        new = ('<inertial pos="{}" quat="{}" mass="{:.8g}" diaginertia="{}"/>'
               .format(' '.join(f'{v:.8g}' for v in c),
                       ' '.join(f'{v:.8g}' for v in q), m,
                       ' '.join(f'{v:.8g}' for v in np.maximum(w, 1e-9))))
        # inertial sits right after the <body name="X"> opening tag
        pat = re.compile(r'(<body name="%s"[^>]*>\s*)<inertial[^>]*/>' % re.escape(name))
        s2, n = pat.subn(lambda mo: mo.group(1) + new, s, count=1)
        assert n == 1, f'MJCF inertial not found for {name}'
        s = s2
    open(MJCF, 'w').write(s)
    print('mjcf updated (3 inertials)')

    # --- Isaac URDF: replace the 3 <inertial> blocks ---
    s = open(ISAAC).read()
    for name, (m, c, I) in out.items():
        xyz = ' '.join(f'{v:.8g}' for v in c)
        vals = dict(ixx=f'{I[0,0]:.8g}', iyy=f'{I[1,1]:.8g}', izz=f'{I[2,2]:.8g}',
                    ixy=f'{I[0,1]:.8g}', ixz=f'{I[0,2]:.8g}', iyz=f'{I[1,2]:.8g}')
        pat = re.compile(
            (r'(<link name="%s">\s*<inertial>\s*)<origin[^>]*/>(\s*<mass value=")[^"]+("\s*/>\s*'
             r'<inertia )[^\n]+(\n\s*</inertial>)') % re.escape(name),
            re.S)
        def repl(mo):
            origin = f'<origin xyz="{xyz}" rpy="0 0 0" />'
            inertia = ' '.join(f'{k}="{v}"' for k, v in vals.items())
            return (mo.group(1) + origin + mo.group(2) + f'{m:.8g}' +
                    mo.group(3) + inertia + ' />' + mo.group(4))
        s2, n = pat.subn(repl, s, count=1)
        assert n == 1, f'isaac inertial not found for {name}'
        s = s2
    open(ISAAC, 'w').write(s)
    print('isaac urdf updated (3 inertials)')

if __name__ == '__main__':
    main()
