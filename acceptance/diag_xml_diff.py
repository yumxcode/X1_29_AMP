"""Diff joint definitions between gmr_x1_assets/x1.xml and X1_29DOF mjcf."""
import xml.etree.ElementTree as ET


def collect(path):
    tree = ET.parse(path)
    joints = {}
    bodies = {}
    for body in tree.iter('body'):
        bname = body.get('name')
        for j in body.findall('joint'):
            name = j.get('name')
            joints[name] = {
                'axis': j.get('axis', '0 0 1'),
                'ref': j.get('ref', '0'),
                'range': j.get('range'),
                'parent_body': bname,
                'body_pos': body.get('pos', '0 0 0'),
                'body_quat': body.get('quat', '1 0 0 0'),
            }
        bodies[bname] = {'pos': body.get('pos', '0 0 0'),
                         'quat': body.get('quat', '1 0 0 0')}
    return joints, bodies


gj, gb = collect('gmr_x1_assets/x1.xml')
mj, mb = collect('X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml')

print(f"gmr xml: {len(gj)} joints, {len(gb)} bodies | mjcf: {len(mj)} joints, {len(mb)} bodies")
only_g = set(gj) - set(mj)
only_m = set(mj) - set(gj)
if only_g: print("only in gmr:", only_g)
if only_m: print("only in mjcf:", only_m)

print(f"\n{'joint':32s} {'axis gmr':16s} {'axis mjcf':16s} {'ref':>5s} {'bodypos g':20s} {'bodypos m':20s} {'bodyquat':s}")
n_axis = n_pos = 0
for name in [k for k in gj if k in mj]:
    g, m = gj[name], mj[name]
    ax_ok = g['axis'] == m['axis']
    bp_ok = g['body_pos'] == m['body_pos'] and g['body_quat'] == m['body_quat']
    if not ax_ok: n_axis += 1
    if not bp_ok: n_pos += 1
    flag = '' if (ax_ok and bp_ok) else '  <<<<'
    if flag:
        print(f"{name:32s} {g['axis']:16s} {m['axis']:16s} "
              f"{g['ref']:>5s}/{m['ref']:>4s} {g['body_pos']:20s} {m['body_pos']:20s}"
              f" q:{g['body_quat']}/{m['body_quat']}{flag}")
print(f"\nmismatched joints: axis={n_axis}, body-offset={n_pos}, total compared={len(set(gj)&set(mj))}")
