#!/usr/bin/env python3
"""v57 reward CALL-LEVEL dry-run (mock env, correct tensor shapes).

Covers the two new/reworked callables:
  heel_first_stance — rising-edge detection across calls, lead
    classification (+1.0 heel-first / +0.3 flat / 0 toe-first), walk gate,
    per-foot independence.
  stance_sole_flat_walk (v57 phased) — foot-flat frames penalized,
    heel-only (heel-strike) & toe-only (push-off) frames EXEMPT, walk
    gate preserved, neither-sensed falls back to legacy penalty.
No isaaclab needed (same mock-injection pattern as dryrun_reward_v56).
"""
import math
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


class _Math(types.SimpleNamespace):
    @staticmethod
    def quat_apply(q, v):
        # interpret (w,x,y,z); rotate v (identity q => v)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        v0, v1, v2 = v[:, 0], v[:, 1], v[:, 2]
        # v' = v + 2*cross(q.xyz, v)*w + 2*cross(q.xyz, cross(q.xyz, v))
        cx = y * v2 - z * v1
        cy = z * v0 - x * v2
        cz = x * v1 - y * v0
        t0 = v0 + 2 * (w * cx + (y * cz - z * cy))
        t1 = v1 + 2 * (w * cy + (z * cx - x * cz))
        t2 = v2 + 2 * (w * cz + (x * cy - y * cx))
        return torch.stack([t0, t1, t2], dim=1)


def _install_mocks():
    isaaclab = types.ModuleType("isaaclab")
    envs = types.ModuleType("isaaclab.envs")
    mdp_mod = types.ModuleType("isaaclab.envs.mdp")
    managers = types.ModuleType("isaaclab.managers")
    sensors = types.ModuleType("isaaclab.sensors")
    assets = types.ModuleType("isaaclab.assets")
    utils = types.ModuleType("isaaclab.utils")
    math_mod = types.ModuleType("isaaclab.utils.math")

    class SceneEntityCfg(types.SimpleNamespace):
        def __init__(self, name, joint_names=None, body_names=None,
                     preserve_order=False):
            super().__init__(name=name, joint_names=joint_names,
                             body_names=body_names,
                             preserve_order=preserve_order,
                             joint_ids=None, body_ids=None)

    envs.mdp = mdp_mod

    class ManagerBasedRLEnv:
        pass

    envs.ManagerBasedRLEnv = ManagerBasedRLEnv
    isaaclab.envs = envs
    managers.SceneEntityCfg = SceneEntityCfg
    isaaclab.managers = managers
    sensors.ContactSensor = object
    sensors.RayCaster = object
    isaaclab.sensors = sensors
    assets.Articulation = object
    assets.RigidObject = object
    isaaclab.assets = assets
    math_mod.quat_apply = _Math.quat_apply
    utils.math = math_mod
    isaaclab.utils = utils
    sys.modules.update({
        "isaaclab": isaaclab, "isaaclab.envs": envs,
        "isaaclab.envs.mdp": mdp_mod, "isaaclab.managers": managers,
        "isaaclab.sensors": sensors, "isaaclab.assets": assets,
        "isaaclab.utils": utils, "isaaclab.utils.math": math_mod,
    })


_install_mocks()

import importlib.util as _ilu  # noqa: E402
_rp = (ROOT / "roboparty_train" / "robolab" / "robolab" / "tasks"
       / "manager_based" / "amp" / "mdp" / "rewards.py").resolve()
_spec = _ilu.spec_from_file_location("amp_mdp_rewards_dryrun57", _rp)
R = _ilu.module_from_spec(_spec)
sys.modules["amp_mdp_rewards_dryrun57"] = R
_spec.loader.exec_module(R)


class _CmdMgr:
    def __init__(self, cmd, n):
        row = torch.tensor(cmd, dtype=torch.float32)
        self._c = row.unsqueeze(0).expand(n, 3).contiguous()

    def get_command(self, name):
        return self._c


def qmul(a, b):
    """Hamilton product of (w,x,y,z) quats (numpy lists -> tensor-ready)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2]


def qy(deg):
    r = math.radians(deg) / 2
    return [math.cos(r), 0.0, math.sin(r), 0.0]


# vendor neutral: local x->world y, y->world z (sole normal UP),
# z->world x (heel end forward) — the (0.5,0.5,0.5,0.5) cyclic quat
Q_NEUTRAL = [0.5, 0.5, 0.5, 0.5]


def make_env(n=4, contact=(True, True), cmd=(1.0, 0.0, 0.0),
             pitch_deg=(0.0, 0.0), mid_z=0.016):
    """Mock env with the VENDOR foot frame. pitch_deg > 0 = heel end DOWN
    (heel-strike). Foot ends: heel_z = mid_z - 0.07 sin(pitch),
    toe_z = mid_z + 0.07 sin(pitch); sole normal world z = cos(pitch)."""
    ankle_ids = [2, 3]
    nf = torch.zeros(n, 2, 6, 3)
    for f, on in enumerate(contact):
        nf[:, :, ankle_ids[f], 2] = 5.0 if on else 0.0
    sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(net_forces_w_history=nf))
    bq = torch.zeros(n, 6, 4)
    bp = torch.zeros(n, 6, 3)
    for f in range(2):
        q = qmul(qy(pitch_deg[f]), Q_NEUTRAL)
        bq[:, ankle_ids[f]] = torch.tensor(q)
        bp[:, ankle_ids[f], 2] = mid_z
    jp = torch.zeros(n, 29)
    asset = types.SimpleNamespace(data=types.SimpleNamespace(
        joint_pos=jp, default_joint_pos=torch.zeros(n, 29),
        joint_vel=torch.zeros(n, 29), joint_acc=torch.zeros(n, 29),
        body_quat_w=bq, body_pos_w=bp))
    scene = {"contact_forces": sensor, "robot": asset}
    env = types.SimpleNamespace(
        scene=scene, num_envs=n, device="cpu", step_dt=0.02,
        command_manager=_CmdMgr(cmd, n),
        action_manager=types.SimpleNamespace(
            action=torch.zeros(n, 29), prev_action=torch.zeros(n, 29)),
        termination_manager=types.SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool)),
    )
    return env, ankle_ids


def foot_cfgs(ankle_ids):
    sc = types.SimpleNamespace(name="contact_forces", body_ids=ankle_ids)
    ac = types.SimpleNamespace(name="robot", body_ids=ankle_ids)
    return sc, ac


def call_hf(env, sc, ac):
    return R.heel_first_stance(env, sc, ac, command_name="base_velocity",
                               max_cmd_speed=1.5, touch_z=0.010, lead_z=0.002)


def call_sf(env, sc, ac):
    return R.stance_sole_flat_walk(env, sc, ac, command_name="base_velocity",
                                   max_cmd_speed=1.5)


def main():
    ok = []

    def check(label, got, want, tol=1e-6):
        good = abs(float(got) - float(want)) <= tol
        ok.append(good)
        print(f"  {label:52s} r={float(got):+.4f} expect~{float(want):+.4f} "
              f"{'OK' if good else 'MISMATCH'}")

    # ---------------- heel_first_stance ------------------------------
    print("[dryrun] heel_first_stance:")
    # fresh state key per scenario: bump id(env) via a counter object
    # heel-strike: pitch +10deg -> heel = mid-12.1mm, toe = mid+12.1mm.
    # mid 16mm: heel 3.9mm (<=10 ON), toe 28.1mm; lead +24.2mm -> +1.0
    R._EDGE_STATE.clear()
    env2, ai2 = make_env(contact=(True, False), pitch_deg=(10.0, 0.0), mid_z=0.016)
    sc, ac = foot_cfgs(ai2)
    check("heel-strike edge (L +10° pitch, L-only contact)",
          call_hf(env2, sc, ac)[0], 1.0)
    check("second call, no new edge -> 0", call_hf(env2, sc, ac)[0], 0.0)
    # flat landing: pitch 0, both ends at mid=4mm <=10mm, lead 0 ->
    # ramp (0+15)/19 = 0.789 per foot (v57b continuous ramp)
    env3, ai3 = make_env(contact=(True, True), pitch_deg=(0.0, 0.0), mid_z=0.004)
    sc3, ac3 = foot_cfgs(ai3)
    check("flat landing both feet (ramp 35/39 x2)",
          call_hf(env3, sc3, ac3)[0], 2*(0.035/0.039))
    # shallow toe-first (the v56 policy population): pitch -5deg, mid 2mm
    # -> heel 8.1mm ON, lead -12.2mm -> score (2.8/19)=0.147 x2
    env3b, ai3b = make_env(contact=(True, True), pitch_deg=(-5.0, -5.0), mid_z=0.002)
    sc3b, ac3b = foot_cfgs(ai3b)
    # exact: lead = 2*0.07*sin(5°) = 12.208mm; score=(lead+15mm)/19mm
    lead_exact = 2*0.07*math.sin(math.radians(5))
    check("shallow toe-first graded (lead exact)",
          call_hf(env3b, sc3b, ac3b)[0], 2*((0.035-lead_exact)/0.039), 1e-4)
    # deep toe-first: pitch -12deg, mid 12mm -> lead = -2*0.07*sin(12°)
    # = -29.1mm < -15mm -> 0 (ramp floor)
    env4, ai4 = make_env(contact=(True, True), pitch_deg=(-12.0, -12.0), mid_z=0.012)
    sc4, ac4 = foot_cfgs(ai4)
    lead_deep = 2*0.07*math.sin(math.radians(12))
    check("deep toe-first (lead -29mm) graded low (v57c)",
          call_hf(env4, sc4, ac4)[0], 2*((0.035-lead_deep)/0.039), 1e-4)
    # v57c REGRESSION: shallow toe-first with the heel HIGH (mid 20mm,
    # -5deg -> heel 26mm, toe 14mm, lead -12.2mm) — the old near-ground
    # gate (heel<=10mm) killed exactly this population (the -10mm lead
    # toe-first landings v56 policies actually produce); must now grade
    lead_exact2 = 2*0.07*math.sin(math.radians(5))
    env4b, ai4b = make_env(contact=(True, True), pitch_deg=(-5.0, -5.0), mid_z=0.020)
    sc4b, ac4b = foot_cfgs(ai4b)
    check("shallow toe-first heel-high (v57c gate removal) graded",
          call_hf(env4b, sc4b, ac4b)[0], 2*((0.035-lead_exact2)/0.039), 1e-4)
    # jog gate
    env5, ai5 = make_env(contact=(True, True), pitch_deg=(10.0, 10.0),
                         mid_z=0.016, cmd=(2.5, 0.0, 0.0))
    sc5, ac5 = foot_cfgs(ai5)
    check("jog cmd 2.5 -> walk gate OFF", call_hf(env5, sc5, ac5)[0], 0.0)

    # ---------------- stance_sole_flat_walk (phased) ------------------
    print("[dryrun] stance_sole_flat_walk (v57 phased):")
    # foot-flat: pitch 4deg, mid 2mm -> heel -2.9mm, toe 6.9mm both ON
    # (<=12mm) -> sin^2(4°) = 0.0049 per foot
    envf, aif = make_env(contact=(True, True), pitch_deg=(4.0, 4.0), mid_z=0.002)
    scf, acf = foot_cfgs(aif)
    sf = call_sf(envf, scf, acf)
    want = 2 * (math.sin(math.radians(4)) ** 2)
    check("foot-flat phase (both ends down, 4° tilt) penalized", sf[0], want, 1e-4)
    # heel-strike phase: pitch +10deg, mid 6mm -> heel -6.1mm ON, toe
    # 18.1mm OFF => EXEMPT (v27 version would tax sin^2(10°)=0.030/foot)
    envh, aih = make_env(contact=(True, True), pitch_deg=(10.0, 10.0), mid_z=0.006)
    sch, ach = foot_cfgs(aih)
    check("heel-strike phase (heel-only) EXEMPT", call_sf(envh, sch, ach)[0], 0.0)
    # push-off: pitch -10deg, mid 6mm -> toe -6.1mm ON, heel 18.1mm OFF
    envp, aip = make_env(contact=(True, True), pitch_deg=(-10.0, -10.0), mid_z=0.006)
    scp, acp = foot_cfgs(aip)
    check("push-off phase (toe-only) EXEMPT", call_sf(envp, scp, acp)[0], 0.0)
    # swing (no contact): 0
    envs_, ais = make_env(contact=(False, False), pitch_deg=(20.0, 20.0), mid_z=0.05)
    scs, acs = foot_cfgs(ais)
    check("swing -> 0", call_sf(envs_, scs, acs)[0], 0.0)
    # jog gate off
    envj, aij = make_env(contact=(True, True), pitch_deg=(4.0, 4.0), mid_z=0.002,
                         cmd=(2.0, 0.0, 0.0))
    scj, acj = foot_cfgs(aij)
    check("jog cmd -> gate OFF", call_sf(envj, scj, acj)[0], 0.0)

    # ---------------- terminal_swing_ankle_rate (v57d) ------------------
    print("[dryrun] terminal_swing_ankle_rate:")

    def call_af(env, sc, ac):
        return R.terminal_swing_ankle_rate(env, sc, ac,
                                           command_name="base_velocity",
                                           max_cmd_speed=1.5,
                                           height_z=0.040, rate_thr=0.70)

    # flick: +110deg/s (=1.92 rad/s) ankle vel, foot low (mid 20mm,
    # pitch 0 -> ends ±20mm <= 40mm), airborne -> excess (1.92-0.70)=1.22
    envA, aiA = make_env(contact=(False, False), pitch_deg=(0.0, 0.0), mid_z=0.020)
    envA.scene["robot"].data.joint_vel = torch.zeros(4, 29)
    envA.scene["robot"].data.joint_vel[:, 21] = math.radians(110)   # L ankle pitch
    envA.scene["robot"].data.joint_vel[:, 27] = math.radians(110)   # R ankle pitch
    scA, acA = types.SimpleNamespace(name="contact_forces", body_ids=aiA), \
        types.SimpleNamespace(name="robot", body_ids=aiA, joint_ids=[21, 27])
    check("flick +110deg/s low+air -> excess x2",
          call_af(envA, scA, acA)[0], 2*(math.radians(110)-0.70), 1e-4)
    # smooth reference-like approach: -18deg/s -> below thr, 0
    envB, aiB = make_env(contact=(False, False), pitch_deg=(0.0, 0.0), mid_z=0.020)
    envB.scene["robot"].data.joint_vel = torch.zeros(4, 29)
    envB.scene["robot"].data.joint_vel[:, 21] = math.radians(-18)
    envB.scene["robot"].data.joint_vel[:, 27] = math.radians(-18)
    scB, acB = types.SimpleNamespace(name="contact_forces", body_ids=aiB), \
        types.SimpleNamespace(name="robot", body_ids=aiB, joint_ids=[21, 27])
    check("dorsiflexion -18deg/s -> 0 (free)", call_af(envB, scB, acB)[0], 0.0)
    # high foot (clearance) with flick: exempt (only terminal-swing gated)
    envC, aiC = make_env(contact=(False, False), pitch_deg=(0.0, 0.0), mid_z=0.12)
    envC.scene["robot"].data.joint_vel = torch.zeros(4, 29)
    envC.scene["robot"].data.joint_vel[:, 21] = math.radians(110)
    envC.scene["robot"].data.joint_vel[:, 27] = math.radians(110)
    scC, acC = types.SimpleNamespace(name="contact_forces", body_ids=aiC), \
        types.SimpleNamespace(name="robot", body_ids=aiC, joint_ids=[21, 27])
    check("high-swing flick exempt (clearance zone)", call_af(envC, scC, acC)[0], 0.0)
    # in contact with flick rate: exempt (stance dynamics not the flick)
    envD, aiD = make_env(contact=(True, True), pitch_deg=(0.0, 0.0), mid_z=0.020)
    envD.scene["robot"].data.joint_vel = torch.zeros(4, 29)
    envD.scene["robot"].data.joint_vel[:, 21] = math.radians(110)
    envD.scene["robot"].data.joint_vel[:, 27] = math.radians(110)
    scD, acD = types.SimpleNamespace(name="contact_forces", body_ids=aiD), \
        types.SimpleNamespace(name="robot", body_ids=aiD, joint_ids=[21, 27])
    check("stance flick exempt", call_af(envD, scD, acD)[0], 0.0)
    # mild positive 45deg/s: excess (0.785-0.70)=0.085 x2
    envE, aiE = make_env(contact=(False, False), pitch_deg=(0.0, 0.0), mid_z=0.020)
    envE.scene["robot"].data.joint_vel = torch.zeros(4, 29)
    envE.scene["robot"].data.joint_vel[:, 21] = math.radians(45)
    envE.scene["robot"].data.joint_vel[:, 27] = math.radians(45)
    scE, acE = types.SimpleNamespace(name="contact_forces", body_ids=aiE), \
        types.SimpleNamespace(name="robot", body_ids=aiE, joint_ids=[21, 27])
    check("mild +45deg/s graded (0.085 x2)",
          call_af(envE, scE, acE)[0], 2*(math.radians(45)-0.70), 1e-4)

    # ---------------- heel_down_ready (v57e) ----------------------------
    print("[dryrun] heel_down_ready:")

    def call_hd(env, sc, ac):
        return R.heel_down_ready(env, sc, ac, command_name="base_velocity",
                                 max_cmd_speed=1.5, height_z=0.040)

    # READY posture: pitch +8deg (heel end DOWN in the vendor frame),
    # mid 15mm airborne -> heel=5.3mm toe=24.7mm; delta=toe-heel=+19.4mm
    # -> (19.4+35)/52 = 1.0 per foot
    envH1, aiH1 = make_env(contact=(False, False), pitch_deg=(8.0, 8.0), mid_z=0.015)
    scH1 = types.SimpleNamespace(name="contact_forces", body_ids=aiH1)
    acH1 = types.SimpleNamespace(name="robot", body_ids=aiH1)
    check("ready posture (heel down ~19mm) -> 1.0 x2",
          call_hd(envH1, scH1, acH1)[0], 2.0, 1e-4)
    # FLICK posture: pitch -8deg (toe down), mid 15mm -> delta=-19.4mm
    envH2, aiH2 = make_env(contact=(False, False), pitch_deg=(-8.0, -8.0), mid_z=0.015)
    scH2 = types.SimpleNamespace(name="contact_forces", body_ids=aiH2)
    acH2 = types.SimpleNamespace(name="robot", body_ids=aiH2)
    check("flick posture (toe down ~19mm) -> 0.30 x2",
          call_hd(envH2, scH2, acH2)[0],
          2 * ((0.035 - 2 * 0.07 * math.sin(math.radians(8))) / 0.052), 1e-4)
    envH3, aiH3 = make_env(contact=(False, False), pitch_deg=(-30.0, -30.0), mid_z=0.015)
    scH3 = types.SimpleNamespace(name="contact_forces", body_ids=aiH3)
    acH3 = types.SimpleNamespace(name="robot", body_ids=aiH3)
    check("deep flick (toe down ~70mm) -> 0", call_hd(envH3, scH3, acH3)[0], 0.0)
    envH4, aiH4 = make_env(contact=(True, True), pitch_deg=(8.0, 8.0), mid_z=0.015)
    scH4 = types.SimpleNamespace(name="contact_forces", body_ids=aiH4)
    acH4 = types.SimpleNamespace(name="robot", body_ids=aiH4)
    check("stance exempt", call_hd(envH4, scH4, acH4)[0], 0.0)
    envH5, aiH5 = make_env(contact=(False, False), pitch_deg=(8.0, 8.0), mid_z=0.12)
    scH5 = types.SimpleNamespace(name="contact_forces", body_ids=aiH5)
    acH5 = types.SimpleNamespace(name="robot", body_ids=aiH5)
    check("clearance zone exempt", call_hd(envH5, scH5, acH5)[0], 0.0)
    envH6, aiH6 = make_env(contact=(False, False), pitch_deg=(8.0, 8.0), mid_z=0.015,
                           cmd=(2.0, 0.0, 0.0))
    scH6 = types.SimpleNamespace(name="contact_forces", body_ids=aiH6)
    acH6 = types.SimpleNamespace(name="robot", body_ids=aiH6)
    check("jog gate OFF", call_hd(envH6, scH6, acH6)[0], 0.0)
    print(f"[dryrun] {'ALL PASS' if all(ok) else 'FAILURES'} ({sum(ok)}/{len(ok)})")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
