"""
sim_engine.py — Titan Humanoid V2 Simulation Engine
Day 3 → Phase 1 complete build.

Usage:
    python sim_engine.py [--mode suspend|walk]
"""
from __future__ import annotations
import argparse
import sys
import os
import numpy as np
import mujoco
import mujoco_viewer

# ── resolve model path relative to this file ──────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
MODEL_XML = os.path.join(_HERE, "models", "titan_v2.xml")

# ── controller imports ─────────────────────────────────────────────────
sys.path.insert(0, _HERE)
from controllers.cpg_gait import CPGGait


# ══════════════════════════════════════════════════════════════════════
#  INITIAL STATE SETUP
# ══════════════════════════════════════════════════════════════════════

def set_initial_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """
    Set freejoint root + joint angles for a stable initial stance.

    freejoint qpos layout (7 DOF): [x, y, z, qw, qx, qy, qz]
    Remaining qpos entries: one per hinge joint in tree order.

    z = 1.02 m places feet just above the floor at model-rest geometry.
    Knees slightly bent to avoid singularity at full extension.
    """
    mujoco.mj_resetData(model, data)

    # ── Root: upright, feet ~2mm above floor ──────────────────────────
    root_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qpos_adr = model.jnt_qposadr[root_jnt_id]

    data.qpos[root_qpos_adr + 0] = 0.0     # x
    data.qpos[root_qpos_adr + 1] = 0.0     # y
    data.qpos[root_qpos_adr + 2] = 1.02    # z (pelvis height)
    data.qpos[root_qpos_adr + 3] = 1.0     # qw  (upright quaternion)
    data.qpos[root_qpos_adr + 4] = 0.0     # qx
    data.qpos[root_qpos_adr + 5] = 0.0     # qy
    data.qpos[root_qpos_adr + 6] = 0.0     # qz

    # ── Joints: slight knee bend for stability ─────────────────────────
    def set_joint(name: str, angle_rad: float) -> None:
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr  = model.jnt_qposadr[jid]
        data.qpos[adr] = angle_rad

    set_joint("r_knee", np.radians(15))
    set_joint("l_knee", np.radians(15))
    set_joint("r_ankle_pitch", np.radians(-8))
    set_joint("l_ankle_pitch", np.radians(-8))
    set_joint("r_elbow", np.radians(-45))
    set_joint("l_elbow", np.radians(-45))

    # Sync derived quantities
    mujoco.mj_forward(model, data)


# ══════════════════════════════════════════════════════════════════════
#  TELEMETRY (printed to terminal every N steps)
# ══════════════════════════════════════════════════════════════════════

class Telemetry:
    PRINT_EVERY = 500   # sim steps

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model   = model
        self.data    = data
        self._step   = 0

        # Sensor address cache
        def _sadr(name: str) -> tuple[int, int]:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return model.sensor_adr[sid], model.sensor_dim[sid]

        self._r_force_adr, self._r_force_dim = _sadr("r_sole_force")
        self._l_force_adr, self._l_force_dim = _sadr("l_sole_force")
        self._gyro_adr,    self._gyro_dim    = _sadr("pelvis_gyro")
        self._quat_adr,    self._quat_dim    = _sadr("pelvis_quat")

        # Body ID for CoM
        self._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def step(self) -> None:
        self._step += 1
        if self._step % self.PRINT_EVERY != 0:
            return

        d, m = self.data, self.model

        rf = np.linalg.norm(d.sensordata[self._r_force_adr:self._r_force_adr + self._r_force_dim])
        lf = np.linalg.norm(d.sensordata[self._l_force_adr:self._l_force_adr + self._l_force_dim])
        gyr = d.sensordata[self._gyro_adr:self._gyro_adr + self._gyro_dim]
        quat = d.sensordata[self._quat_adr:self._quat_adr + self._quat_dim]

        com_z = d.subtree_com[self._pelvis_id, 2]

        # Check NaN / divergence
        nan_flag = "⚠️  NaN in qpos!" if np.any(np.isnan(d.qpos)) else "OK"

        print(
            f"[t={d.time:7.3f}s] CoM_z={com_z:.3f}m | "
            f"GRF R={rf:6.1f}N L={lf:6.1f}N | "
            f"ω=({gyr[0]:5.2f},{gyr[1]:5.2f},{gyr[2]:5.2f}) rad/s | "
            f"{nan_flag}"
        )

        # Divergence guard
        if np.any(np.isnan(d.qpos)) or com_z < 0.3:
            print("❌  Simulation diverged or robot fell. Check gains/timestep.")


# ══════════════════════════════════════════════════════════════════════
#  SUSPEND MODE — original kinematic test (no gravity, no freejoint)
# ══════════════════════════════════════════════════════════════════════

def run_suspend_mode() -> None:
    """
    Loads a version without freejoint for pure kinematic leg-swing testing.
    Reproduces original Day-3 behaviour with the V2 DOF set.
    """
    import tempfile, re

    with open(MODEL_XML) as f:
        xml = f.read()

    # Strip freejoint and set gravity to zero for suspended test
    xml = xml.replace('<freejoint name="root"/>', '')
    xml = re.sub(r'gravity="[^"]*"', 'gravity="0 0 0"', xml)
    # Move pelvis to fixed mid-air position
    xml = xml.replace('pos="0 0 1.02"', 'pos="0 0 1.2"')

    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    ctrl  = CPGGait(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    print("▶  SUSPEND MODE — kinematic leg swing test")
    while viewer.is_alive:
        ctrl.apply(data.time)
        mujoco.mj_step(model, data)
        viewer.render()
    viewer.close()


# ══════════════════════════════════════════════════════════════════════
#  WALK MODE — full dynamics, freejoint, ground contact
# ══════════════════════════════════════════════════════════════════════

def run_walk_mode() -> None:
    model  = mujoco.MjModel.from_xml_path(MODEL_XML)
    data   = mujoco.MjData(model)

    set_initial_pose(model, data)

    ctrl      = CPGGait(model, data)
    telemetry = Telemetry(model, data)
    viewer    = mujoco_viewer.MujocoViewer(model, data)

    print("▶  WALK MODE — full dynamics + CPG gait")
    print(f"   Model DOF: {model.nv}  |  Actuators: {model.nu}  |  Sensors: {model.nsensor}")

    # Brief settle phase: hold zero pose for 0.3s before gait starts
    SETTLE_TIME = 0.3

    while viewer.is_alive:
        t = data.time

        if t < SETTLE_TIME:
            # Hold neutral stance — zero all ctrl (spring/damper holds pose)
            data.ctrl[:] = 0.0
        else:
            ctrl.apply(t - SETTLE_TIME)

        mujoco.mj_step(model, data)
        telemetry.step()
        viewer.render()

    viewer.close()


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titan Humanoid V2 Simulation")
    parser.add_argument("--mode", choices=["suspend", "walk"], default="walk",
                        help="suspend = kinematic test (no gravity), walk = full dynamics")
    args = parser.parse_args()

    if args.mode == "suspend":
        run_suspend_mode()
    else:
        run_walk_mode()
