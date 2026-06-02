"""
cuRobo Integration Strategy — MuJoCo to NVIDIA cuRobo GPU Planner
Map: MuJoCo 6-DOF arm → cuRobo robot model for parallel collision-free motion
"""
import numpy as np
import mujoco
# from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig  # requires cuRobo install


# ══════════════════════════════════════════════════════════════════════
#  STEP 1: Extract MuJoCo Arm Kinematics
# ══════════════════════════════════════════════════════════════════════

def extract_arm_chain(model: mujoco.MjModel, arm_joints: list[str]) -> dict:
    """
    Extract DH parameters or URDF-equivalent from MuJoCo model.
    cuRobo expects: link lengths, joint axes, parent frames.
    """
    chain = []
    for jname in arm_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        body_id = model.jnt_bodyid[jid]
        
        # Get joint axis and parent transform
        axis = model.jnt_axis[jid]
        pos = model.body_pos[body_id]  # local offset
        
        chain.append({
            "name": jname,
            "axis": axis,
            "offset": pos,
            "range": model.jnt_range[jid],
        })
    return {"joints": chain, "dof": len(arm_joints)}


# ══════════════════════════════════════════════════════════════════════
#  STEP 2: Build cuRobo World Representation
# ══════════════════════════════════════════════════════════════════════

def build_curobo_obstacles(model: mujoco.MjModel, data: mujoco.MjData) -> list:
    """
    Convert MuJoCo geoms (table, floor, etc.) to cuRobo spheres/boxes.
    cuRobo collision: sphere-sphere and sphere-mesh (GPU-accelerated).
    """
    obstacles = []
    for i in range(model.ngeom):
        geom_type = model.geom_type[i]
        geom_size = model.geom_size[i]
        geom_pos  = data.geom_xpos[i]
        
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            obstacles.append({
                "type": "box",
                "pose": geom_pos,
                "dims": geom_size,
            })
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            obstacles.append({
                "type": "sphere",
                "center": geom_pos,
                "radius": geom_size[0],
            })
    return obstacles


# ══════════════════════════════════════════════════════════════════════
#  STEP 3: cuRobo Motion Generation API (Pseudocode)
# ══════════════════════════════════════════════════════════════════════

"""
# Initialize cuRobo planner
config = MotionGenConfig.from_dict({
    "robot": "custom_6dof",  # Load from extracted chain
    "world_model": build_curobo_obstacles(model, data),
    "interpolation_dt": 0.01,
})
planner = MotionGen(config)

# Batch parallel planning for both arms
targets = {
    "left_arm":  target_pose_L,   # (7,) [x,y,z,qw,qx,qy,qz]
    "right_arm": target_pose_R,
}
result = planner.plan_batch(
    start_state=current_q,
    goal_poses=targets,
    num_seeds=50,  # GPU parallelizes 50 random seeds
)

# Extract best collision-free trajectory
trajectory = result.get_paths()[0]  # (T, 6) joint path
"""


# ══════════════════════════════════════════════════════════════════════
#  CRITICAL MAPPING: MuJoCo qpos ↔ cuRobo State
# ══════════════════════════════════════════════════════════════════════

def mujoco_to_curobo_state(data: mujoco.MjData, arm_dof_ids: list[int]) -> np.ndarray:
    """Extract current joint state for cuRobo planner."""
    return np.array([data.qpos[d] for d in arm_dof_ids])

def curobo_to_mujoco_ctrl(traj: np.ndarray, arm_actuator_ids: list[int], 
                           data: mujoco.MjData, t_idx: int) -> None:
    """Write cuRobo trajectory point to MuJoCo ctrl."""
    for i, act_id in enumerate(arm_actuator_ids):
        data.ctrl[act_id] = traj[t_idx, i]