"""
TITAN HUMANOID V3 — Stage 2 Execution (Definitive WSL/Asus Build)
Features: Mechanical Grippers, Admittance Prep, Regex Injection, WSL OpenGL Bypass.
"""
import mujoco
import mujoco_viewer
import numpy as np
import re

from run_bimanual import HUMANOID_XML, DummyAction
from locomotion_core import ZMPGaitGenerator, GaitParams, FullBodyIK
from reactive_fsm import ReactiveActionExecutor
from sensor_telemetry import SensorTelemetry
from mujoco_ik import MuJoCoIKSolver
from tampas_core import Const
from robot_vision import TitanVisionBridge

# --- VLM BRIDGE IMPORT ---
try:
    from thinker_vlm import Thinker4B_Bridge
    VLM_AVAILABLE = True
except ImportError:
    print("\n[WARNING] Thinker-4B Bridge not found. Running in Blind Fallback Mode.\n")
    VLM_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2 ENVIRONMENT & MECHANICAL GRIPPER INJECTION
# ══════════════════════════════════════════════════════════════════════

STAGE_2_ENV_XML = """
    <body name="block_A" pos="0.60 0.15 0.65">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" rgba="0 1 0 1" mass="0.1" friction="1 0.005 0.0001"/>
    </body>
    <body name="block_B" pos="0.60 -0.15 0.65">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" rgba="1 0 0 1" mass="0.1" friction="1 0.005 0.0001"/>
    </body>

    <body name="grinder" pos="0.75 0.0 0.62">
      <geom type="cylinder" size="0.06 0.02" rgba="0.3 0.3 0.3 1" friction="1.5 0.005 0.0001"/>
      <site name="grinder_target" pos="0 0 0.025" size="0.01" rgba="1 1 0 1"/>
    </body>

    <body name="bin_green" pos="0.65 0.35 0.61">
      <geom type="box" size="0.06 0.06 0.005" rgba="0 0.8 0 0.4"/>
      <geom type="box" size="0.06 0.005 0.04" pos="0  0.06 0.04" rgba="0 0.8 0 0.4"/>
      <geom type="box" size="0.06 0.005 0.04" pos="0 -0.06 0.04" rgba="0 0.8 0 0.4"/>
      <geom type="box" size="0.005 0.06 0.04" pos=" 0.06 0 0.04" rgba="0 0.8 0 0.4"/>
      <geom type="box" size="0.005 0.06 0.04" pos="-0.06 0 0.04" rgba="0 0.8 0 0.4"/>
    </body>

    <body name="bin_red" pos="0.65 -0.35 0.61">
      <geom type="box" size="0.06 0.06 0.005" rgba="0.8 0 0 0.4"/>
      <geom type="box" size="0.06 0.005 0.04" pos="0  0.06 0.04" rgba="0.8 0 0 0.4"/>
      <geom type="box" size="0.06 0.005 0.04" pos="0 -0.06 0.04" rgba="0.8 0 0 0.4"/>
      <geom type="box" size="0.005 0.06 0.04" pos=" 0.06 0 0.04" rgba="0.8 0 0 0.4"/>
      <geom type="box" size="0.005 0.06 0.04" pos="-0.06 0 0.04" rgba="0.8 0 0 0.4"/>
    </body>
"""

LEFT_GRIPPER_XML = """
        <geom type="box" size="0.03 0.04 0.01" pos="0 0 -0.01" rgba="0.2 0.2 0.2 1"/>
        <body name="l_jaw1" pos="0.02 0 -0.02">
            <joint name="l_finger1" type="slide" axis="1 0 0" range="-0.025 0"/>
            <geom type="box" size="0.005 0.02 0.04" pos="0 0 -0.04" rgba="0.5 0.5 0.5 1" friction="2.0 0.1 0.001" solref="0.01 1" solimp="0.99 0.99 0.01"/>
        </body>
        <body name="l_jaw2" pos="-0.02 0 -0.02">
            <joint name="l_finger2" type="slide" axis="1 0 0" range="0 0.025"/>
            <geom type="box" size="0.005 0.02 0.04" pos="0 0 -0.04" rgba="0.5 0.5 0.5 1" friction="2.0 0.1 0.001" solref="0.01 1" solimp="0.99 0.99 0.01"/>
        </body>
"""

RIGHT_GRIPPER_XML = """
        <geom type="box" size="0.03 0.04 0.01" pos="0 0 -0.01" rgba="0.2 0.2 0.2 1"/>
        <body name="r_jaw1" pos="0.02 0 -0.02">
            <joint name="r_finger1" type="slide" axis="1 0 0" range="-0.025 0"/>
            <geom type="box" size="0.005 0.02 0.04" pos="0 0 -0.04" rgba="0.5 0.5 0.5 1" friction="2.0 0.1 0.001" solref="0.01 1" solimp="0.99 0.99 0.01"/>
        </body>
        <body name="r_jaw2" pos="-0.02 0 -0.02">
            <joint name="r_finger2" type="slide" axis="1 0 0" range="0 0.025"/>
            <geom type="box" size="0.005 0.02 0.04" pos="0 0 -0.04" rgba="0.5 0.5 0.5 1" friction="2.0 0.1 0.001" solref="0.01 1" solimp="0.99 0.99 0.01"/>
        </body>
"""

GRIPPER_MOTORS_XML = """
        <position name="l_f1_mot" joint="l_finger1" kp="5000"/>
        <position name="l_f2_mot" joint="l_finger2" kp="5000"/>
        <position name="r_f1_mot" joint="r_finger1" kp="5000"/>
        <position name="r_f2_mot" joint="r_finger2" kp="5000"/>
"""

# Base replacements (with RGB Camera injected into root)
COMPETITION_XML = HUMANOID_XML.replace(
    '<freejoint name="root"/>', 
    '<joint name="root_x" type="slide" axis="1 0 0" limited="false"/>\n      <camera name="rgb_cam" pos="0.1 0 0.5" euler="0 0.7 0" fovy="60"/>'
).replace(
    'pos="0 0 1.02"', 
    'pos="0 0 0.93"'
).replace(
    '</actuator>',
    '    <position name="root_x_motor" joint="root_x" kp="50000"/>\n' + GRIPPER_MOTORS_XML + '\n  </actuator>'
).replace(
    '</worldbody>',
    STAGE_2_ENV_XML + '\n  </worldbody>'
)

# Robust Regex Injection for Grippers (avoids XML mismatch crashes)
COMPETITION_XML = re.sub(r'(<site name="l_wrist_site"[^>]*>)', r'\1\n' + LEFT_GRIPPER_XML, COMPETITION_XML)
COMPETITION_XML = re.sub(r'(<site name="r_wrist_site"[^>]*>)', r'\1\n' + RIGHT_GRIPPER_XML, COMPETITION_XML)

# Nuke old yellow stumps
COMPETITION_XML = re.sub(r'<geom[^>]*rgba="1 1 0 1"[^>]*>', '', COMPETITION_XML)


# ══════════════════════════════════════════════════════════════════════
#  CONTROLLERS
# ══════════════════════════════════════════════════════════════════════

class PhysicalGripperController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.data = data
        self.l_f1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "l_f1_mot")
        self.l_f2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "l_f2_mot")
        self.r_f1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r_f1_mot")
        self.r_f2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r_f2_mot")

        self.open_width = 0.020 
        self.close_width = 0.010 
        
        if self.l_f1 != -1:
            self.data.ctrl[self.l_f1] = -self.open_width
            self.data.ctrl[self.l_f2] = self.open_width
            self.data.ctrl[self.r_f1] = -self.open_width
            self.data.ctrl[self.r_f2] = self.open_width

    def activate_grasp(self, arm_name: str):
        if arm_name == "left_arm" and self.l_f1 != -1:
            self.data.ctrl[self.l_f1] = -self.close_width
            self.data.ctrl[self.l_f2] = self.close_width
        elif arm_name == "right_arm" and self.r_f1 != -1:
            self.data.ctrl[self.r_f1] = -self.close_width
            self.data.ctrl[self.r_f2] = self.close_width

    def release(self, arm_name: str):
        if arm_name == "left_arm" and self.l_f1 != -1:
            self.data.ctrl[self.l_f1] = -self.open_width
            self.data.ctrl[self.l_f2] = self.open_width
        elif arm_name == "right_arm" and self.r_f1 != -1:
            self.data.ctrl[self.r_f1] = -self.open_width
            self.data.ctrl[self.r_f2] = self.open_width

class AdmittanceController:
    """Pre-built for Stage 2 Grinding."""
    def __init__(self, stiffness=np.array([0.0005, 0.0005, 0.001])):
        self.Kp = stiffness 
        self.deadband = 3.0
        
    def step(self, target_pos, force_reading):
        delta_pos = np.zeros(3)
        for i in range(3):
            if abs(force_reading[i]) > self.deadband:
                raw_yield = force_reading[i] * self.Kp[i]
                delta_pos[i] = np.clip(raw_yield, -0.01, 0.01) 
        return target_pos + delta_pos


def get_actuator_ids(model, joint_names):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in joint_names]
    if -1 in ids: raise ValueError(f"CRITICAL ERROR: Found -1 for {joint_names}")
    return ids

def set_initial_pose(model: mujoco.MjModel, data: mujoco.MjData, leg_ik: FullBodyIK):
    smart_guess = np.array([-0.2, 0.0, 0.4, -0.2, 0.0])
    for i, dof in enumerate(leg_ik.ik_left_leg.dof_ids):
        data.qpos[dof] = smart_guess[i]
        data.qpos[leg_ik.ik_right_leg.dof_ids[i]] = smart_guess[i]
    mujoco.mj_kinematics(model, data)


# ══════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

def main():
    print("▶ Initializing Titan Humanoid V3 Stage 2 Sequence...")
    model = mujoco.MjModel.from_xml_string(COMPETITION_XML)
    
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    block_A_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block_A")
    block_B_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block_B")
    
    if table_id != -1: model.body_pos[table_id] = [0.85, 0.0, 0.60]
    if block_A_id != -1: model.body_pos[block_A_id] = [0.60, 0.15, 0.65]
    if block_B_id != -1: model.body_pos[block_B_id] = [0.60, -0.15, 0.65]
    
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    root_motor_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "root_x_motor")
    root_joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")

    actuator_L_leg = get_actuator_ids(model, ["l_hip_pitch_mot", "l_hip_roll_mot", "l_knee_mot", "l_ankle_pitch_mot", "l_ankle_roll_mot"])
    actuator_R_leg = get_actuator_ids(model, ["r_hip_pitch_mot", "r_hip_roll_mot", "r_knee_mot", "r_ankle_pitch_mot", "r_ankle_roll_mot"])
    actuator_L_arm = get_actuator_ids(model, ["l_shoulder_p_mot", "l_shoulder_r_mot", "l_elbow_mot", "l_wrist_mot"])
    actuator_R_arm = get_actuator_ids(model, ["r_shoulder_p_mot", "r_shoulder_r_mot", "r_elbow_mot", "r_wrist_mot"])

    # SUBSYSTEM BOOTUP
    telemetry = SensorTelemetry(model)
    gripper_ctrl = PhysicalGripperController(model, data)
    leg_ik = FullBodyIK(model, data)
    
    # THE FIX: Only boot the OpenGL off-screen renderer if we actually need the VLM (WSL Safe)
    if VLM_AVAILABLE:
        vision = TitanVisionBridge(model, "rgb_cam", width=640, height=480)
        vlm_brain = Thinker4B_Bridge()
    else:
        vision = None
        
    ik_L = MuJoCoIKSolver(model, data, "l_wrist_site", ["l_shoulder_pitch", "l_shoulder_roll", "l_elbow", "l_wrist_pitch"])
    ik_R = MuJoCoIKSolver(model, data, "r_wrist_site", ["r_shoulder_pitch", "r_shoulder_roll", "r_elbow", "r_wrist_pitch"])

    set_initial_pose(model, data, leg_ik)

    print("▶ Planning ZMP Locomotion...")
    gait_params = GaitParams(step_length=0.15, step_height=0.06, step_time=0.6, step_width=0.20, com_height=0.93)
    zmp = ZMPGaitGenerator(gait_params)
    walk_traj = zmp.generate_walking_trajectory(n_steps=3, dt=model.opt.timestep)
    
    total_walk_steps = len(walk_traj['t'])
    walk_duration = walk_traj['t'][-1]
    final_x = walk_traj['com_xy'][-1][0]

    # Pre-solve setup
    data.qpos[root_joint_idx] = final_x
    mujoco.mj_forward(model, data)
    
    ready_pose = np.array([0.0, 0.0, -0.2, 0.0])
    walk_arm_pose = np.array([0.0, 0.0, -0.2, 0.0]) 
    reach_guess_L = np.array([-0.5, 0.0, -0.5, 0.0])
    reach_guess_R = np.array([-0.5, 0.0, -0.5, 0.0])

    print("▶ Launching Live Simulation...")
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    PHASE_WALK = 0
    PHASE_VISION = 1
    PHASE_PLANNING = 2
    PHASE_MANIPULATE = 3
    current_phase = PHASE_WALK
    
    walk_idx = 0
    executed_events = set()
    q_legs_frozen = {'left': np.zeros(5), 'right': np.zeros(5)}
    
    dynamic_schedule = []
    dynamic_executors = []

    while viewer.is_alive:
        for _ in range(10): 
            current_time = data.time
            
            # --- PHASE 0: WALK ---
            if current_phase == PHASE_WALK:
                if walk_idx < total_walk_steps:
                    target_com_x = walk_traj['com_xy'][walk_idx][0]
                    data.ctrl[root_motor_idx] = target_com_x
                    
                    actual_root_x = data.qpos[root_joint_idx]
                    data.qpos[root_joint_idx] = target_com_x
                    mujoco.mj_kinematics(model, data)
                    
                    target_L = walk_traj['left_foot'][walk_idx]
                    target_R = walk_traj['right_foot'][walk_idx]
                    q_targets = leg_ik.solve_stance(target_L, target_R)
                    
                    data.qpos[root_joint_idx] = actual_root_x
                    mujoco.mj_kinematics(model, data)
                    
                    if q_targets:
                        data.ctrl[actuator_L_leg] = q_targets['left']
                        data.ctrl[actuator_R_leg] = q_targets['right']
                        q_legs_frozen = q_targets 
                    
                    data.ctrl[actuator_L_arm] = walk_arm_pose
                    data.ctrl[actuator_R_arm] = walk_arm_pose
                    walk_idx += 1
                else:
                    print(f"\n[{current_time:.2f}s] ═══ WALK COMPLETE ═══")
                    current_phase = PHASE_VISION
            
            # --- PHASE 1: VLM VISION ---        
            elif current_phase == PHASE_VISION:
                pos_A_world = np.array([0.60, 0.15, 0.65]) 
                
                if VLM_AVAILABLE and vision is not None:
                    img = vision.capture_frame(data)
                    print(f"[{current_time:.2f}s] ✓ OPTICAL BRIDGE: Captured {img.shape} RGB Frame")
                    cam_pos, cam_mat = vision.get_camera_matrix(data)
                    
                    print(f"[{current_time:.2f}s] ▶ Querying Thinker-4B VLM...")
                    pos_A_cam = vlm_brain.predict_pose(img, "Find the green block.")
                    if pos_A_cam is not None:
                        pos_A_world = cam_pos + cam_mat @ pos_A_cam
                        print(f"[{current_time:.2f}s] ✓ VLM TARGET ACQUIRED: {pos_A_world}")
                    else:
                        print(f"[{current_time:.2f}s] ✗ VLM FAILED. Using blind fallback.")
                else:
                    print(f"[{current_time:.2f}s] ▶ Optical Bridge Bypassed (WSL Compatibility). Using hardcoded fallback.")
                
                current_phase = PHASE_PLANNING
            
            # --- PHASE 2: DYNAMIC PLANNING ---
            elif current_phase == PHASE_PLANNING:
                print(f"[{current_time:.2f}s] ▶ Generating Dynamic IK Targets...")
                
                pos_center_left = np.array([final_x + 0.20, -0.05, 0.85])
                pos_center_right = np.array([final_x + 0.20, 0.05, 0.85])
                pos_grinder = np.array([0.70, 0.0, 0.65]) 
                
                res_A = ik_L.solve(pos_A_world, q_init=reach_guess_L)
                q_grasp_A = res_A if res_A is not None else reach_guess_L
                
                res_CL = ik_L.solve(pos_center_left, q_init=q_grasp_A)
                q_center_L = res_CL if res_CL is not None else reach_guess_L
                
                res_CR = ik_R.solve(pos_center_right, q_init=reach_guess_R)
                q_center_R = res_CR if res_CR is not None else reach_guess_R
                
                res_GR = ik_R.solve(pos_grinder, q_init=q_center_R)
                q_grind_R = res_GR if res_GR is not None else reach_guess_R
                
                left_arm_c = Const("left_arm", "arm")
                right_arm_c = Const("right_arm", "arm")

                dynamic_actions = [
                    DummyAction("move",     [left_arm_c, "q_home_L", None, "q_grasp_A"]),
                    DummyAction("pick",     [left_arm_c]),
                    DummyAction("move",     [left_arm_c, "q_grasp_A", None, "q_center_L"]),
                    DummyAction("move",     [right_arm_c, "q_home_R", None, "q_center_R"]),
                    DummyAction("handover", [left_arm_c, right_arm_c]),
                    DummyAction("move",     [right_arm_c, "q_center_R", None, "q_grind_R"]),
                ]

                t_offset = current_time + 0.5
                dynamic_schedule = [
                    (0, t_offset + 0.0, t_offset + 1.5),
                    (1, t_offset + 1.5, t_offset + 2.0),
                    (2, t_offset + 2.0, t_offset + 3.5),
                    (3, t_offset + 2.5, t_offset + 4.0),
                    (4, t_offset + 4.0, t_offset + 5.0), 
                    (5, t_offset + 5.0, t_offset + 6.5),
                ]

                physical_values = {
                    "q_home_L": ready_pose, "q_home_R": ready_pose,
                    "q_grasp_A": q_grasp_A, "q_center_L": q_center_L,
                    "q_center_R": q_center_R, "q_grind_R": q_grind_R
                }

                ft_map = {"left_arm": "l_sole_force", "right_arm": "r_sole_force"}
                for aid, t_start, t_end in dynamic_schedule:
                    action = dynamic_actions[aid]
                    if action.name == "move":
                        arm_name = action.params[0].name
                        q_start = physical_values.get(action.params[1], ready_pose)
                        q_end = physical_values.get(action.params[3], ready_pose)
                        act_ids = actuator_L_arm if arm_name == "left_arm" else actuator_R_arm
                        
                        ex = ReactiveActionExecutor(
                            arm_name=arm_name, q_start=q_start, q_end=q_end, t_start=t_start, 
                            t_end=t_end, actuator_ids=act_ids, ft_sensor=ft_map[arm_name]
                        )
                        dynamic_executors.append(ex)
                
                print("  ✓ Sequence Locked. Commencing Execution.")
                current_phase = PHASE_MANIPULATE
            
            # --- PHASE 3: MANIPULATION ---
            elif current_phase == PHASE_MANIPULATE:
                data.ctrl[actuator_L_leg] = q_legs_frozen['left']
                data.ctrl[actuator_R_leg] = q_legs_frozen['right']
                
                for schedule_idx, (aid, t_start, t_end) in enumerate(dynamic_schedule):
                    action = dynamic_actions[aid]
                    if current_time >= t_start and schedule_idx not in executed_events:
                        executed_events.add(schedule_idx)
                        if action.name == "pick":
                            gripper_ctrl.activate_grasp(action.params[0].name)
                            print(f"[{current_time:.2f}s] ✓ PHYSICAL GRASP INITIATED")
                        elif action.name == "handover":
                            gripper_ctrl.activate_grasp(action.params[1].name)
                            gripper_ctrl.release(action.params[0].name)
                            print(f"[{current_time:.2f}s] ✓ HANDOVER TRANSFERRING")
                
                for ex in dynamic_executors:
                    ex.step(current_time, data, telemetry)
            
            mujoco.mj_step(model, data)
        viewer.render()
    
    viewer.close()

if __name__ == "__main__":
    main()