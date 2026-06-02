"""
Titan Bimanual Assembly — Full Pipeline (Stage 1 COMPLETE)
Fixed: NumPy Truth Value Array Crash & 4-DOF Kinematics.
"""
import os
import mujoco
import mujoco_viewer
import numpy as np
from tampas_core import Const
from mujoco_ik import MuJoCoIKSolver
from sensor_telemetry import SensorTelemetry
from reactive_fsm import ReactiveActionExecutor, ActionState

def load_titan_xml():
    filepath = "titan_v2.xml"
    if not os.path.exists(filepath):
        filepath = os.path.join("models", "titan_v2.xml")
    with open(filepath, "r") as f:
        return f.read()

HUMANOID_XML = load_titan_xml()

class WeldController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data
        self._welds = {
            ("left_arm", "block_A"):  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "left_grasp_A"),
            ("right_arm", "block_A"): mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "right_grasp_A"),
        }
    
    def activate_grasp(self, arm: str, obj: str) -> None:
        eq_id = self._welds.get((arm, obj))
        if eq_id is not None and eq_id >= 0:
            self.model.eq_data[eq_id, 3:6] = [0, 0, -0.08]
            self.model.eq_data[eq_id, 6:10] = [1, 0, 0, 0]
            self.data.eq_active[eq_id] = 1
            
    def release(self, arm: str, obj: str) -> None:
        eq_id = self._welds.get((arm, obj))
        if eq_id is not None and eq_id >= 0:
            self.data.eq_active[eq_id] = 0

def execute_schedule(schedule, actions, physical_values, model, data):
    print("▶ Launching MuJoCo Viewer: Task 2 Mid-Air Handover Sequence...")
    viewer = mujoco_viewer.MujocoViewer(model, data)
    telemetry = SensorTelemetry(model)
    weld_ctrl = WeldController(model, data)
    
    actuator_map = {
        "left_arm": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ["l_shoulder_p_mot", "l_shoulder_r_mot", "l_elbow_mot", "l_wrist_mot"]],
        "right_arm": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ["r_shoulder_p_mot", "r_shoulder_r_mot", "r_elbow_mot", "r_wrist_mot"]]
    }
    
    ft_map = {"left_arm": "l_sole_force", "right_arm": "r_sole_force"}
    
    executors = []
    for aid, t_start, t_end in schedule:
        action = actions[aid]
        if action.name == "move":
            arm_name = action.params[0].name
            q_start = physical_values.get(action.params[1])
            q_end   = physical_values.get(action.params[3])
            
            if q_start is not None and q_end is not None:
                if t_start == 0.0: data.ctrl[actuator_map[arm_name]] = q_start
                executor = ReactiveActionExecutor(
                    arm_name=arm_name, q_start=q_start, q_end=q_end,
                    t_start=t_start, t_end=t_end, 
                    actuator_ids=actuator_map[arm_name], ft_sensor=ft_map[arm_name]
                )
                executors.append(executor)

    render_skip = 10 
    executed_events = set()
    
    while viewer.is_alive:
        for _ in range(render_skip):
            current_time = data.time
            
            for schedule_idx, (aid, t_start, t_end) in enumerate(schedule):
                action = actions[aid]
                if current_time >= t_start and schedule_idx not in executed_events:
                    executed_events.add(schedule_idx)
                    
                    if action.name == "pick":
                        weld_ctrl.activate_grasp(action.params[0].name, action.params[1].name)
                    elif action.name == "handover":
                        weld_ctrl.release(action.params[0].name, action.params[2].name)
                        weld_ctrl.activate_grasp(action.params[1].name, action.params[2].name)
                    elif action.name == "place":
                        weld_ctrl.release(action.params[0].name, action.params[1].name)

            for ex in executors:
                ex.step(current_time, data, telemetry)
                    
            mujoco.mj_step(model, data)
        viewer.render()
    viewer.close()

class DummyAction:
    def __init__(self, name, params):
        self.name = name
        self.params = params

def main():
    print("▶ Initializing Titan Stage 1 Handover...")
    model = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    ik_L = MuJoCoIKSolver(model, data, "l_wrist_site", ["l_shoulder_pitch", "l_shoulder_roll", "l_elbow", "l_wrist_pitch"])
    ik_R = MuJoCoIKSolver(model, data, "r_wrist_site", ["r_shoulder_pitch", "r_shoulder_roll", "r_elbow", "r_wrist_pitch"])
    
    pos_A_start      = np.array([0.35, -0.20, 0.68])
    pos_center_left  = np.array([0.25, -0.12, 1.05])
    pos_center_right = np.array([0.25,  0.12, 1.05])
    pos_place_right  = np.array([0.40,  0.35, 0.70])
    
    ready_L = np.array([-0.5,  0.0, -0.5, 0.0])
    ready_R = np.array([-0.5, -0.0, -0.5, 0.0])
    
    # THE FIX: Bulletproof NumPy Array Truth Value Checks
    res_A = ik_L.solve(pos_A_start, q_init=ready_L)
    q_grasp_A = res_A if res_A is not None else ready_L
    
    res_CL = ik_L.solve(pos_center_left, q_init=q_grasp_A)
    q_center_L = res_CL if res_CL is not None else ready_L
    
    res_CR = ik_R.solve(pos_center_right, q_init=ready_R)
    q_center_R = res_CR if res_CR is not None else ready_R
    
    res_PR = ik_R.solve(pos_place_right, q_init=q_center_R)
    q_place_R = res_PR if res_PR is not None else ready_R

    physical_values = {
        "q_home_L": ready_L, "q_home_R": ready_R,
        "q_grasp_A": q_grasp_A, "q_center_L": q_center_L,
        "q_center_R": q_center_R, "q_place_R": q_place_R
    }

    left_arm  = Const("left_arm", "arm")
    right_arm = Const("right_arm", "arm")
    block_A   = Const("block_A", "obj")
    
    actions = [
        DummyAction("move",     [left_arm, "q_home_L", None, "q_grasp_A"]),   
        DummyAction("pick",     [left_arm, block_A]),                         
        DummyAction("move",     [left_arm, "q_grasp_A", None, "q_center_L"]), 
        DummyAction("move",     [right_arm, "q_home_R", None, "q_center_R"]), 
        DummyAction("handover", [left_arm, right_arm, block_A]),              
        DummyAction("move",     [right_arm, "q_center_R", None, "q_place_R"]),
        DummyAction("place",    [right_arm, block_A])                         
    ]
    
    schedule = [
        (0, 0.0, 1.5), (1, 1.5, 2.0), (2, 2.0, 3.5), 
        (3, 2.5, 4.0), (4, 4.0, 4.5), (5, 4.5, 6.0), (6, 6.0, 6.5)
    ]

    execute_schedule(schedule, actions, physical_values, model, data)

if __name__ == "__main__":
    main()