"""
Phase 1: Air-Walk Verification (Patched)
Testing ZMP Gait Generation and 12-DOF Leg IK without gravity falling.
"""
import mujoco
import mujoco_viewer
import numpy as np

from run_bimanual import HUMANOID_XML 
from locomotion_core import ZMPGaitGenerator, GaitParams, FullBodyIK

def main():
    print("▶ Initializing Titan V3 Air-Walk Test...")
    model = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # WIDEN THE STANCE to match the robot's physical hips (0.2m apart)
    gait_params = GaitParams(step_length=0.15, step_height=0.06, step_time=0.6, step_width=0.20)
    zmp = ZMPGaitGenerator(gait_params)
    leg_ik = FullBodyIK(model, data)

    print("▶ Generating ZMP Trajectory for 6 steps...")
    traj = zmp.generate_walking_trajectory(n_steps=6, dt=model.opt.timestep)
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    left_leg_actuators = [0, 1, 2, 3, 4, 5]
    right_leg_actuators = [6, 7, 8, 9, 10, 11]

    # THE FIX: Bring the "invisible treadmill" up to the robot's feet
    # Pelvis is at 1.0m, legs reach down to 0.15m. Set target to 0.20m for bent knees.
    Z_OFFSET = np.array([0, 0, 0.20])

    home_left = traj['left_foot'][0] + Z_OFFSET
    home_right = traj['right_foot'][0] + Z_OFFSET
    
    # Smart guess to prevent the knees from bending backward like a bird
    # THE FIX: Aggressive Humanoid Smart Guess
    # Hip pitched forward (-0.5), Knee deeply bent (1.2), Ankle compensating (-0.5)
    smart_guess_L = np.array([-0.5, 0.0, 0.0, 1.2, -0.5, 0.0]) 
    smart_guess_R = np.array([-0.5, 0.0, 0.0, 1.2, -0.5, 0.0])

    leg_ik.ik_left_leg.data.qpos[leg_ik.ik_left_leg.dof_ids] = smart_guess_L
    leg_ik.ik_right_leg.data.qpos[leg_ik.ik_right_leg.dof_ids] = smart_guess_R

    initial_q = leg_ik.solve_stance(home_left, home_right)
    
    if initial_q:
        data.ctrl[left_leg_actuators] = initial_q['left']
        data.ctrl[right_leg_actuators] = initial_q['right']
        mujoco.mj_step(model, data)

    print("▶ Starting Air-Walk Execution...")
    
    step_idx = 0
    total_steps = len(traj['t'])
    render_skip = 10 

    while viewer.is_alive and step_idx < total_steps:
        for _ in range(render_skip):
            if step_idx >= total_steps:
                break
                
            # Add the Z-offset to every frame of the trajectory
            target_L = traj['left_foot'][step_idx] + Z_OFFSET
            target_R = traj['right_foot'][step_idx] + Z_OFFSET
            
            q_targets = leg_ik.solve_stance(target_L, target_R)
            
            if q_targets:
                data.ctrl[left_leg_actuators] = q_targets['left']
                data.ctrl[right_leg_actuators] = q_targets['right']
                
            mujoco.mj_step(model, data)
            step_idx += 1
            
        viewer.render()

    print("▶ Sequence Complete.")
    viewer.close()

if __name__ == "__main__":
    main()