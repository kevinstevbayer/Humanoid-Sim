"""
Locomotion Core — Perfect Target Tracking & 5-DOF IK
Fixed: Integer-based array filling to prevent floating-point truncation.
"""
import numpy as np
import mujoco
from dataclasses import dataclass

@dataclass
class GaitParams:
    step_length: float = 0.15
    step_width: float = 0.20
    step_height: float = 0.06
    step_time: float = 0.6
    double_support: float = 0.2
    com_height: float = 0.75
    g: float = 9.81

class ZMPGaitGenerator:
    def __init__(self, params: GaitParams = GaitParams()):
        self.p = params
        self.omega = np.sqrt(self.p.g / self.p.com_height)
    
    def generate_walking_trajectory(self, n_steps: int, dt: float = 0.001):
        # Calculate exact number of array frames per phase
        N_ds = int(round(self.p.double_support / dt))
        N_ss = int(round(self.p.step_time / dt))
        N_step = N_ds + N_ss
        N_total = n_steps * N_step
        
        t = np.arange(N_total) * dt
        com_xy = np.zeros((N_total, 2))
        left_foot = np.zeros((N_total, 3))
        right_foot = np.zeros((N_total, 3))
        phase = np.zeros(N_total, dtype=int)
        
        FLOOR_Z = 0.02
        com_x = 0.0
        current_left_x = 0.0
        current_right_x = 0.0
        
        idx = 0
        for step in range(n_steps):
            is_final_step = (step == n_steps - 1)
            
            # --- DOUBLE SUPPORT ---
            for _ in range(N_ds):
                if idx >= N_total: break
                com_xy[idx, 0] = com_x
                phase[idx] = 0
                left_foot[idx] = [current_left_x, self.p.step_width/2, FLOOR_Z]
                right_foot[idx] = [current_right_x, -self.p.step_width/2, FLOOR_Z]
                idx += 1
                
            # --- SINGLE SUPPORT ---
            swing_foot = 'left' if step % 2 == 0 else 'right'
            
            # The CoM drives the feet
            target_com_x = com_x if is_final_step else com_x + self.p.step_length
            target_foot_x = target_com_x
            
            for i in range(N_ss):
                if idx >= N_total: break
                tau = i / float(N_ss - 1) if N_ss > 1 else 1.0
                
                # CoM advances linearly
                com_xy[idx, 0] = com_x + (target_com_x - com_x) * tau
                phase[idx] = 1 if swing_foot == 'left' else 2
                
                # Swing Foot Quintic Polynomial
                s = 10*tau**3 - 15*tau**4 + 6*tau**5
                
                if swing_foot == 'left':
                    left_foot[idx, 0] = current_left_x + (target_foot_x - current_left_x) * s
                    left_foot[idx, 1] = self.p.step_width/2
                    left_foot[idx, 2] = FLOOR_Z + self.p.step_height * np.sin(np.pi * tau)
                    
                    right_foot[idx] = [current_right_x, -self.p.step_width/2, FLOOR_Z]
                else:
                    right_foot[idx, 0] = current_right_x + (target_foot_x - current_right_x) * s
                    right_foot[idx, 1] = -self.p.step_width/2
                    right_foot[idx, 2] = FLOOR_Z + self.p.step_height * np.sin(np.pi * tau)
                    
                    left_foot[idx] = [current_left_x, self.p.step_width/2, FLOOR_Z]
                    
                idx += 1
                
            # Finalize positions for the next step
            com_x = target_com_x
            if swing_foot == 'left':
                current_left_x = target_foot_x
            else:
                current_right_x = target_foot_x
                
        return {'t': t, 'com_xy': com_xy, 'left_foot': left_foot, 'right_foot': right_foot, 'phase': phase}

class LegIKSolver:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 foot_site: str, leg_joints: list[str],
                 max_iters: int = 50, tol: float = 1e-3, damping: float = 0.01):
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, foot_site)
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in leg_joints]
        self.dof_ids = [model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.ndof = len(self.dof_ids)
        self.max_iters = max_iters
        self.tol = tol
        self.damping = damping
    
    def solve(self, target_pos: np.ndarray, q_init: np.ndarray | None = None) -> np.ndarray:
        saved_leg_q = np.array([self.data.qpos[d] for d in self.dof_ids])
        
        if q_init is not None:
            for i, dof in enumerate(self.dof_ids):
                self.data.qpos[dof] = q_init[i]
        
        mujoco.mj_kinematics(self.model, self.data) 
        
        best_err = float('inf')
        best_q = np.array([self.data.qpos[d] for d in self.dof_ids])
        
        for _ in range(self.max_iters):
            site_pos = self.data.site_xpos[self.site_id].copy()
            error = target_pos - site_pos
            err_norm = np.linalg.norm(error)
            
            if err_norm < best_err:
                best_err = err_norm
                best_q = np.array([self.data.qpos[d] for d in self.dof_ids])
            
            if err_norm < self.tol:
                break
            
            jac_pos = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, None, self.site_id)
            J = jac_pos[:, self.dof_ids]
            
            JJT = J @ J.T + self.damping * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            
            for i, dof in enumerate(self.dof_ids):
                jid = self.joint_ids[i]
                q_new = self.data.qpos[dof] + dq[i]
                q_min, q_max = self.model.jnt_range[jid]
                self.data.qpos[dof] = np.clip(q_new, q_min, q_max)
            
            mujoco.mj_kinematics(self.model, self.data)
        
        for i, dof in enumerate(self.dof_ids):
            self.data.qpos[dof] = saved_leg_q[i]
        mujoco.mj_kinematics(self.model, self.data)
        
        return best_q

class FullBodyIK:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.ik_left_leg = LegIKSolver(model, data, "l_sole",
            ["l_hip_pitch", "l_hip_roll", "l_knee", "l_ankle_pitch", "l_ankle_roll"])
        self.ik_right_leg = LegIKSolver(model, data, "r_sole",
            ["r_hip_pitch", "r_hip_roll", "r_knee", "r_ankle_pitch", "r_ankle_roll"])
    
    def solve_stance(self, left_foot: np.ndarray, right_foot: np.ndarray) -> dict:
        q_left = self.ik_left_leg.solve(left_foot)
        q_right = self.ik_right_leg.solve(right_foot)
        return {'left': q_left, 'right': q_right}