"""
Native MuJoCo IK Solver — Damped Least Squares Jacobian Pseudoinverse
Fixed: Swapped mj_forward for mj_kinematics to prevent QACC explosions.
"""
import numpy as np
import mujoco

class MuJoCoIKSolver:
    """
    Iterative IK using Jacobian transpose or pseudoinverse.
    Targets position only (not orientation).
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 site_name: str, joint_names: list[str],
                 max_iters: int = 100, tol: float = 1e-3, damping: float = 0.01):
        self.model = model
        self.data  = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        
        # Joint indices for this arm
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            for jname in joint_names
        ]
        self.dof_ids = [model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.ndof = len(self.dof_ids)
        
        self.max_iters = max_iters
        self.tol = tol
        self.damping = damping

    def solve(self, target_pos: np.ndarray, q_init: np.ndarray | None = None) -> np.ndarray | None:
        """
        Solve IK for target_pos (x,y,z).
        Returns joint config (ndof,) or None if failed.
        """
        # Initialize at current config or provided init
        if q_init is not None:
            for i, dof in enumerate(self.dof_ids):
                self.data.qpos[dof] = q_init[i]
        
        # CRITICAL FIX: Use mj_kinematics instead of mj_forward!
        # This updates spatial positions without calculating collision dynamics.
        mujoco.mj_kinematics(self.model, self.data)
        
        for _ in range(self.max_iters):
            # Current end-effector position
            site_pos = self.data.site_xpos[self.site_id].copy()
            error = target_pos - site_pos
            
            if np.linalg.norm(error) < self.tol:
                # Success
                return np.array([self.data.qpos[d] for d in self.dof_ids])
            
            # Compute Jacobian (position only, 3×ndof)
            jac_pos = np.zeros((3, self.model.nv))
            # We don't need rotational jacobian for position-only IK, pass None to save compute
            mujoco.mj_jacSite(self.model, self.data, jac_pos, None, self.site_id)
            
            # Extract columns for this arm's DOFs
            J = jac_pos[:, self.dof_ids]
            
            # Damped least squares: Δq = J^T (JJ^T + λI)^-1 error
            JJT = J @ J.T + self.damping * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            
            # Update qpos
            for i, dof in enumerate(self.dof_ids):
                self.data.qpos[dof] += dq[i]
                
                # Clamp the mathematical solver to physical joint limits
                jnt_id = self.joint_ids[i]
                if self.model.jnt_limited[jnt_id]:
                    limit_min = self.model.jnt_range[jnt_id, 0]
                    limit_max = self.model.jnt_range[jnt_id, 1]
                    self.data.qpos[dof] = np.clip(self.data.qpos[dof], limit_min, limit_max)
            
            # CRITICAL FIX: Update kinematics for the next iteration
            mujoco.mj_kinematics(self.model, self.data)
        
        # Failed to converge
        return None

    def solve_with_restarts(self, target_pos: np.ndarray, n_restarts: int = 3) -> np.ndarray | None:
        """Try multiple random initializations."""
        for _ in range(n_restarts):
            q_init = np.random.uniform(
                [self.model.jnt_range[jid, 0] for jid in self.joint_ids],
                [self.model.jnt_range[jid, 1] for jid in self.joint_ids],
            )
            result = self.solve(target_pos, q_init)
            if result is not None:
                return result
        return None