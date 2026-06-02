"""
Minimum Jerk Polynomial Trajectory
5th-order polynomial: ensures C2 continuity (zero jerk at endpoints)
"""
import numpy as np


def minimum_jerk_trajectory(q_start: np.ndarray, q_end: np.ndarray, 
                             duration: float, dt: float = 0.002) -> np.ndarray:
    """
    Returns (T, ndof) array of joint positions.
    Polynomial: q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    Constraints: q(0)=q_start, q(T)=q_end, vel/acc/jerk=0 at endpoints
    """
    T = duration
    t = np.arange(0, T, dt)
    
    # Coefficient matrix for 5th-order polynomial boundary conditions
    # q(0)=qs, q'(0)=0, q''(0)=0, q(T)=qe, q'(T)=0, q''(T)=0
    tau = t / T  # normalized time [0,1]
    s = 10*tau**3 - 15*tau**4 + 6*tau**5  # scaling function
    
    traj = q_start[None, :] + (q_end - q_start)[None, :] * s[:, None]
    return traj


def minimum_jerk_velocity(q_start: np.ndarray, q_end: np.ndarray,
                           duration: float, dt: float = 0.002) -> np.ndarray:
    """Returns (T, ndof) velocities."""
    T = duration
    t = np.arange(0, T, dt)
    tau = t / T
    ds_dt = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    return (q_end - q_start)[None, :] * ds_dt[:, None]