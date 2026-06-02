"""
Reactive FSM Execution Layer
Fixed: Safely reads data.ctrl instead of mismatched qpos arrays.
"""
from enum import Enum, auto
import numpy as np
import mujoco

class ActionState(Enum):
    IDLE      = auto()
    EXECUTING = auto()
    HALTED    = auto()  # Force breach detected
    RECOVERY  = auto()  # Executing reflex pull-back
    FAILED    = auto()
    DONE      = auto()

class ReactiveActionExecutor:
    def __init__(self, arm_name: str, q_start: np.ndarray, q_end: np.ndarray, 
                 t_start: float, t_end: float, actuator_ids: list[int], ft_sensor: str):
        self.arm_name = arm_name
        self.q_start = q_start
        self.q_end = q_end
        self.t_start = t_start
        self.t_end = t_end
        self.actuator_ids = actuator_ids
        self.ft_sensor = ft_sensor
        
        self.state = ActionState.IDLE
        self.recovery_start_time = 0.0
        self.q_halted = None 
        self.threshold_n = 15.0 # 15 Newtons of force limit
        
    def step(self, current_time: float, data: mujoco.MjData, telemetry) -> ActionState:
        if self.state == ActionState.IDLE:
            if current_time >= self.t_start:
                self.state = ActionState.EXECUTING
            return self.state
            
        if self.state == ActionState.DONE or self.state == ActionState.FAILED:
            return self.state

        # --- 1. THE SENSOR CHECK (With Grace Period) ---
        if self.state == ActionState.EXECUTING:
            
            # Ignore sensors for the first 0.3 seconds to let the physics engine settle
            if (current_time - self.t_start) > 0.3:
                force_array = telemetry.read(data, self.ft_sensor)
                force_mag = np.linalg.norm(force_array)
                
                # THE REFLEX TRIPWIRE
                if force_mag > self.threshold_n:
                    print(f"\n[CRITICAL REFLEX] {self.arm_name} impact detected: {force_mag:.2f} N!")
                    self.state = ActionState.HALTED
                    # CRITICAL FIX: Read current motor commands safely instead of mismatched qpos indices
                    self.q_halted = np.array([data.ctrl[act_id] for act_id in self.actuator_ids]) 
                    return self.state
            
            # --- 2. THE HARDWARE-SAFE MOTION ---
            target_q = self.get_min_jerk_q(current_time)
            
            # Bulletproof motor assignment
            for i, act_id in enumerate(self.actuator_ids):
                data.ctrl[act_id] = target_q[i]
            
            if current_time >= self.t_end:
                self.state = ActionState.DONE
                
        # --- 3. THE RECOVERY REFLEX ---
        elif self.state == ActionState.HALTED:
            self.recovery_start_time = current_time
            self.state = ActionState.RECOVERY
            print(f"[RECOVERY] Pulling {self.arm_name} back to safe distance...")
            
        elif self.state == ActionState.RECOVERY:
            elapsed = current_time - self.recovery_start_time
            if elapsed < 0.5: 
                recovery_q = np.copy(self.q_halted)
                recovery_q[0] -= 0.2 
                for i, act_id in enumerate(self.actuator_ids):
                    data.ctrl[act_id] = recovery_q[i]
            else:
                self.state = ActionState.FAILED
                print(f"[FSM] Action Failed. Requires High-Level TAMPAS Re-plan.")
                
        return self.state

    def get_min_jerk_q(self, current_time: float):
        if current_time <= self.t_start: return self.q_start
        if current_time >= self.t_end: return self.q_end
        tau = (current_time - self.t_start) / (self.t_end - self.t_start)
        s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
        return self.q_start + s * (self.q_end - self.q_start)