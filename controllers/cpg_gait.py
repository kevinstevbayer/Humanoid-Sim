# controllers/cpg_gait.py
"""
Central Pattern Generator for Titan Humanoid V2.

Coordinates: hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll (both legs)
             + arm counter-swing (shoulder_pitch, elbow) for angular momentum

Phase convention:
  phi_r = 2π·f·t          (right leg)
  phi_l = phi_r + π       (left leg, anti-phase)

All output in RADIANS.
"""
from __future__ import annotations
import numpy as np
import mujoco

from .base import BaseController


class CPGGait(BaseController):
    # ── Gait parameters (tune these for competition) ──────────────────
    FREQ_HZ      = 0.85    # step frequency
    HIP_AMP      = 0.38    # rad  (~22°) hip sagittal swing
    KNEE_AMP     = 0.40    # rad  (~23°) knee swing above bias
    KNEE_BIAS    = -0.22   # rad  (~-13°) knees slightly bent at rest
    ANKLE_AMP    = 0.18    # rad  (~10°) ankle push-off
    ANKLE_BIAS   = -0.05   # rad  slight plantar-flex bias
    HIP_ROLL_AMP = 0.04    # rad  lateral weight shift
    ARM_AMP      = 0.30    # rad  arm counter-swing amplitude
    ELBOW_BIAS   = -0.90   # rad  (~-52°) elbow bent for natural gait
    ELBOW_AMP    = 0.15    # rad  elbow swing

    # Ankle lead phase (push-off happens before foot lifts)
    ANKLE_PHASE_LEAD = 0.45   # rad

    # ── Postural bias for stable stance ───────────────────────────────
    TORSO_PITCH_BIAS = 0.04   # rad forward lean (helps ZMP stay in support polygon)

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 freq: float | None = None):
        super().__init__(model, data)
        self.freq = freq or self.FREQ_HZ

    def compute(self, t: float) -> dict[str, float]:
        phi_r = 2.0 * np.pi * self.freq * t
        phi_l = phi_r + np.pi

        # ── Knee: rectified (never hyperextend), phase-advanced 0.3 rad
        knee_r = self.KNEE_BIAS + self.KNEE_AMP * max(0.0, np.sin(phi_r + 0.30))
        knee_l = self.KNEE_BIAS + self.KNEE_AMP * max(0.0, np.sin(phi_l + 0.30))

        # ── Hip roll: lateral weight shift (peaks at mid-stance)
        hip_roll_r =  self.HIP_ROLL_AMP * np.sin(phi_r + np.pi / 2)
        hip_roll_l = -self.HIP_ROLL_AMP * np.sin(phi_r + np.pi / 2)

        # ── Ankle: push-off leads hip by ANKLE_PHASE_LEAD
        ankle_r = self.ANKLE_BIAS + self.ANKLE_AMP * np.sin(phi_r + self.ANKLE_PHASE_LEAD)
        ankle_l = self.ANKLE_BIAS + self.ANKLE_AMP * np.sin(phi_l + self.ANKLE_PHASE_LEAD)

        # ── Arms: counter-swing to right leg (anti-phase)
        arm_r_pitch = self.ARM_AMP * np.sin(phi_l)   # right arm swings with left leg
        arm_l_pitch = self.ARM_AMP * np.sin(phi_r)
        elbow_r = self.ELBOW_BIAS + self.ELBOW_AMP * np.sin(phi_l + 0.2)
        elbow_l = self.ELBOW_BIAS + self.ELBOW_AMP * np.sin(phi_r + 0.2)

        return {
            # ── Right leg
            "r_hip_pitch_mot":  self.HIP_AMP * np.sin(phi_r),
            "r_hip_roll_mot":   hip_roll_r,
            "r_knee_mot":       knee_r,
            "r_ankle_pitch_mot": ankle_r,
            "r_ankle_roll_mot":  0.0,

            # ── Left leg
            "l_hip_pitch_mot":  self.HIP_AMP * np.sin(phi_l),
            "l_hip_roll_mot":   hip_roll_l,
            "l_knee_mot":       knee_l,
            "l_ankle_pitch_mot": ankle_l,
            "l_ankle_roll_mot":  0.0,

            # ── Torso
            "torso_pitch_mot":  self.TORSO_PITCH_BIAS,

            # ── Arms
            "r_shoulder_p_mot": arm_r_pitch,
            "r_shoulder_r_mot": 0.0,
            "r_elbow_mot":      elbow_r,
            "r_wrist_mot":      0.0,

            "l_shoulder_p_mot": arm_l_pitch,
            "l_shoulder_r_mot": 0.0,
            "l_elbow_mot":      elbow_l,
            "l_wrist_mot":      0.0,
        }
