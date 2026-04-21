# controllers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import mujoco


class BaseController(ABC):
    """
    Abstract base for all Titan controllers.
    Subclasses implement compute() → {actuator_name: float (radians)}.
    apply() writes directly to data.ctrl via name-indexed map.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data
        # Build actuator name → ctrl index map once at init — never hardcode indices
        self._ctrl_map: Dict[str, int] = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }
        # Build sensor name → (adr, dim) map
        self._sensor_map: Dict[str, tuple] = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i): (
                model.sensor_adr[i],
                model.sensor_dim[i],
            )
            for i in range(model.nsensor)
        }

    @abstractmethod
    def compute(self, t: float) -> Dict[str, float]:
        """Return {actuator_name: target_angle_radians}."""
        ...

    def apply(self, t: float) -> None:
        targets = self.compute(t)
        for name, val in targets.items():
            idx = self._ctrl_map.get(name)
            if idx is not None:
                self.data.ctrl[idx] = float(val)

    def read_sensor(self, name: str) -> np.ndarray:
        adr, dim = self._sensor_map[name]
        return self.data.sensordata[adr : adr + dim].copy()

    def read_contact_force(self, sole_site: str) -> np.ndarray:
        """Returns (fx, fy, fz) at a sole site in world frame."""
        return self.read_sensor(sole_site)
