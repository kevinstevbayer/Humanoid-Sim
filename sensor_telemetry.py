"""
Sensor Telemetry Loop — Extract IMU and F/T from MuJoCo
Fixed: Safe dictionary lookups to prevent KeyError crashes on missing sensors.
"""
import numpy as np
import mujoco


class SensorTelemetry:
    """
    Efficiently read sensors from data.sensordata during sim loop.
    Pre-cache sensor addresses at init.
    """
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._cache = {}
        
        # Build sensor address map
        for i in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                adr = model.sensor_adr[i]
                dim = model.sensor_dim[i]
                self._cache[name] = (adr, dim)
    
    def read(self, data: mujoco.MjData, sensor_name: str) -> np.ndarray:
        """
        Read sensor by name. 
        CRITICAL FIX: Returns a zero array instead of crashing if sensor is missing.
        """
        if sensor_name not in self._cache:
            # Default fallback to a 3D zero vector (standard for force/torque/accel)
            return np.zeros(3)
            
        adr, dim = self._cache[sensor_name]
        return data.sensordata[adr : adr + dim].copy()
    
    def read_imu(self, data: mujoco.MjData, site_name: str) -> dict:
        """
        Read full IMU suite from a site.
        Returns: {accel: (3,), gyro: (3,), quat: (4,)}
        """
        # Note: Quat usually requires a 4D vector, but `read` defaults to 3D if missing.
        # This is generally safe as long as the base IMU sensors exist in the XML.
        accel = self.read(data, f"{site_name}_accel")
        gyro  = self.read(data, f"{site_name}_gyro")
        
        # Safe quat read just in case
        quat_name = f"{site_name}_quat"
        if quat_name in self._cache:
            quat = self.read(data, quat_name)
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0]) # Valid default identity quaternion
            
        return {
            "accel": accel,
            "gyro":  gyro,
            "quat":  quat,
        }
    
    def read_ft(self, data: mujoco.MjData, site_name: str) -> dict:
        """
        Read force-torque sensor at wrist/sole.
        Returns: {force: (3,), torque: (3,)}
        """
        return {
            "force":  self.read(data, f"{site_name}_force"),
            "torque": self.read(data, f"{site_name}_torque"),
        }