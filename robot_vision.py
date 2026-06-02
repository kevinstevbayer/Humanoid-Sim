"""
robot_vision.py — Egocentric Vision Pipeline for Titan
Extracts clean RGB frames from MuJoCo without lagging the physics thread.
"""
import mujoco
import numpy as np

class TitanVisionBridge:
    def __init__(self, model: mujoco.MjModel, camera_name: str = "rgb_cam", width: int = 640, height: int = 480):
        self.model = model
        self.camera_name = camera_name
        self.width = width
        self.height = height
        
        # Verify the camera actually exists in the XML so we don't crash
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id == -1:
            raise ValueError(f"[VISION ERROR] Camera '{camera_name}' not found in XML!")
            
        # Initialize the hardware-accelerated off-screen renderer
        self.renderer = mujoco.Renderer(model, height, width)
        print(f"✓ Optical Bridge Online. Resolution: {width}x{height}")

    def capture_frame(self, data: mujoco.MjData) -> np.ndarray:
        """
        Takes a snapshot of the current physics state through Titan's eyes.
        Returns a standard RGB numpy array (Height, Width, 3).
        """
        # Sync the renderer's visual state with the current physics state
        self.renderer.update_scene(data, camera=self.camera_name)
        
        # Extract the raw pixel array
        img_array = self.renderer.render()
        
        return img_array

    def get_camera_matrix(self, data: mujoco.MjData) -> tuple:
        """
        Extracts the exact X,Y,Z and Rotation matrix of the camera lens in the world.
        We WILL need this tomorrow to convert VLM pixel coordinates into IK world targets.
        """
        cam_pos = data.cam_xpos[self.cam_id].copy()
        cam_mat = data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        return cam_pos, cam_mat