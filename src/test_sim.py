import mujoco
import mujoco_viewer
import numpy as np
import time

# MJCF Model: Defining a Hip and a Knee
model_xml = """
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.002"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".2 .2 .2 1"/>
        
        <body pos="0 0 1.5">
            <joint name="hip" type="hinge" axis="0 1 0" damping="0.5"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" rgba="0 1 0 1"/>
            
            <body pos="0 0 -0.5">
                <joint name="knee" type="hinge" axis="0 1 0" damping="0.5"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" rgba="0 0 1 1"/>
                
                <body pos="0 0 -0.5">
                    <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

print("Starting Simulation... Press 'Esc' in the 3D window to stop.")

while viewer.is_alive:
    # Apply a random 'kick' every 2 seconds to see the physics
    if int(data.time) % 2 == 0 and data.time % 1 < 0.01:
        data.qvel[0] = np.random.uniform(-5, 5)
    
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()
