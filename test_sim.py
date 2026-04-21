import mujoco
import mujoco_viewer
import time
import math

HUMANOID_XML = """
<mujoco model="Titan_Humanoid_v2">
    <compiler angle="degree" coordinate="local"/>
    <option gravity="0 0 -9.81" timestep="0.002"/>

    <default>
        <joint type="hinge" damping="0.5"/>
        <geom type="capsule" rgba="0.7 0.7 0.7 1" density="1000"/>
        <default class="limb"><geom size="0.04"/></default>
        <default class="torso"><geom size="0.07"/></default>
    </default>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="5 5 0.1" rgba=".2 .3 .4 1"/>

        <body name="table" pos="0.8 0 0.4">
            <geom type="box" size="0.3 0.6 0.02" rgba=".4 .2 .1 1"/>
            <geom type="box" size="0.02 0.02 0.2" pos="0.25 0.55 -0.2" rgba=".3 .3 .3 1"/>
            <geom type="box" size="0.02 0.02 0.2" pos="-0.25 0.55 -0.2" rgba=".3 .3 .3 1"/>
            <geom type="box" size="0.02 0.02 0.2" pos="0.25 -0.55 -0.2" rgba=".3 .3 .3 1"/>
            <geom type="box" size="0.02 0.02 0.2" pos="-0.25 -0.55 -0.2" rgba=".3 .3 .3 1"/>
        </body>

        <body name="pelvis" pos="0 0 1.2">
            <geom class="torso" fromto="0 0 0 0 0 0.2"/>
            <body name="head" pos="0 0 0.3">
                <geom type="sphere" size="0.09" rgba=".9 .6 .5 1"/>
            </body>

            <body name="right_leg" pos="0 -0.1 -0.1">
                <joint name="right_hip" axis="0 1 0"/>
                <geom class="limb" fromto="0 0 0 0 0 -0.4"/>
                <body name="right_shin" pos="0 0 -0.4">
                    <joint name="right_knee" axis="0 1 0"/>
                    <geom class="limb" fromto="0 0 0 0 0 -0.4"/>
                </body>
            </body>
            
            <body name="left_leg" pos="0 0.1 -0.1">
                <joint name="left_hip" axis="0 1 0"/>
                <geom class="limb" fromto="0 0 0 0 0 -0.4"/>
                <body name="left_shin" pos="0 0 -0.4">
                    <joint name="left_knee" axis="0 1 0"/>
                    <geom class="limb" fromto="0 0 0 0 0 -0.4"/>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <position name="r_hip_motor" joint="right_hip" kp="200"/>
        <position name="r_knee_motor" joint="right_knee" kp="200"/>
        <position name="l_hip_motor" joint="left_hip" kp="200"/>
        <position name="l_knee_motor" joint="left_knee" kp="200"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(HUMANOID_XML)
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

# Start Simulation
while viewer.is_alive:
    # Calculate a simple sine wave for movement
    time_sec = data.time
    hip_angle = 30 * math.sin(time_sec * 2)  # Swing between -30 and 30 degrees
    knee_angle = -30 + 30 * math.sin(time_sec * 2) # Bend knee
    
    # Send control signals to the 4 actuators [r_hip, r_knee, l_hip, l_knee]
    # We invert the left leg so it walks in an alternating pattern
    data.ctrl[:] = [hip_angle, knee_angle, -hip_angle, -knee_angle]

    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()
