# Titan Humanoid Simulation (V3)

This repository contains the simulation environment and control logic for the Titan Humanoid robot, developed for competition-level bimanual manipulation and locomotion tasks. The simulation is built using [MuJoCo](https://mujoco.org/) and integrates with [ROS2](https://docs.ros.org/en/humble/index.html) for scalable robotic control.
[![Titan Simulation Demo](docs/media/thumbnail.png)](https://youtu.be/7tqXmx2JkS0)

## Key Features
* **Bimanual Manipulation:** Coordinated control of dual robotic arms for complex pick-and-place and handover tasks.
* **Locomotion Planning:** Zero Moment Point (ZMP) based locomotion for stable humanoid walking.
* **Reactive FSM:** A robust Finite State Machine (`reactive_fsm.py`) handles dynamic task scheduling and event-driven execution.
* **Dynamic IK Generation:** Real-time Inverse Kinematics calculations for precise end-effector positioning.

## Technologies Used
* **Physics Engine:** MuJoCo
* **Middleware:** ROS2
* **Language:** Python 3
* **Environment:** WSL (Windows Subsystem for Linux)

## Setup and Installation
1.  Clone the repository:
```bash
    git clone [https://github.com/kevinstevbayer/humanoid_sim.git](https://github.com/kevinstevbayer/humanoid_sim.git)
    cd humanoid_sim
    ```
2.  Set up the virtual environment (recommended):
```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
```bash
    pip install -r requirements.txt
    ```

## Usage
To launch the main competition simulation sequence:
```bash
python3 src/run_competition.py
