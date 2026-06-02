"""
Bimanual Assembly Domain — Titan Humanoid
Actions: move, pick
Streams: sample_ik, sample_motion
"""
from tampas_core import *
import numpy as np

def make_move_action(arm: Const, q_start: Const, traj: Const, q_end: Const) -> DurativeAction:
    return DurativeAction(
        name="move",
        params=(arm, q_start, traj, q_end),
        start_pre={Predicate("at_conf", (arm, q_start))},
        start_del={Predicate("at_conf", (arm, q_start))},
        start_add={Predicate("moving", (arm,))},
        duration=Function("duration", (traj,)),
        end_add={Predicate("at_conf", (arm, q_end))},
        end_del={Predicate("moving", (arm,))},
    )

def make_pick_action(arm: Const, obj: Const, q_grasp: Const) -> DurativeAction:
    return DurativeAction(
        name="pick",
        params=(arm, obj, q_grasp),
        start_pre={
            Predicate("at_conf", (arm, q_grasp)),
            Predicate("at", (obj, Const("table", "surf"))),
            Predicate("hand_empty", (arm,)),
        },
        start_add={Predicate("grasping", (arm,))},
        duration=0.5,
        end_add={Predicate("holding", (arm, obj))},
        end_del={
            Predicate("at", (obj, Const("table", "surf"))),
            Predicate("hand_empty", (arm,)),
            Predicate("grasping", (arm,)),
        },
    )

class IKStream(Stream):
    def __init__(self, mujoco_model, ik_solver):
        super().__init__("sample_ik", ["obj", "grasp"], ["conf"])
        self.model = mujoco_model
        self.ik = ik_solver

    def generate(self, obj: Const, grasp: Const):
        pass

class MotionStream(Stream):
    def __init__(self, planner):
        super().__init__("sample_motion", ["conf", "conf"], ["traj"])
        self.planner = planner

    def generate(self, q_start: Const, q_end: Const):
        pass

def init_bimanual_domain(mujoco_model):
    arms = [Const("left_arm", "arm"), Const("right_arm", "arm")]
    objs = [Const("block_A", "obj"), Const("block_B", "obj")]
    
    init_state = State(
        preds={
            Predicate("at_conf", (arms[0], Const("q0_L", "conf"))),
            Predicate("at_conf", (arms[1], Const("q0_R", "conf"))),
            Predicate("hand_empty", (arms[0],)),
            Predicate("hand_empty", (arms[1],)),
            Predicate("at", (objs[0], Const("table", "surf"))),
            Predicate("at", (objs[1], Const("table", "surf"))),
        },
        consts={c.name: c for c in arms + objs}
    )
    
    return init_state, arms, objs