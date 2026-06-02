"""
TAMPAS Core — Python-native ScheduleStream framework
Garrett & Ramos architecture for GPU-accelerated bimanual TAMP
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterator, Any
from abc import ABC, abstractmethod
import numpy as np


# ══════════════════════════════════════════════════════════════════════
#  PRIMITIVES
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Const:
    """Symbolic constant (object, config, trajectory)."""
    name: str
    type: str  # 'obj' | 'conf' | 'traj'
    value: Any = None  # lazy: None until sampled

@dataclass
class Predicate:
    """State predicate: at_conf(arm, q), holding(arm, obj)."""
    name: str
    args: tuple[Const, ...]
    
    def __hash__(self): return hash((self.name, self.args))
    def __eq__(self, o): return self.name == o.name and self.args == o.args

@dataclass
class Function:
    """Numeric function: duration(traj) → float."""
    name: str
    args: tuple[Const, ...]
    value: float | None = None


# ══════════════════════════════════════════════════════════════════════
#  DURATIVE ACTIONS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DurativeAction:
    """
    Hybrid durative action: start event + duration + end event.
    Ex: move(arm, q_start, traj, q_end)
      start: at_conf(arm, q_start) → ¬at_conf(arm, q_start)
      dur:   duration(traj)
      end:   → at_conf(arm, q_end)
    """
    name: str
    params: tuple[Const, ...]
    
    # Start event
    start_pre: set[Predicate]  = field(default_factory=set)
    start_add: set[Predicate]  = field(default_factory=set)
    start_del: set[Predicate]  = field(default_factory=set)
    
    # Duration constraint
    duration: Function | float | None = None
    
    # End event
    end_pre: set[Predicate]    = field(default_factory=set)
    end_add: set[Predicate]    = field(default_factory=set)
    end_del: set[Predicate]    = field(default_factory=set)

    def to_schedule_actions(self) -> tuple[Action, Action]:
        """Compile to start/end instantaneous actions for A* search."""
        start_act = Action(
            name=f"{self.name}_start",
            params=self.params,
            pre=self.start_pre,
            add=self.start_add,
            delete=self.start_del,
        )
        end_act = Action(
            name=f"{self.name}_end",
            params=self.params,
            pre=self.end_pre,
            add=self.end_add,
            delete=self.end_del,
        )
        return start_act, end_act


@dataclass
class Action:
    """Instantaneous action (for scheduling search)."""
    name: str
    params: tuple[Const, ...]
    pre: set[Predicate]    = field(default_factory=set)
    add: set[Predicate]    = field(default_factory=set)
    delete: set[Predicate] = field(default_factory=set)


# ══════════════════════════════════════════════════════════════════════
#  STREAMS (lazy samplers)
# ══════════════════════════════════════════════════════════════════════

class Stream(ABC):
    """
    Conditional generator: given input consts, yield output consts.
    Ex: sample_ik(obj, grasp) → [q_conf1, q_conf2, ...]
    """
    def __init__(self, name: str, inputs: list[str], outputs: list[str]):
        self.name    = name
        self.inputs  = inputs   # type names
        self.outputs = outputs

    @abstractmethod
    def generate(self, *args: Const) -> Iterator[tuple[Const, ...]]:
        """Yield tuples of output consts. May be infinite."""
        ...

    def lazy_output(self, *inputs: Const) -> tuple[Const, ...]:
        """Return placeholder consts before sampling."""
        uid = hash((self.name, inputs))
        return tuple(
            Const(f"{self.name}_{otype}_{uid}", otype)
            for otype in self.outputs
        )


# ══════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class State:
    """Planning state: set of predicates + const bindings."""
    preds: set[Predicate] = field(default_factory=set)
    consts: dict[str, Const] = field(default_factory=dict)

    def satisfies(self, pre: set[Predicate]) -> bool:
        return pre.issubset(self.preds)

    def apply(self, add: set[Predicate], delete: set[Predicate]) -> State:
        new_preds = (self.preds - delete) | add
        return State(new_preds, self.consts.copy())