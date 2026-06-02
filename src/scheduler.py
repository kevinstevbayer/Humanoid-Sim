"""
Scheduling Subroutine — Lazy Weighted A* for Minimal Makespan
Compiles durative actions into start/end events, searches schedule skeleton
"""
from tampas_core import *
from collections import defaultdict
import heapq


# ══════════════════════════════════════════════════════════════════════
#  SCHEDULE REPRESENTATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ScheduleNode:
    """
    Node in schedule search space.
    state: current predicates
    timeline: {action_id: (start_time, end_time)}
    makespan: max(end_time)
    """
    state: State
    timeline: dict[int, tuple[float, float]] = field(default_factory=dict)
    makespan: float = 0.0
    g: float = 0.0  # cost so far
    
    def __lt__(self, o): return self.g < o.g


# ══════════════════════════════════════════════════════════════════════
#  SCHEDULER
# ══════════════════════════════════════════════════════════════════════

class LazyScheduler:
    """
    Weighted A* search over schedule skeletons.
    Durative actions → start/end event pairs.
    """
    def __init__(self, durative_actions: list[DurativeAction], weight: float = 1.5):
        self.durative_actions = durative_actions
        self.weight = weight
        
        # Compile to start/end pairs
        self.start_actions = {}
        self.end_actions   = {}
        for i, da in enumerate(durative_actions):
            s, e = da.to_schedule_actions()
            self.start_actions[i] = (s, da)
            self.end_actions[i]   = (e, da)

    def search(self, init_state: State, goal: set[Predicate], max_depth: int = 20) -> list:
        """
        Returns schedule: [(action_id, start_time, end_time), ...]
        """
        init_node = ScheduleNode(state=init_state)
        pq = [(0.0, init_node)]
        visited = set()
        
        while pq:
            _, node = heapq.heappop(pq)
            
            if goal.issubset(node.state.preds):
                return self._extract_schedule(node)
            
            state_sig = frozenset(node.state.preds)
            if state_sig in visited:
                continue
            visited.add(state_sig)
            
            if len(node.timeline) >= max_depth:
                continue
            
            # Try starting any durative action
            for aid, (start_act, dur_act) in self.start_actions.items():
                if aid in node.timeline:
                    continue  # already started
                if not node.state.satisfies(start_act.pre):
                    continue
                
                # Apply start event
                new_state = node.state.apply(start_act.add, start_act.delete)
                
                # Compute earliest start time (no resource conflicts)
                t_start = self._earliest_start(node, dur_act)
                dur_val = dur_act.duration.value if isinstance(dur_act.duration, Function) else dur_act.duration
                if dur_val is None:
                    dur_val = 1.0  # lazy placeholder
                
                new_timeline = node.timeline.copy()
                new_timeline[aid] = (t_start, t_start + dur_val)
                new_makespan = max(new_timeline.values(), key=lambda x: x[1])[1]
                
                new_node = ScheduleNode(
                    state=new_state,
                    timeline=new_timeline,
                    makespan=new_makespan,
                    g=new_makespan,
                )
                
                # Check if we can apply end event immediately (for instantaneous)
                end_act, _ = self.end_actions[aid]
                if new_state.satisfies(end_act.pre):
                    final_state = new_state.apply(end_act.add, end_act.delete)
                    final_node = ScheduleNode(
                        state=final_state,
                        timeline=new_timeline,
                        makespan=new_makespan,
                        g=new_makespan,
                    )
                    f = final_node.g + self.weight * self._heuristic(final_node, goal)
                    heapq.heappush(pq, (f, final_node))
        
        return None  # no solution

    def _earliest_start(self, node: ScheduleNode, action: DurativeAction) -> float:
        """Resource conflict check: arms can't overlap."""
        # Extract which arm this action uses
        arm_param = action.params[0] if action.params else None
        if arm_param is None:
            return node.makespan
        
        # Find last end time for this arm
        last_t = 0.0
        for aid, (ts, te) in node.timeline.items():
            other_action = self.durative_actions[aid]
            if other_action.params and other_action.params[0] == arm_param:
                last_t = max(last_t, te)
        
        return last_t

    def _heuristic(self, node: ScheduleNode, goal: set[Predicate]) -> float:
        """Num unsatisfied predicates."""
        return len(goal - node.state.preds)

    def _extract_schedule(self, node: ScheduleNode) -> list:
        return [(aid, ts, te) for aid, (ts, te) in node.timeline.items()]


# ══════════════════════════════════════════════════════════════════════
#  INTEGRATION HOOK
# ══════════════════════════════════════════════════════════════════════

def plan_bimanual_assembly(init_state: State, goal: set[Predicate], 
                            actions: list[DurativeAction]) -> list:
    """
    Main entry point: given init state + goal, return schedule.
    """
    scheduler = LazyScheduler(actions, weight=1.5)
    schedule = scheduler.search(init_state, goal, max_depth=10)
    
    if schedule is None:
        raise RuntimeError("No schedule found")
    
    return schedule