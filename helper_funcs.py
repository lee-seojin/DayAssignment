from data_type import SchedTuple, DAYS_5, ALL_DAYS_7, WEEKS, Stop
from typing import Dict, Tuple

def a(schedule: SchedTuple, l: int, d: str) -> int:
    """schedule이 (week=l, day=d)에 방문하면 1 else 0"""
    week_t, day_b = schedule
    return int((l in week_t) and (day_b[ALL_DAYS_7.index(d)] == 1))


def compute_V(stops: Dict[int, Stop], p: Dict[int, SchedTuple]) -> Dict[Tuple[int, str], float]:
    """
    V[(l,d)] = 해당 (week,day)의 총 volume
    - (l=1..4, d=MON..FRI)만 계산
    """
    V: Dict[Tuple[int, str], float] = {(l, d): 0.0 for l in WEEKS for d in DAYS_5}
    for i, sched in p.items():
        vol = float(stops[i].volume)  # type: ignore
        for l in WEEKS:
            for d in DAYS_5:
                if a(sched, l, d):
                    V[(l, d)] += vol
    return V


def manhattan(stop_i: Stop, stop_j: Stop) -> float:
    return abs(float(stop_i.xcoord) - float(stop_j.xcoord)) + abs(float(stop_i.ycoord) - float(stop_j.ycoord))  # type: ignore


def get_dist(
    i: int,
    j: int,
    Ddist: Dict[Tuple[int, int], float],
    stops: Dict[int, Stop],
) -> float:
    """
    OD가 있으면 사용, 없으면 Manhattan fallback
    """
    if (i, j) in Ddist:
        return float(Ddist[(i, j)])
    return manhattan(stops[i], stops[j])

def try_apply_change(
    stop_id: int,
    new_sched: SchedTuple,
    p: Dict[int, SchedTuple],
    baseline_sched: Dict[int, SchedTuple],
    changed: Dict[int, int],
    C_used: int,
    C_max: int,
) -> Tuple[bool, int]:

    before = changed[stop_id]
    after = 1 if new_sched != baseline_sched[stop_id] else 0
    new_C_used = C_used - before + after

    if new_C_used > C_max:
        return False, C_used

    p[stop_id] = new_sched
    changed[stop_id] = after
    return True, new_C_used
