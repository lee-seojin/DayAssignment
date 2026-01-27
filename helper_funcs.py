from data_type import SchedTuple, DAYS_5, ALL_DAYS_7, WEEKS, Stop
from typing import Dict, Tuple
import math
from datetime import datetime

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


EARTH_R_M = 6371000.0  # meters

def manhattan(stop_i: Stop, stop_j: Stop) -> float:
    # lat/lon in degrees -> Manhattan distance in meters.
    lon1 = float(stop_i.xcoord)
    lat1 = float(stop_i.ycoord)
    lon2 = float(stop_j.xcoord)
    lat2 = float(stop_j.ycoord)

    # degrees -> radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lam1 = math.radians(lon1)
    lam2 = math.radians(lon2)

    dphi = abs(phi2 - phi1)
    dlam = abs(lam2 - lam1)

    # north-south component
    d_lat = EARTH_R_M * dphi

    # east-west component (scaled by cos(mean latitude))
    phi_m = 0.5 * (phi1 + phi2)
    d_lon = EARTH_R_M * math.cos(phi_m) * dlam

    return d_lat + d_lon

OD_CM_TO_M = 0.01

def get_dist(
        i: int,
        j: int,
        Ddist: Dict[Tuple[int, int], float],
        stops: Dict[int, Stop],
) -> float:
    # OD가 있으면 사용, 없으면 Manhattan fallback -> 단위는 meter

    if (i, j) in Ddist:
        return float(Ddist[(i, j)]) * OD_CM_TO_M

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

def make_run_prefix(dataset: str) -> str:
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")   # 20260127
    time_str = now.strftime("%H%M%S")   # 153012
    return f"{dataset}_{date_str}_{time_str}"