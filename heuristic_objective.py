from __future__ import annotations
from typing import Dict, Tuple, List
from data_type import Stop, SchedTuple
from helper_funcs import WEEKS, DAYS_5, a, get_dist


Cell = Tuple[int, str]  # (l, d)

def compute_V_cell(stops: Dict[int, Stop], p: Dict[int, SchedTuple]) -> Dict[Cell, float]:
    """V[(l,d)] = sum_i volume_i * a(p[i], l, d)"""
    V: Dict[Cell, float] = {(l, d): 0.0 for l in WEEKS for d in DAYS_5}
    for stop_id, s in stops.items():
        sched = p[stop_id]
        vol = float(s.volume)
        for l in WEEKS:
            for d in DAYS_5:
                if a(sched, l, d) == 1:
                    V[(l, d)] += vol
    return V


def volume_balance_term(V: Dict[Cell, float]) -> float:
    """sum_l (Vmax_l - Vmin_l)"""
    tot = 0.0
    for l in WEEKS:
        vals = [V[(l, d)] for d in DAYS_5]
        tot += (max(vals) - min(vals))
    return tot


def density_term_w(stops: Dict[int, Stop],
                   p: Dict[int, SchedTuple],
                   Ddist: Dict[Tuple[int, int], float]) -> float:
    """
    w = sum_{l,d} w_{l,d}, where
    w_{l,d} = sum_{i visited in (l,d)} z_{i,l,d}
    z_{i,l,d} = max_{j visited in (l,d), j!=i} dist(i,j)   (formulation-consistent)
    """
    total = 0.0

    # 1) cell 별 방문 stop 리스트 만들기
    visited_by_cell: Dict[Cell, List[int]] = {(l, d): [] for l in WEEKS for d in DAYS_5}
    for stop_id in stops.keys():
        sched = p[stop_id]
        for l in WEEKS:
            for d in DAYS_5:
                if a(sched, l, d) == 1:
                    visited_by_cell[(l, d)].append(stop_id)

    # 2) 각 cell에서 i별 min distance 합산
    for (l, d), ids in visited_by_cell.items():
        if len(ids) <= 1:
            continue

        # i마다 min_j dist(i,j)
        for i in ids:
            best = 0.0
            for j in ids:
                if j == i:
                    continue
                dij = get_dist(i, j, Ddist, stops)  # OD 없으면 Manhattan fallback 포함되어 있어야 함
                if dij < best:
                    best = dij
            total += best

    return total

def compute_objective(artifacts: dict,
                      p: Dict[int, SchedTuple],
                      w1: float = 1.0,
                      w2: float = 1.0) -> Dict[str, float]:

    stops: Dict[int, Stop] = artifacts["stops"]
    Ddist: Dict[Tuple[int, int], float] = artifacts["Ddist"]

    V = compute_V_cell(stops, p)
    w_term = density_term_w(stops, p, Ddist)
    vbal = volume_balance_term(V)
    obj = w1 * w_term + w2 * vbal
    return {"obj": obj, "density": w_term, "vol_balance": vbal}