from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from data_type import SchedTuple, Stop, DAYS_5
from optimal_utils import a, get_dist, OD_CM_TO_M, EARTH_R_M


def solve_formulation_medoid(
    stops: Dict[int, Stop],
    pi: Dict[int, List[SchedTuple]],
    baseline_sched: Dict[int, SchedTuple],
    timecycle: int,
    v_max: float,
    g_max: float,
    c_max: int,
    Ddist: Dict[Tuple[int, int], float],
    w1: float = 1.0,
    w2: float = 1.0,
    time_limit: Optional[int] = 600,
    mip_gap: Optional[float] = 0.0,
    vol_tolerance: float = 0.5,
) -> Tuple[gp.Model, Dict[int, SchedTuple], Dict[int, int]]:

    WEEKS_local = list(range(1, timecycle + 1))
    ids = list(stops.keys())

    # Big-M: identical pattern to solve_formulation() in solver.py
    max_od_m = max(Ddist.values()) * OD_CM_TO_M if Ddist else 0.0
    lons = [float(stops[i].xcoord) for i in ids]
    lats = [float(stops[i].ycoord) for i in ids]
    max_manh_m = (abs(max(lats) - min(lats)) + abs(max(lons) - min(lons))) * (math.pi / 180.0) * EARTH_R_M
    M = max(max_od_m, max_manh_m, 1.0)

    # Model
    model = gp.Model("DayAssign_MSSC_Medoid")
    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)
    if mip_gap is not None:
        model.setParam(GRB.Param.MIPGap, mip_gap)

    # Decision variables
    # x_{ip} ∈ {0,1}: stop i selects schedule p
    x = {(i, p): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_W{p[0]}_D{p[1]}")
         for i in ids for p in pi[i]}

    # y_{ild} ∈ {0,1}: stop i is visited on (l, d)
    y = {(i, l, d): model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{l}_{d}")
         for i in ids for l in WEEKS_local for d in DAYS_5}

    # c_i ∈ {0,1}: 1 if stop i changes its schedule
    c = {i: model.addVar(vtype=GRB.BINARY, name=f"c_{i}") for i in ids}

    # m_{jld} ∈ {0,1}: 1 if node j is medoid of (l, d)
    m_med = {(j, l, d): model.addVar(vtype=GRB.BINARY, name=f"m_{j}_{l}_{d}")
             for j in ids for l in WEEKS_local for d in DAYS_5}

    # t_{ild} ∈ [0, M]: distance from stop i to medoid of (l, d)
    t = {(i, l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=M, name=f"t_{i}_{l}_{d}")
         for i in ids for l in WEEKS_local for d in DAYS_5}

    # w_{ld}: workload (= total volume) on day (l, d)
    w = {(l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w_{l}_{d}")
         for l in WEEKS_local for d in DAYS_5}

    # z_{ld} ∈ {0,1}: 1 if at least one stop is assigned to (l, d)
    z = {(l, d): model.addVar(vtype=GRB.BINARY, name=f"z_{l}_{d}")
         for l in WEEKS_local for d in DAYS_5}

    # w_bar: average workload scalar
    w_bar = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="w_bar")

    # wb_z_{ld}: linearisation of w_bar * z_{ld}  (continuous × binary)
    wb_z = {(l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"wb_z_{l}_{d}")
            for l in WEEKS_local for d in DAYS_5}

    model.update()

    # Objective: min Σ_{l,d} Σ_i t_{ild}
    model.setObjective(
        gp.quicksum(t[(i, l, d)] for i in ids for l in WEEKS_local for d in DAYS_5),
        GRB.MINIMIZE
    )

    # Constraints
    # (1) Each stop selects exactly one feasible schedule
    for i in ids:
        model.addConstr(gp.quicksum(x[(i, p)] for p in pi[i]) == 1,
                        name=f"choose1_{i}")

    # (2) y definition via helper a(p, l, d)
    for i in ids:
        for l in WEEKS_local:
            for d in DAYS_5:
                model.addConstr(
                    y[(i, l, d)] == gp.quicksum(a(p, l, d) * x[(i, p)] for p in pi[i]),
                    name=f"ydef_{i}_{l}_{d}")

    # (3) Daily volume capacity
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(
                gp.quicksum(stops[i].volume * y[(i, l, d)] for i in ids) <= v_max,
                name=f"capV_{l}_{d}")

    # (4) Daily weight capacity
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(
                gp.quicksum(stops[i].weight * y[(i, l, d)] for i in ids) <= g_max,
                name=f"capG_{l}_{d}")

    # (5) Change indicator: c_i = 1 - x_{i,p0}
    for i in ids:
        model.addConstr(c[i] == 1 - x[(i, baseline_sched[i])], name=f"change_{i}")

    # (6) Change budget
    model.addConstr(gp.quicksum(c[i] for i in ids) <= c_max, name="change_budget")

    # (7) Exactly one medoid if stops present, zero if not: Σ_j m_{jld} = z_{ld}
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(
                gp.quicksum(m_med[(j, l, d)] for j in ids) == z[(l, d)],
                name=f"medoid_eq_z_{l}_{d}")

    # (8) Medoid only if stop is assigned to (l, d)
    for j in ids:
        for l in WEEKS_local:
            for d in DAYS_5:
                model.addConstr(m_med[(j, l, d)] <= y[(j, l, d)],
                                name=f"medoid_assigned_{j}_{l}_{d}")

    # (9) t_{ild} definition: active when y_{ild}=1 and m_{jld}=1
    #     t_{ild} >= D_{ij} - M * (2 - y_{ild} - m_{jld})   ∀ i, j, l, d
    for i in ids:
        for j in ids:
            for l in WEEKS_local:
                for d in DAYS_5:
                    dij = get_dist(i, j, Ddist, stops)
                    model.addConstr(
                        t[(i, l, d)] >= dij - dij * (2 - y[(i, l, d)] - m_med[(j, l, d)]),
                        name=f"tdef_{i}_{j}_{l}_{d}")

    # (10) w_{ld} = 0 when no stops assigned
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(w[(l, d)] <= M * z[(l, d)],
                            name=f"w_zero_if_empty_{l}_{d}")

    # (11) Workload definition
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(
                w[(l, d)] == gp.quicksum(stops[i].volume * y[(i, l, d)] for i in ids),
                name=f"wdef_{l}_{d}")

    # (12) w_bar definition: w_bar * L * |D| = Σ_{l,d} w_{ld}
    L_count = len(WEEKS_local)
    D_count = len(DAYS_5)
    model.addConstr(
        w_bar * (L_count * D_count) == gp.quicksum(w[(l, d)] for l in WEEKS_local for d in DAYS_5),
        name="wbar_def")

    # (13) Workload tolerance: 0.5*wb_z ≤ w_{ld} ≤ 2*wb_z
    # wb_z_{ld} = w_bar * z_{ld}  (continuous × binary, McCormick linearisation)
    lo = 1.0 - vol_tolerance   # default 0.5
    hi = 1.0 + vol_tolerance   # default 1.5 → override to 2.0 per slides
    for l in WEEKS_local:
        for d in DAYS_5:
            wbz = wb_z[(l, d)]
            model.addConstr(wbz <= M * z[(l, d)],               name=f"wbz_ub1_{l}_{d}")
            model.addConstr(wbz <= w_bar,                        name=f"wbz_ub2_{l}_{d}")
            model.addConstr(wbz >= w_bar - M * (1 - z[(l, d)]), name=f"wbz_lb_{l}_{d}")
            model.addConstr(w[(l, d)] >= 0.5 * wbz,             name=f"wtol_lo_{l}_{d}")
            model.addConstr(w[(l, d)] <= 2.0 * wbz,             name=f"wtol_hi_{l}_{d}")

    # Solve
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Optimization ended with status {model.Status}")

    # Extract results
    chosen_tuple: Dict[int, SchedTuple] = {}
    for i in ids:
        for p in pi[i]:
            if x[(i, p)].X > 0.5:
                chosen_tuple[i] = p
                break

    changed_map: Dict[int, int] = {i: int(round(c[i].X)) for i in ids}

    return model, chosen_tuple, changed_map