from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from data_type import SchedTuple, Stop, DAYS_5
from optimal_utils import a, get_dist, OD_CM_TO_M, EARTH_R_M, build_k_neighborhood

def solve_formulation_wo_balancing(
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
    k_neigh: int = 100,
):

    WEEKS_local = list(range(1, timecycle + 1))
    ids = list(stops.keys())

    max_od_m = max(Ddist.values()) * OD_CM_TO_M if Ddist else 0.0
    lons = [float(stops[i].xcoord) for i in ids]
    lats = [float(stops[i].ycoord) for i in ids]
    max_manh_m = (abs(max(lats) - min(lats)) + abs(max(lons) - min(lons))) * (math.pi / 180.0) * EARTH_R_M
    M = max(max_od_m, max_manh_m)

    model = gp.Model("DayAssign_NearestNeighbor")
    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)
    if mip_gap is not None:
        model.setParam(GRB.Param.MIPGap, mip_gap)

    # Variables
    # x_{ip} ∈ {0,1}: stop i selects schedule p
    x = {(i, p): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_W{p[0]}_D{p[1]}")
         for i in ids for p in pi[i]}

    # y_{ild} ∈ {0,1}: stop i is visited on (l, d)
    y = {(i, l, d): model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{l}_{d}")
         for i in ids for l in WEEKS_local for d in DAYS_5}

    # z_{ild} ≥ 0: minimum nearest-neighbor distance of stop i on (l, d)
    z = {(i, l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{i}_{l}_{d}")
         for i in ids for l in WEEKS_local for d in DAYS_5}

    # k-nearest neighborhood to limit v variable count
    neigh = build_k_neighborhood(ids=ids, Ddist=Ddist, stops=stops, k=k_neigh)

    # v_{ijld} ∈ {0,1}: j is chosen as i's nearest neighbor on (l,d)
    # j != i: self cannot be nearest neighbor (D_{ii}=0 would trivially minimize z)
    v = {(i, j, l, d): model.addVar(vtype=GRB.BINARY, name=f"v_{i}_{j}_{l}_{d}")
         for i in ids for j in neigh[i] if j != i
         for l in WEEKS_local for d in DAYS_5}

    # c_i ∈ {0,1}: 1 if stop i changes its schedule
    c = {i: model.addVar(vtype=GRB.BINARY, name=f"c_{i}") for i in ids}

    # w_{ld}: daily workload = sum of nearest-neighbor distances
    w = {(l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w_{l}_{d}")
         for l in WEEKS_local for d in DAYS_5}

    model.update()

    # Objective: min Σ_{l,d} w_{ld}
    model.setObjective(
        gp.quicksum(w[(l, d)] for l in WEEKS_local for d in DAYS_5),
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
        p0 = baseline_sched[i]
        model.addConstr(c[i] + x[(i, p0)] == 1, name=f"change_{i}")

    # (6) Change budget
    model.addConstr(gp.quicksum(c[i] for i in ids) <= c_max, name="change_budget")

    for l in WEEKS_local:
        for d in DAYS_5:
            for i in ids:
                neighbors_i = [j for j in neigh[i] if j != i]

                # (9) z active only if i is visited
                model.addConstr(z[(i, l, d)] <= M * y[(i, l, d)],
                                name=f"z_act_{i}_{l}_{d}")

                # (10) exactly one neighbor selected iff i is visited
                model.addConstr(
                    gp.quicksum(v[(i, j, l, d)] for j in neighbors_i) == y[(i, l, d)],
                    name=f"chooseNbr_{i}_{l}_{d}")

                for j in neighbors_i:
                    dij = get_dist(i, j, Ddist, stops)

                    # (7) z_{ild} ≤ D_{ij} + M*(2 - y_{ild} - y_{jld})
                    model.addConstr(
                        z[(i, l, d)] <= dij + M * (2 - y[(i, l, d)] - y[(j, l, d)]),
                        name=f"z_ub_{i}_{j}_{l}_{d}")

                    # (8) z_{ild} ≥ D_{ij} - M*(1 - v_{ijld})
                    model.addConstr(
                        z[(i, l, d)] >= dij - M * (1 - v[(i, j, l, d)]),
                        name=f"z_lb_{i}_{j}_{l}_{d}")

                    # (11) neighbor j must be visited on same (l,d)
                    model.addConstr(v[(i, j, l, d)] <= y[(j, l, d)],
                                    name=f"v_le_y_{i}_{j}_{l}_{d}")

            # (12) w_{ld} = Σ_i z_{ild}
            model.addConstr(
                w[(l, d)] == gp.quicksum(z[(i, l, d)] for i in ids),
                name=f"wdef_{l}_{d}")

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

    return model, chosen_tuple, {i: int(round(c[i].X)) for i in ids}