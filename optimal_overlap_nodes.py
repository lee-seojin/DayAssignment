from __future__ import annotations

import itertools
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from data_type import SchedTuple, Stop, DAYS_5
from optimal_utils import a

def solve_formulation_overlap_nodes(
    stops: Dict[int, Stop],
    pi: Dict[int, List[SchedTuple]],
    baseline_sched: Dict[int, SchedTuple],
    timecycle: int,
    v_max: float,
    g_max: float,
    c_max: int,
    Ddist: Dict[Tuple[int, int], float],
    w1: float = 0.0,
    w2: float = 1.0,
    time_limit: Optional[int] = 600,
    mip_gap: Optional[float] = 0.0,
):
    WEEKS_local = list(range(1, int(timecycle) + 1))
    ids = list(stops.keys())

    x_vals = [float(stops[i].xcoord) for i in ids]
    y_vals = [float(stops[i].ycoord) for i in ids]

    XMIN, XMAX = min(x_vals), max(x_vals)
    YMIN, YMAX = min(y_vals), max(y_vals)

    MX = max(XMAX - XMIN, 1.0)
    MY = max(YMAX - YMIN, 1.0)

    model = gp.Model("Optimal_Overlap_Nodes")

    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)

    if mip_gap is not None:
        model.setParam(GRB.Param.MIPGap, mip_gap)

    # Decision Variables
    # s_ip: stop i chooses schedule p
    s = {
        (i, p): model.addVar(vtype=GRB.BINARY, name=f"s_{i}_W{p[0]}_D{p[1]}")
        for i in ids for p in pi[i]
    }

    # z_ild: stop i is visited in week l, day d
    z = {
        (i, l, d): model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{l}_{d}")
        for i in ids for l in WEEKS_local for d in DAYS_5
    }

    # c_i: stop i changed from baseline schedule
    c = {i: model.addVar(vtype=GRB.BINARY, name=f"c_{i}") for i in ids}

    # Rectangle bounds for each week/day
    x1 = {
        (l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=XMIN, ub=XMAX, name=f"x1_{l}_{d}")
        for l in WEEKS_local for d in DAYS_5
    }
    x2 = {
        (l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=XMIN, ub=XMAX, name=f"x2_{l}_{d}")
        for l in WEEKS_local for d in DAYS_5
    }
    y1 = {
        (l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=YMIN, ub=YMAX, name=f"y1_{l}_{d}")
        for l in WEEKS_local for d in DAYS_5
    }
    y2 = {
        (l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=YMIN, ub=YMAX, name=f"y2_{l}_{d}")
        for l in WEEKS_local for d in DAYS_5
    }

    # Side indicators for stop i being inside rectangle (l,e)
    Lside = {
        (i, l, e): model.addVar(vtype=GRB.BINARY, name=f"L_{i}_{l}_{e}")
        for i in ids for l in WEEKS_local for e in DAYS_5
    }
    Rside = {
        (i, l, e): model.addVar(vtype=GRB.BINARY, name=f"R_{i}_{l}_{e}")
        for i in ids for l in WEEKS_local for e in DAYS_5
    }
    Bside = {
        (i, l, e): model.addVar(vtype=GRB.BINARY, name=f"B_{i}_{l}_{e}")
        for i in ids for l in WEEKS_local for e in DAYS_5
    }
    Uside = {
        (i, l, e): model.addVar(vtype=GRB.BINARY, name=f"U_{i}_{l}_{e}")
        for i in ids for l in WEEKS_local for e in DAYS_5
    }

    # q_ile: stop i is inside rectangle of week l, day e
    q = {
        (i, l, e): model.addVar(vtype=GRB.BINARY, name=f"q_{i}_{l}_{e}")
        for i in ids for l in WEEKS_local for e in DAYS_5
    }

    # qn_ile: stop i is inside rectangle of day e but does not visit day e
    qn = {
        (i, l, e): model.addVar(vtype=GRB.BINARY, name=f"qn_{i}_{l}_{e}")
        for i in ids for l in WEEKS_local for e in DAYS_5
    }

    model.update()

    # Objective
    rect_size = gp.quicksum(
        (x2[(l, d)] - x1[(l, d)]) + (y2[(l, d)] - y1[(l, d)]) for l in WEEKS_local for d in DAYS_5
    )

    # Count +1 when stop i lies inside rectangle of a day e it does not visit.
    node_overlap = gp.quicksum(
        qn[(i, l, e)] for i in ids for l in WEEKS_local for e in DAYS_5
    )

    model.setObjective(w1 * rect_size + w2 * node_overlap, GRB.MINIMIZE)

    # Constraints
    # (1) Pattern assign
    for i in ids:
        model.addConstr(
            gp.quicksum(s[(i, p)] for p in pi[i]) == 1, name=f"pattern_assign_{i}",
        )

    # (2) Day assign
    for i in ids:
        for l in WEEKS_local:
            for d in DAYS_5:
                model.addConstr(
                    z[(i, l, d)] == gp.quicksum(a(p, l, d) * s[(i, p)] for p in pi[i]), name=f"day_assign_{i}_{l}_{d}",
                )

    # (3) Volume constraint
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(
                gp.quicksum(float(stops[i].volume) * z[(i, l, d)] for i in ids) <= v_max, name=f"capV_{l}_{d}",
            )

    # (4) Weight constraint
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(
                gp.quicksum(float(stops[i].weight) * z[(i, l, d)] for i in ids) <= g_max, name=f"capG_{l}_{d}",
            )

    # (5) Pattern change
    for i in ids:
        p0 = baseline_sched[i]

        if (i, p0) not in s:
            raise KeyError(f"baseline schedule {p0} for stop {i} is not included in pi[{i}]")

        model.addConstr(c[i] + s[(i, p0)] == 1, name=f"change_{i}")

    # (6) Pattern alternation constraint
    model.addConstr(gp.quicksum(c[i] for i in ids) <= c_max, name="change_budget")

    # Rectangle Bounds
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(x2[(l, d)] >= x1[(l, d)], name=f"x_order_{l}_{d}")
            model.addConstr(y2[(l, d)] >= y1[(l, d)], name=f"y_order_{l}_{d}")

            for i in ids:
                xi = float(stops[i].xcoord)
                yi = float(stops[i].ycoord)

                # If z_ild = 1, rectangle (l,d) must contain stop i.
                model.addConstr(
                    x1[(l, d)] <= xi + MX * (1 - z[(i, l, d)]), name=f"xmin_{i}_{l}_{d}",
                )
                model.addConstr(
                    x2[(l, d)] >= xi - MX * (1 - z[(i, l, d)]), name=f"xmax_{i}_{l}_{d}",
                )
                model.addConstr(
                    y1[(l, d)] <= yi + MY * (1 - z[(i, l, d)]), name=f"ymin_{i}_{l}_{d}",
                )
                model.addConstr(
                    y2[(l, d)] >= yi - MY * (1 - z[(i, l, d)]), name=f"ymax_{i}_{l}_{d}",
                )

    eps = 1e-5

    # Inside-rectangle Indicators
    for i in ids:
        xi = float(stops[i].xcoord)
        yi = float(stops[i].ycoord)

        for l in WEEKS_local:
            for e in DAYS_5:
                L = Lside[(i, l, e)]
                R = Rside[(i, l, e)]
                B = Bside[(i, l, e)]
                U = Uside[(i, l, e)]
                Qin = q[(i, l, e)]
                Qnonvisit = qn[(i, l, e)]

                # L = 1 iff xi >= x1
                model.addConstr(xi >= x1[(l, e)] - MX * (1 - L), name=f"L_lb_{i}_{l}_{e}")
                model.addConstr(xi <= x1[(l, e)] - eps + MX * L, name=f"L_ub_{i}_{l}_{e}")

                # R = 1 iff xi <= x2
                model.addConstr(xi <= x2[(l, e)] + MX * (1 - R), name=f"R_ub_{i}_{l}_{e}")
                model.addConstr(xi >= x2[(l, e)] + eps - MX * R, name=f"R_lb_{i}_{l}_{e}")

                # B = 1 iff yi >= y1
                model.addConstr(yi >= y1[(l, e)] - MY * (1 - B), name=f"B_lb_{i}_{l}_{e}")
                model.addConstr(yi <= y1[(l, e)] - eps + MY * B, name=f"B_ub_{i}_{l}_{e}")

                # U = 1 iff yi <= y2
                model.addConstr(yi <= y2[(l, e)] + MY * (1 - U), name=f"U_ub_{i}_{l}_{e}")
                model.addConstr(yi >= y2[(l, e)] + eps - MY * U, name=f"U_lb_{i}_{l}_{e}")

                # q = L AND R AND B AND U
                model.addConstr(Qin <= L, name=f"q_le_L_{i}_{l}_{e}")
                model.addConstr(Qin <= R, name=f"q_le_R_{i}_{l}_{e}")
                model.addConstr(Qin <= B, name=f"q_le_B_{i}_{l}_{e}")
                model.addConstr(Qin <= U, name=f"q_le_U_{i}_{l}_{e}")
                model.addConstr(Qin >= L + R + B + U - 3, name=f"q_ge_all_{i}_{l}_{e}")

                # qn = q AND (1 - z_e)
                model.addConstr(Qnonvisit <= Qin, name=f"qn_le_q_{i}_{l}_{e}")
                model.addConstr(Qnonvisit <= 1 - z[(i, l, e)], name=f"qn_le_nonvisit_{i}_{l}_{e}")
                model.addConstr(Qnonvisit >= Qin - z[(i, l, e)], name=f"qn_ge_q_minus_visit_{i}_{l}_{e}")

    # Solve
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Optimization ended with status {model.Status}")

    # Extract Solution
    chosen_tuple: Dict[int, SchedTuple] = {}

    for i in ids:
        for p in pi[i]:
            if s[(i, p)].X > 0.5:
                chosen_tuple[i] = p
                break

    changed_map: Dict[int, int] = {i: int(round(c[i].X)) for i in ids}

    rect_term_val = sum(
        (x2[(l, d)].X - x1[(l, d)].X) + (y2[(l, d)].X - y1[(l, d)].X)
        for l in WEEKS_local for d in DAYS_5
    )

    node_overlap_val = sum(
        qn[(i, l, e)].X
        for i in ids for l in WEEKS_local for e in DAYS_5
    )

    print(f"\n[MODEL #4] RECT_SIZE term          = {rect_term_val:.6f}")
    print(f"[MODEL #4] RECT_NODE_OVERLAP term = {node_overlap_val:.6f}")
    print(f"[MODEL #4] Objective              = {w1 * rect_term_val + w2 * node_overlap_val:.6f}")
    print(f"[MODEL #4] Objective type         = Linear binary objective")

    return model, chosen_tuple, changed_map