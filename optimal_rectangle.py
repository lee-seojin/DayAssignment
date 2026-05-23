from __future__ import annotations

import itertools
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from data_type import SchedTuple, Stop, DAYS_5
from optimal_utils import a


def solve_formulation_rectangle(
    stops: Dict[int, Stop],
    pi: Dict[int, List[SchedTuple]],
    baseline_sched: Dict[int, SchedTuple],
    timecycle: int,
    v_max: float,
    g_max: float,
    c_max: int,
    Ddist: Dict[Tuple[int, int], float],   # 시그니처 통일용
    w1: float = 1.0,
    w2: float = 1.0,
    time_limit: Optional[int] = 600,
    mip_gap: Optional[float] = 0.0,
):
    WEEKS_local = list(range(1, timecycle + 1))
    ids = list(stops.keys())
    day_pairs = list(itertools.combinations(DAYS_5, 2))

    x_is = [float(stops[i].xcoord) for i in ids]
    y_is = [float(stops[i].ycoord) for i in ids]

    XMIN, XMAX = min(x_is), max(x_is)
    YMIN, YMAX = min(y_is), max(y_is)

    MX = max(XMAX - XMIN, 1.0)
    MY = max(YMAX - YMIN, 1.0)
    MWC = MX + MY

    model = gp.Model("Optimal_Rectangle")
    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)
    if mip_gap is not None:
        model.setParam(GRB.Param.MIPGap, mip_gap)

    # Decision Variables
    # s_ip
    s = {(i, p): model.addVar(vtype=GRB.BINARY, name=f"s_{i}_W{p[0]}_D{p[1]}")
         for i in ids for p in pi[i]}

    # z_ild
    z = {(i, l, d): model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{l}_{d}")
         for i in ids for l in WEEKS_local for d in DAYS_5}

    # c_i
    c = {i: model.addVar(vtype=GRB.BINARY, name=f"c_{i}") for i in ids}

    # x_ld^1, x_ld^2, y_ld^1, y_ld^2
    x1_ld = {(l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=XMIN, ub=XMAX, name=f"x1_{l}_{d}")
             for l in WEEKS_local for d in DAYS_5}
    x2_ld = {(l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=XMIN, ub=XMAX, name=f"x2_{l}_{d}")
             for l in WEEKS_local for d in DAYS_5}
    y1_ld = {(l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=YMIN, ub=YMAX, name=f"y1_{l}_{d}")
             for l in WEEKS_local for d in DAYS_5}
    y2_ld = {(l, d): model.addVar(vtype=GRB.CONTINUOUS, lb=YMIN, ub=YMAX, name=f"y2_{l}_{d}")
             for l in WEEKS_local for d in DAYS_5}

    # x_lde^1, x_lde^2, y_lde^1, y_lde^2
    x1_lde = {(l, d, e): model.addVar(vtype=GRB.CONTINUOUS, lb=XMIN, ub=XMAX, name=f"x1_{l}_{d}_{e}")
              for l in WEEKS_local for d, e in day_pairs}
    x2_lde = {(l, d, e): model.addVar(vtype=GRB.CONTINUOUS, lb=XMIN, ub=XMAX, name=f"x2_{l}_{d}_{e}")
              for l in WEEKS_local for d, e in day_pairs}
    y1_lde = {(l, d, e): model.addVar(vtype=GRB.CONTINUOUS, lb=YMIN, ub=YMAX, name=f"y1_{l}_{d}_{e}")
              for l in WEEKS_local for d, e in day_pairs}
    y2_lde = {(l, d, e): model.addVar(vtype=GRB.CONTINUOUS, lb=YMIN, ub=YMAX, name=f"y2_{l}_{d}_{e}")
              for l in WEEKS_local for d, e in day_pairs}

    # m^x_lde, m^y_lde for min selectors
    m_x_lde = {(l, d, e): model.addVar(vtype=GRB.BINARY, name=f"mx_{l}_{d}_{e}")
               for l in WEEKS_local for d, e in day_pairs}
    m_y_lde = {(l, d, e): model.addVar(vtype=GRB.BINARY, name=f"my_{l}_{d}_{e}")
               for l in WEEKS_local for d, e in day_pairs}

    # M^x_lde, M^y_lde for max selectors
    M_x_lde = {(l, d, e): model.addVar(vtype=GRB.BINARY, name=f"Mx_{l}_{d}_{e}")
               for l in WEEKS_local for d, e in day_pairs}
    M_y_lde = {(l, d, e): model.addVar(vtype=GRB.BINARY, name=f"My_{l}_{d}_{e}")
               for l in WEEKS_local for d, e in day_pairs}

    # w_lde^x, w_lde^y, w_lde^c
    wx_lde = {(l, d, e): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"wx_{l}_{d}_{e}")
              for l in WEEKS_local for d, e in day_pairs}
    wy_lde = {(l, d, e): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"wy_{l}_{d}_{e}")
              for l in WEEKS_local for d, e in day_pairs}
    wc_lde = {(l, d, e): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"wc_{l}_{d}_{e}")
              for l in WEEKS_local for d, e in day_pairs}

    # o_lde^x, o_lde^y, o_lde^c
    ox_lde = {(l, d, e): model.addVar(vtype=GRB.BINARY, name=f"ox_{l}_{d}_{e}")
              for l in WEEKS_local for d, e in day_pairs}
    oy_lde = {(l, d, e): model.addVar(vtype=GRB.BINARY, name=f"oy_{l}_{d}_{e}")
              for l in WEEKS_local for d, e in day_pairs}
    oc_lde = {(l, d, e): model.addVar(vtype=GRB.BINARY, name=f"oc_{l}_{d}_{e}")
              for l in WEEKS_local for d, e in day_pairs}

    model.update()

    # Objective Function
    rect_size = gp.quicksum(
        (x2_ld[(l, d)] - x1_ld[(l, d)]) + (y2_ld[(l, d)] - y1_ld[(l, d)])
        #(x2_ld[(l, d)] - x1_ld[(l, d)]) * (y2_ld[(l, d)] - y1_ld[(l, d)])
    for l in WEEKS_local for d in DAYS_5
    )

    overlap_size = gp.quicksum(
        wc_lde[(l, d, e)] for l in WEEKS_local for d, e in day_pairs
    )

    model.setObjective(w1 * rect_size + w2 * overlap_size, GRB.MINIMIZE)

    # Constraints
    # (1) Pattern assign
    for i in ids:
        model.addConstr(
            gp.quicksum(s[(i, p)] for p in pi[i]) == 1,
            name=f"pattern_assign_{i}"
        )

    # (2) Day assign
    for i in ids:
        for l in WEEKS_local:
            for d in DAYS_5:
                model.addConstr(
                    z[(i, l, d)] == gp.quicksum(a(p, l, d) * s[(i, p)] for p in pi[i]),
                    name=f"day_assign_{i}_{l}_{d}"
                )

    # (3) Volume constraint
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(
                gp.quicksum(stops[i].volume * z[(i, l, d)] for i in ids) <= v_max,
                name=f"capV_{l}_{d}"
            )

    # (4) Weight constraint
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(
                gp.quicksum(stops[i].weight * z[(i, l, d)] for i in ids) <= g_max,
                name=f"capG_{l}_{d}"
            )

    # (5) Pattern change
    for i in ids:
        p0 = baseline_sched[i]
        model.addConstr(c[i] + s[(i, p0)] == 1, name=f"change_{i}")

    # (6) Pattern alternation constraint
    model.addConstr(
        gp.quicksum(c[i] for i in ids) <= c_max, name="change_budget")

    # (7) Minimum x coordinate
    for l in WEEKS_local:
        for d in DAYS_5:
            for i in ids:
                xi = float(stops[i].xcoord)
                model.addConstr(
                    x1_ld[(l, d)] <= xi + MX * (1 - z[(i, l, d)]),
                    name=f"xmin_{i}_{l}_{d}"
                )

    # (8) Minimum y coordinate
    for l in WEEKS_local:
        for d in DAYS_5:
            for i in ids:
                yi = float(stops[i].ycoord)
                model.addConstr(
                    y1_ld[(l, d)] <= yi + MY * (1 - z[(i, l, d)]),
                    name=f"ymin_{i}_{l}_{d}"
                )

    # (9) Maximum x coordinate
    for l in WEEKS_local:
        for d in DAYS_5:
            for i in ids:
                xi = float(stops[i].xcoord)
                model.addConstr(
                    x2_ld[(l, d)] >= xi - MX * (1 - z[(i, l, d)]),
                    name=f"xmax_{i}_{l}_{d}"
                )

    # (10) Maximum y coordinate
    for l in WEEKS_local:
        for d in DAYS_5:
            for i in ids:
                yi = float(stops[i].ycoord)
                model.addConstr(
                    y2_ld[(l, d)] >= yi - MY * (1 - z[(i, l, d)]),
                    name=f"ymax_{i}_{l}_{d}"
                )

    # rectangle ordering
    for l in WEEKS_local:
        for d in DAYS_5:
            model.addConstr(x2_ld[(l, d)] >= x1_ld[(l, d)], name=f"x_order_{l}_{d}")
            model.addConstr(y2_ld[(l, d)] >= y1_ld[(l, d)], name=f"y_order_{l}_{d}")

    # (11) ~ (18) big rectangle including day d, e
    for l in WEEKS_local:
        for d, e in day_pairs:
            model.addConstr(x1_lde[(l, d, e)] <= x1_ld[(l, d)], name=f"x1lde_d_{l}_{d}_{e}")
            model.addConstr(x1_lde[(l, d, e)] <= x1_ld[(l, e)], name=f"x1lde_e_{l}_{d}_{e}")
            model.addConstr(
                x1_lde[(l, d, e)] >= x1_ld[(l, d)] - MX * (1 - m_x_lde[(l, d, e)]),
                name=f"x1ge_1_{l}_{d}_{e}"
            )
            model.addConstr(
                x1_lde[(l, d, e)] >= x1_ld[(l, e)] - MX * m_x_lde[(l, d, e)],
                name=f"x1ge_2_{l}_{d}_{e}"
            )

            model.addConstr(y1_lde[(l, d, e)] <= y1_ld[(l, d)], name=f"y1lde_d_{l}_{d}_{e}")
            model.addConstr(y1_lde[(l, d, e)] <= y1_ld[(l, e)], name=f"y1lde_e_{l}_{d}_{e}")
            model.addConstr(
                y1_lde[(l, d, e)] >= y1_ld[(l, d)] - MY * (1 - m_y_lde[(l, d, e)]),
                name=f"y1ge_1_{l}_{d}_{e}"
            )
            model.addConstr(
                y1_lde[(l, d, e)] >= y1_ld[(l, e)] - MY * m_y_lde[(l, d, e)],
                name=f"y1ge_2_{l}_{d}_{e}"
            )

            model.addConstr(x2_lde[(l, d, e)] >= x2_ld[(l, d)], name=f"x2lde_d_{l}_{d}_{e}")
            model.addConstr(x2_lde[(l, d, e)] >= x2_ld[(l, e)], name=f"x2lde_e_{l}_{d}_{e}")
            model.addConstr(
                x2_lde[(l, d, e)] <= x2_ld[(l, d)] + MX * (1 - M_x_lde[(l, d, e)]),
                name=f"x2le_1_{l}_{d}_{e}"
            )
            model.addConstr(
                x2_lde[(l, d, e)] <= x2_ld[(l, e)] + MX * M_x_lde[(l, d, e)],
                name=f"x2le_2_{l}_{d}_{e}"
            )

            model.addConstr(y2_lde[(l, d, e)] >= y2_ld[(l, d)], name=f"y2lde_d_{l}_{d}_{e}")
            model.addConstr(y2_lde[(l, d, e)] >= y2_ld[(l, e)], name=f"y2lde_e_{l}_{d}_{e}")
            model.addConstr(
                y2_lde[(l, d, e)] <= y2_ld[(l, d)] + MY * (1 - M_y_lde[(l, d, e)]),
                name=f"y2le_1_{l}_{d}_{e}"
            )
            model.addConstr(
                y2_lde[(l, d, e)] <= y2_ld[(l, e)] + MY * M_y_lde[(l, d, e)],
                name=f"y2le_2_{l}_{d}_{e}"
            )

            model.addConstr(x2_lde[(l, d, e)] >= x1_lde[(l, d, e)], name=f"xpair_order_{l}_{d}_{e}")
            model.addConstr(y2_lde[(l, d, e)] >= y1_lde[(l, d, e)], name=f"ypair_order_{l}_{d}_{e}")

    # (19) Overlap x-axis
    for l in WEEKS_local:
        for d, e in day_pairs:
            model.addConstr(
                wx_lde[(l, d, e)] >= (x2_ld[(l, d)] - x1_ld[(l, d)]) + (x2_ld[(l, e)] - x1_ld[(l, e)]) - (x2_lde[(l, d, e)] - x1_lde[(l, d, e)]),
                name=f"wx_{l}_{d}_{e}"
            )

    # (20) Overlap y-axis
    for l in WEEKS_local:
        for d, e in day_pairs:
            model.addConstr(
                wy_lde[(l, d, e)] >= (y2_ld[(l, d)] - y1_ld[(l, d)]) + (y2_ld[(l, e)] - y1_ld[(l, e)]) - (y2_lde[(l, d, e)] - y1_lde[(l, d, e)]),
                name=f"wy_{l}_{d}_{e}"
            )

    # (21) if overlap exist in x-axis
    for l in WEEKS_local:
        for d, e in day_pairs:
            model.addConstr(
                wx_lde[(l, d, e)] <= MX * ox_lde[(l, d, e)],
                name=f"ox_link_{l}_{d}_{e}"
            )

    # (22) if overlap exist in y-axis
    for l in WEEKS_local:
        for d, e in day_pairs:
            model.addConstr(
                wy_lde[(l, d, e)] <= MY * oy_lde[(l, d, e)],
                name=f"oy_link_{l}_{d}_{e}"
            )

    # (23) Overlap count needed if both overlapped
    for l in WEEKS_local:
        for d, e in day_pairs:
            model.addConstr(
                oc_lde[(l, d, e)] >= ox_lde[(l, d, e)] + oy_lde[(l, d, e)] - 1,
                name=f"oc_lb_{l}_{d}_{e}"
            )
            model.addConstr(
                oc_lde[(l, d, e)] <= ox_lde[(l, d, e)],
                name=f"oc_ubx_{l}_{d}_{e}"
            )
            model.addConstr(
                oc_lde[(l, d, e)] <= oy_lde[(l, d, e)],
                name=f"oc_uby_{l}_{d}_{e}"
            )

    # (24) Overlap amount
    for l in WEEKS_local:
        for d, e in day_pairs:
            model.addConstr(
                wc_lde[(l, d, e)] >= wx_lde[(l, d, e)] + wy_lde[(l, d, e)] - MWC * (1 - oc_lde[(l, d, e)]),
                # wc_lde[(l, d, e)] >= wx_lde[(l, d, e)] * wy_lde[(l, d, e)] - MWC * (1 - oc_lde[(l, d, e)]),
                name=f"wc_{l}_{d}_{e}"
            )

    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Optimization ended with status {model.Status}")

    chosen_tuple: Dict[int, SchedTuple] = {}
    for i in ids:
        for p in pi[i]:
            if s[(i, p)].X > 0.5:
                chosen_tuple[i] = p
                break

    changed_map: Dict[int, int] = {i: int(round(c[i].X)) for i in ids}

    rect_term_val = sum(
        (x2_ld[(l, d)].X - x1_ld[(l, d)].X) + (y2_ld[(l, d)].X - y1_ld[(l, d)].X)
        for l in WEEKS_local for d in DAYS_5
    )

    overlap_term_val = sum(
        wc_lde[(l, d, e)].X for l in WEEKS_local for d, e in day_pairs
    )

    print(f"\n[MODEL] RECT_SIZE term = {rect_term_val:.6f}")
    print(f"[MODEL] OVERLAP term   = {overlap_term_val:.6f}")
    print(f"[MODEL] RECT_OBJ kk      = {w1*rect_term_val + w2*overlap_term_val:.6f}\n")

    return model, chosen_tuple, changed_map