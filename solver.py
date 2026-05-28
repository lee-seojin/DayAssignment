from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import math
from typing import Dict, Tuple

from data_type import SchedTuple, ScheduleView, Stop, DAYS_5, ALL_DAYS_7
from optimal_utils import a, get_dist, OD_CM_TO_M, EARTH_R_M, make_run_prefix, build_k_neighborhood, load_artifacts, build_pi_from_artifacts
from optimal_overlap_nodes import solve_formulation_overlap_nodes
from heuristic_alns import alns_improve

#DATA_SET = "1027633"
#DATA_SET = "1042199"
#DATA_SET = "1004812"
DATA_SET = "1004940"

W1, W2 = 1.0, 1.0
RUN_TIME = 60*200

def filter_by_frequency_1(
    stops: Dict[int, Stop],
    baseline_sched: Dict[int, SchedTuple],
):
    selected_ids = {
        i for i, s in stops.items()
        if int(s.frequency) == 1
    }

    stops_f = {i: s for i, s in stops.items() if i in selected_ids}
    baseline_f = {i: p for i, p in baseline_sched.items() if i in selected_ids}

    return stops_f, baseline_f

def filter_by_dowcd_A(
    stops: Dict[int, Stop],
    baseline_sched: Dict[int, SchedTuple],
):
    selected_ids = {
        i for i, s in stops.items()
        if str(s.dowcd).strip().upper() == "A"
    }

    stops_f = {i: s for i, s in stops.items() if i in selected_ids}
    baseline_f = {i: p for i, p in baseline_sched.items() if i in selected_ids}

    return stops_f, baseline_f

def solve_formulation(
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

    # Model
    m = gp.Model("DayAssign_MinNearestPlusVolumeBalance_AllPairs")
    if time_limit is not None:
        m.setParam(GRB.Param.TimeLimit, time_limit)
    if mip_gap is not None:
        m.setParam(GRB.Param.MIPGap, mip_gap)

    # Variables
    x = {(i, p): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_W{p[0]}_D{p[1]}") for i in ids for p in pi[i]}
    y = {(i, l, d): m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{l}_{d}") for i in ids for l in WEEKS_local for d in DAYS_5}
    # z_{i,l,d}: min-nearest distance surrogate
    z = {(i, l, d): m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{i}_{l}_{d}") for i in ids for l in WEEKS_local for d in DAYS_5}

    neigh = build_k_neighborhood(ids=ids, Ddist=Ddist, stops=stops, k=k_neigh)
    # binary v_{i,j,l,d}: j chosen as i's (selected) neighbor on (l,d)
    v = {}
    for l in WEEKS_local:
        for d in DAYS_5:
            for i in ids:
                for j in neigh[i]:
                    v[(i, j, l, d)] = m.addVar(vtype=GRB.BINARY, name=f"v_{i}_{j}_{l}_{d}")


    c = {i: m.addVar(vtype=GRB.BINARY, name=f"c_{i}") for i in ids}
    # w_{l,d} = sum_i z_{i,l,d}
    w = {(l, d): m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w_{l}_{d}") for l in WEEKS_local for d in DAYS_5}

    # day volume
    Vday = {(l, d): m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Vday_{l}_{d}") for l in WEEKS_local for d in DAYS_5}

    # volume balancing (weekly envelope)
    Vmax = {l: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Vmax_{l}") for l in WEEKS_local}
    Vmin = {l: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Vmin_{l}") for l in WEEKS_local}

    m.update()

    # Objective
    m.setObjective(
        w1 * gp.quicksum(w[(l, d)] for l in WEEKS_local for d in DAYS_5)
        + w2 * gp.quicksum(Vmax[l] - Vmin[l] for l in WEEKS_local), GRB.MINIMIZE
    )

    # Constraints
    # (1) choose exactly one schedule per stop
    for i in ids:
        m.addConstr(gp.quicksum(x[(i, p)] for p in pi[i]) == 1, name=f"choose1_{i}")

    # (2) y definition using helper_funcs.a()
    for i in ids:
        for l in WEEKS_local:
            for d in DAYS_5:
                m.addConstr(
                    y[(i, l, d)] == gp.quicksum(a(p, l, d) * x[(i, p)] for p in pi[i]),
                    name=f"ydef_{i}_{l}_{d}"
                )

    # (3) volume cap + Vday definition & (4) weight cap
    for l in WEEKS_local:
        for d in DAYS_5:
            vol_sum = gp.quicksum(stops[i].volume * y[(i, l, d)] for i in ids)
            m.addConstr(Vday[(l, d)] == vol_sum, name=f"Vday_def_{l}_{d}")
            m.addConstr(vol_sum <= v_max, name=f"capV_{l}_{d}")

            m.addConstr(
                gp.quicksum(stops[i].weight * y[(i, l, d)] for i in ids) <= g_max,
                name=f"capG_{l}_{d}"
            )

    # (5) change indicator: c_i + x_{i,p0} = 1
    for i in ids:
        p0 = baseline_sched[i]
        m.addConstr(c[i] + x[(i, p0)] == 1, name=f"change_{i}")

    # (6) change budget
    m.addConstr(gp.quicksum(c[i] for i in ids) <= c_max, name="change_budget")

    # (7') min-nearest surrogate with selection binary v (all pairs)
    for l in WEEKS_local:
        for d in DAYS_5:
            for i in ids:
                # z is active only if i visited
                m.addConstr(z[(i, l, d)] <= M * y[(i, l, d)], name=f"z_act_{i}_{l}_{d}")

                # if i visited -> choose exactly one neighbor j (and that j must be visited)
                m.addConstr(
                    gp.quicksum(v[(i, j, l, d)] for j in neigh[i] if j != i) == y[(i, l, d)],
                    name=f"chooseNbr_{i}_{l}_{d}"
                )

                for j in neigh[i]:
                    # can choose j only if j is visited
                    m.addConstr(v[(i, j, l, d)] <= y[(j, l, d)],
                                name=f"v_le_y_{i}_{j}_{l}_{d}")

                    dij = get_dist(i, j, Ddist, stops)

                    # if v=1 then z >= dij
                    m.addConstr(z[(i, l, d)] >= dij - M * (1 - v[(i, j, l, d)]),
                                name=f"z_lb_{i}_{j}_{l}_{d}")

                    # if v=1 then z <= dij; else relaxed by M
                    m.addConstr(z[(i, l, d)] <= dij + M * (2 - y[(i, l, d)] - y[(j, l, d)]),
                                name=f"z_ub_{i}_{j}_{l}_{d}")

            # (w) definition
            m.addConstr(
                w[(l, d)] == gp.quicksum(z[(i, l, d)] for i in ids),
                name=f"wdef_{l}_{d}"
            )

    # Volume balancing envelope
    for l in WEEKS_local:
        for d in DAYS_5:
            m.addConstr(Vmax[l] >= Vday[(l, d)], name=f"Vmax_ge_{l}_{d}")
            m.addConstr(Vmin[l] <= Vday[(l, d)], name=f"Vmin_le_{l}_{d}")

    # Solve
    m.optimize()
    if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Optimization ended with status {m.Status}")

    # extract chosen schedule per stop
    chosen_tuple: Dict[int, SchedTuple] = {}
    for i in ids:
        for p in pi[i]:
            if x[(i, p)].X > 0.5:
                chosen_tuple[i] = p
                break

    return m, chosen_tuple, {i: int(round(c[i].X)) for i in ids}

def _weektuple_to_str(weekt):
    return ",".join(str(int(x)) for x in weekt)   # (1,3) -> "1,3"

def main():
    artifacts_path = Path("baseline_data_store") / f"{DATA_SET}_artifacts.pkl"
    artifacts = load_artifacts(artifacts_path)

    params = artifacts["params"]
    timecycle = artifacts["timecycle"]
    V_MAX = params["V_MAX"]
    G_MAX = params["G_MAX"]
    C_MAX = artifacts["C_max"]

    stops: Dict[int, Stop] = artifacts["stops"]
    baseline_sched: Dict[int, SchedTuple] = artifacts["baseline_sched"]
    sched_cache: Dict[Tuple[str, int], List[SchedTuple]] = artifacts["sched_cache"]
    Ddist: Dict[Tuple[int, int], float] = artifacts["Ddist"]

    #stops, baseline_sched = filter_by_dowcd_A(stops, baseline_sched)
    #stops, baseline_sched = filter_by_frequency_1(stops, baseline_sched)
    Pi = build_pi_from_artifacts(stops, sched_cache, baseline_sched)

    model, chosen_tuple, changed = solve_formulation_overlap_nodes(
        stops=stops,
        pi=Pi,
        baseline_sched=baseline_sched,
        timecycle=timecycle,
        v_max=V_MAX,
        g_max=G_MAX,
        c_max=C_MAX,
        Ddist=Ddist,
        w1=W1,
        w2=W2,
        time_limit=RUN_TIME,
        mip_gap=0.0,
    )


    """chosen_tuple, changed, alns_obj = alns_improve(
        stops=stops,
        pi=Pi,
        baseline_sched=baseline_sched,
        p_initial=chosen_tuple,
        timecycle=timecycle,
        v_max=V_MAX,
        g_max=G_MAX,
        c_max=C_MAX,
        max_iters=300,
        patience=50,
        remove_ratio=0.5,
        seed=42,
    )"""

    for s in stops.values():
        w_t, d_b = chosen_tuple[s.custno]
        s.chosen = ScheduleView(w_t, d_b)
        s.changed = changed[s.custno]

    rows = []

    for s in stops.values():
        base_week = s.baseline.week_tuple
        base_day = s.baseline.day_bits
        ch_week = s.chosen.week_tuple
        ch_day = s.chosen.day_bits

        for l in range(1, timecycle + 1):
            active = (l in ch_week)
            if not active:
                continue

            def weektuple_to_dowcd(wt) -> str:
                weeks = tuple(sorted(int(x) for x in wt))
                if weeks == (1, 2, 3, 4):
                    return "A"
                if weeks == (1, 3):
                    return "O"
                if weeks == (2, 4):
                    return "E"
                if len(weeks) == 1:
                    return str(weeks[0])
                return ",".join(str(w) for w in weeks)

            row = {
                "WEEK#": l,
                "STOP ID": s.custno,
                "PIECES": float(s.qty) if s.qty is not None else None,
                "VOLUME": float(s.volume),
                "BEF_WK_CD": weektuple_to_dowcd(base_week),
                "AFT_WK_CD": weektuple_to_dowcd(ch_week),
                "WK_FREQUENCY": int(s.frequency),
            }

            for idx, d in enumerate(ALL_DAYS_7):
                row[f"BEF_{d}"] = int(base_day[idx]) if l in base_week else 0

            for idx, d in enumerate(ALL_DAYS_7):
                row[f"AFT_{d}"] = int(ch_day[idx]) if l in ch_week else 0

            row["CHANGED"] = int(s.changed)
            row["XCOORD"] = float(s.xcoord)
            row["YCOORD"] = float(s.ycoord)

            rows.append(row)

    out_dir = Path("results_optimal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{make_run_prefix(DATA_SET)}_resultdetail.csv"
    df = pd.DataFrame(rows)

    cols = [
        "WEEK#", "STOP ID", "PIECES", "VOLUME",
        "BEF_WK_CD", "AFT_WK_CD", "WK_FREQUENCY"
    ] + [f"BEF_{d}" for d in ALL_DAYS_7] \
      + [f"AFT_{d}" for d in ALL_DAYS_7] \
      + ["CHANGED", "XCOORD", "YCOORD"]

    df = df[cols]
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Loaded artifacts: {artifacts_path}")
    print(f"Stops: {len(stops)}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()