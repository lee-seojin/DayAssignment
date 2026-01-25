from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import itertools
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Constants (days)
DAYS = ["MON", "TUE", "WED", "THU", "FRI"]
ALL_DAYS_7 = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]  # a_(p,l,d) 인덱싱용
DAY_MAP = {
    "MONDAY": "MON", "TUESDAY": "TUE", "WEDNESDAY": "WED",
    "THURSDAY": "THU", "FRIDAY": "FRI", "SATURDAY": "SAT", "SUNDAY": "SUN",
    "MON": "MON", "TUE": "TUE", "WED": "WED", "THU": "THU", "FRI": "FRI", "SAT": "SAT", "SUN": "SUN",
}

# Schedule object of stop
@dataclass(frozen=True)
class ScheduleView:
    week_tuple: Tuple[int, ...]
    day_bits: Tuple[int, int, int, int, int, int, int]

    def week_key(self) -> Tuple[int, ...]:
        return self.week_tuple

    def day_key(self) -> Tuple[int, int, int, int, int, int, int]:
        return self.day_bits


@dataclass
class Stop:
    custno: int
    xcoord: float
    ycoord: float
    qty: Optional[float]
    volume: float
    weight: float
    sos: Optional[int]
    dowcd: str
    dowlockcd: Optional[int]
    wccd_flag: Optional[int]
    material_typ: Optional[str]
    clusterid: Optional[int]
    frequency: int

    baseline: ScheduleView
    chosen: Optional[ScheduleView] = None
    changed: Optional[int] = None


# Tuple pools (shared objects)
WeekTuple = Tuple[int, ...]                         # e.g., (1,), (1,3), (1,2,3,4)
DayBits = Tuple[int, int, int, int, int, int, int]  # e.g., (1,0,1,0,1,0,0)
SchedTuple = Tuple[WeekTuple, DayBits]              # schedule = (week_tuple, day_bits)


class Pools:
    """Intern pools immutable tuples -- we reuse identical objects across stops."""
    def __init__(self):
        self.week_pool: Dict[WeekTuple, WeekTuple] = {}
        self.day_pool: Dict[DayBits, DayBits] = {}
        self.sched_pool: Dict[SchedTuple, SchedTuple] = {}

    def week(self, weeks: Iterable[int]) -> WeekTuple:
        t = tuple(sorted(tuple(weeks)))
        if t not in self.week_pool:
            self.week_pool[t] = t
        return self.week_pool[t]

    def day(self, bits: DayBits) -> DayBits:
        if bits not in self.day_pool:
            self.day_pool[bits] = bits
        return self.day_pool[bits]

    def schedule(self, week_tuple: WeekTuple, day_bits: DayBits) -> SchedTuple:
        st = (week_tuple, day_bits)
        if st not in self.sched_pool:
            self.sched_pool[st] = st
        return self.sched_pool[st]


# Load params + darules (exactly from your files)
def load_params(params_path: Path):
    p = pd.read_csv(params_path, sep="\t").iloc[0]

    return {
        "timecycle": int(p["timecycle"]),
        "V_MAX": float(p["maxvolperday"]),
        "G_MAX": float(p["maxweightperday"]),
        "MAX_PCT_DAY_CHANGES": float(p["maxpctdaychanges"]),
        "stop_file": str(p["stop_file"]),
        "da_rules_file": str(p["da_rules_file"]),
    }

def load_darules(darules_path: Path, pools: Pools) -> Dict[int, List[DayBits]]:
    """
    Output: Dictionary[key: frequency, value: list of feasible days combinations]
    """
    df = pd.read_csv(darules_path,
                     sep="\t",
                     usecols=["freq", "mon", "tue", "wed", "thu", "fri", "sat", "sun"])
    df.columns = [c.strip().lower() for c in df.columns]

    out: Dict[int, List[DayBits]] = {}
    for _, r in df.iterrows():
        f = int(r["freq"])
        bits: DayBits = (int(r["mon"]), int(r["tue"]), int(r["wed"]), int(r["thu"]),
                         int(r["fri"]), int(r["sat"]), int(r["sun"]))
        out.setdefault(f, [])
        out[f].append(pools.day(bits))

    return out


# dowcd rules (TimeCycle=4 only as you requested)
def dowcd_to_weektuple(dowcd: str, pools: Pools) -> WeekTuple:
    s = str(dowcd).strip().upper()
    if s in {"1", "2", "3", "4"}:
        return pools.week([int(s)])
    if s == "O":
        return pools.week([1, 3])
    if s == "E":
        return pools.week([2, 4])
    if s == "A":
        return pools.week([1, 2, 3, 4])
    raise ValueError(f"Unknown dowcd: {dowcd}")


def week_options_from_dowcd(dowcd: str, pools: Pools) -> List[WeekTuple]:
    s = str(dowcd).strip().upper()
    if s in {"1", "2", "3", "4"}:
        return [pools.week([1]), pools.week([2]), pools.week([3]), pools.week([4])]
    if s in {"O", "E"}:
        return [pools.week([1, 3]), pools.week([2, 4])]
    if s == "A":
        return [pools.week([1, 2, 3, 4])]
    raise ValueError(f"Unknown dowcd: {dowcd}")


# Build Stop objects + feasible schedules (tuple-pool shared)
def daybits_from_bef_row(r: pd.Series, pools: Pools) -> DayBits:
    bits: DayBits = tuple(int(r[f"BEF_{d}"]) for d in DAYS)  # type: ignore
    return pools.day(bits)


def day_options_for_stop(freq: int, baseline_bits: DayBits, darules_map: Dict[int, List[DayBits]]) -> List[DayBits]:
    """
    Rule (2): same-freq darules patterns + baseline always included (even if not in darules)
    """
    opts = list(darules_map.get(freq, []))
    if baseline_bits not in opts:
        opts.append(baseline_bits)

    return opts


def build_stops_and_pi(aggr: pd.DataFrame, pools: Pools, darules_map: Dict[int, List[DayBits]]) \
        -> Tuple[Dict[int, Stop], Dict[int, List[SchedTuple]], Dict[int, SchedTuple]]:

    stops: Dict[int, Stop] = {}
    pi: Dict[int, List[SchedTuple]] = {}  # key: custno / value: feasible schedule options
    baseline_sched: Dict[int, SchedTuple] = {}  # key: custno / value: primary, baseline schedule

    for _, r in aggr.iterrows():
        custno = int(r["custno"])
        freq = int(r["frequency"])
        baseline_day = daybits_from_bef_row(r, pools)
        baseline_week = dowcd_to_weektuple(str(r["dowcd"]), pools)

        # feasible day / week options
        day_opts = day_options_for_stop(freq, baseline_day, darules_map)
        week_opts = week_options_from_dowcd(str(r["dowcd"]), pools)

        # schedules as pooled tuples
        sched_opts = [pools.schedule(w, d) for (w, d) in itertools.product(week_opts, day_opts)]
        base = pools.schedule(baseline_week, baseline_day)
        if base not in sched_opts:
            sched_opts.append(base)

        pi[custno] = sched_opts
        baseline_sched[custno] = base

        # Stop class holds “view” for baseline; chosen later
        s = Stop(
            custno=custno,
            xcoord=float(r["xcoord"]),
            ycoord=float(r["ycoord"]),
            qty=float(r["qty"]),
            volume=float(r["volume"]),
            weight=float(r["weight"]),
            sos=int(r["sos"]),
            dowcd=str(r["dowcd"]),
            dowlockcd=int(r["dowlockcd"]),
            wccd_flag=int(r["wccd_flag"]),
            material_typ=str(r["material_typ"]) if ("material_typ" in aggr.columns and pd.notna(r["material_typ"])) else None,
            clusterid=int(r["clusterid"]) if ("clusterid" in aggr.columns and pd.notna(r["clusterid"])) else None,
            frequency=freq,
            baseline=ScheduleView(baseline_week, baseline_day),
        )
        stops[custno] = s

    return stops, pi, baseline_sched


# Distance matrix (Euclidean)
from pathlib import Path
from typing import Dict, Tuple, Literal

def load_od_matrix(
    od_dir: Path,
    value: Literal["dist", "time"] = "dist",
    assume_symmetric: bool = False,
) -> Dict[Tuple[int, int], float]:
    """
    파일 규칙:
      - 맨 앞 컬럼 = route type (무시)
      - 뒤에서 4개 컬럼: origin, destination, dist, time
      - 동일 (origin, destination) 중복 시: 마지막 값 overwrite
    """
    od_dir = Path(od_dir)
    if not od_dir.exists():
        raise FileNotFoundError(f"OD directory not found: {od_dir}")

    od: Dict[Tuple[int, int], float] = {}

    for fpath in sorted(od_dir.iterdir()):
        if not fpath.is_file():
            continue

        with fpath.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]

                if len(parts) < 4:
                    continue
                try:
                    origin = int(parts[-4])
                    dest = int(parts[-3])
                    dist = float(parts[-2])
                    time = float(parts[-1])
                except ValueError:
                    continue

                v = dist if value == "dist" else time
                od[(origin, dest)] = v
                if assume_symmetric:
                    od[(dest, origin)] = v

    return od

def compute_k_nearest_d(ids: Iterable[int], Ddist: Dict[Tuple[int, int], float], stops: Dict[int, Stop], k: int = 1,) -> Dict[int, float]:

    ids_list = list(ids)
    k_nearest_d: Dict[int, float] = {}

    for i in ids_list:
        stop_i = stops[i]
        cand: List[float] = []
        for j in ids_list:
            if j == i: continue
            if (i, j) in Ddist:
                cand.append(Ddist[(i, j)])
            else:
                cand.append(compute_manhattan_dist(stop_i, stops[j]))

        cand.sort()
        k_nearest_d[i] = cand[min(k - 1, len(cand) - 1)]  # k=1: 최소값

    return k_nearest_d

def compute_manhattan_dist(stop_i: Stop, stop_j: Stop) -> float:
    i_x, i_y = stop_i.xcoord, stop_i.ycoord
    j_x, j_y = stop_j.xcoord, stop_j.ycoord
    dist = abs(i_x - j_x) + abs(i_y - j_y)

    return dist

def solve_formulation(
    stops: Dict[int, Stop],
    pi: Dict[int, List[SchedTuple]],
    baseline_sched: Dict[int, SchedTuple],
    timecycle: int,
    v_max: float,
    g_max: float,
    c_max: int,
    Ddist: Dict[Tuple[int, int], float],
    w1: float = 1.0,                 # stop density weight (w term)
    w2: float = 1.0,                  # volume balancing weight
    time_limit: Optional[int] = 600,
    mip_gap: Optional[float] = 0.0,
):

    WEEKS = list(range(1, timecycle + 1))
    ids = list(stops.keys())

    # Big-M
    M = max(Ddist.values())

    # a[p,l,d] based on schedule tuple
    def a(p: SchedTuple, l: int, d: str) -> int:
        week_t, day_b = p
        return int((l in week_t) and (day_b[ALL_DAYS_7.index(d)] == 1))

    m = gp.Model("DayAssign_DensityPlusVolumeBalance")
    if time_limit is not None:
        m.setParam(GRB.Param.TimeLimit, time_limit)
    if mip_gap is not None:
        m.setParam(GRB.Param.MIPGap, mip_gap)

    # Vars
    x = {(i, p): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_W{p[0]}_D{p[1]}")
         for i in ids for p in pi[i]}

    y = {(i, l, d): m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{l}_{d}")
         for i in ids for l in WEEKS for d in DAYS}

    c = {i: m.addVar(vtype=GRB.BINARY, name=f"c_{i}") for i in ids}

    z = {(i, l, d): m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{i}_{l}_{d}")
         for i in ids for l in WEEKS for d in DAYS}

    w = {(l, d): m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w_{l}_{d}")
         for l in WEEKS for d in DAYS}

    # day volume variable (per week/day)
    Vday = {(l, d): m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Vday_{l}_{d}")
            for l in WEEKS for d in DAYS}

    # per-week max/min day volume for balancing
    Vmax = {l: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Vmax_{l}") for l in WEEKS}
    Vmin = {l: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Vmin_{l}") for l in WEEKS}

    m.update()

    # Objective
    m.setObjective(
        w1 * gp.quicksum(w[(l, d)] for l in WEEKS for d in DAYS)
        + w2 * gp.quicksum(Vmax[l] - Vmin[l] for l in WEEKS),
        GRB.MINIMIZE
    )

    # Constraints
    # (1) choose exactly one schedule per stop
    for i in ids:
        m.addConstr(gp.quicksum(x[(i, p)] for p in pi[i]) == 1, name=f"choose1_{i}")

    # (2) y definition
    for i in ids:
        for l in WEEKS:
            for d in DAYS:
                m.addConstr(
                    y[(i, l, d)] == gp.quicksum(a(p, l, d) * x[(i, p)] for p in pi[i]),
                    name=f"ydef_{i}_{l}_{d}"
                )

    # (3) volume cap + Vday definition & (4) weight cap (keep if you still want it)
    for l in WEEKS:
        for d in DAYS:
            vol_sum = gp.quicksum(stops[i].volume * y[(i, l, d)] for i in ids)
            m.addConstr(Vday[(l, d)] == vol_sum, name=f"Vday_def_{l}_{d}")
            m.addConstr(vol_sum <= v_max, name=f"capV_{l}_{d}")

            m.addConstr(
                gp.quicksum(stops[i].weight * y[(i, l, d)] for i in ids) <= g_max,
                name=f"capG_{l}_{d}"
            )

    # (5) c_i = 1 - x_{i,p0}  <=> c_i + x_{i,p0} = 1
    for i in ids:
        p0 = baseline_sched[i]
        m.addConstr(c[i] + x[(i, p0)] == 1, name=f"change_{i}")

    # (6) change budget
    m.addConstr(gp.quicksum(c[i] for i in ids) <= c_max, name="change_budget")

    # (7) z_{i,l,d} >= D_{i,j} - M(2 - y_i - y_j) for all i!=j (only if OD exists)
    # (8) z_{i,l,d} <= M y_{i,l,d}
    for l in WEEKS:
        for d in DAYS:
            for i in ids:
                m.addConstr(z[(i, l, d)] <= M * y[(i, l, d)], name=f"zub_{i}_{l}_{d}")
                for j in ids:
                    if j == i:
                        continue
                    if (i, j) not in Ddist:
                        continue
                    m.addConstr(
                        z[(i, l, d)] >= Ddist[(i, j)] - M * (2 - y[(i, l, d)] - y[(j, l, d)]),
                        name=f"zlb_{i}_{j}_{l}_{d}"
                    )

            # (9) w_{l,d} = sum_i z_{i,l,d}
            m.addConstr(
                w[(l, d)] == gp.quicksum(z[(i, l, d)] for i in ids),
                name=f"wdef_{l}_{d}"
            )

    # (10)(11) Volume balancing: Vmax/Vmin envelope constraints
    for l in WEEKS:
        for d in DAYS:
            m.addConstr(Vmax[l] >= Vday[(l, d)], name=f"Vmax_ge_{l}_{d}")
            m.addConstr(Vmin[l] <= Vday[(l, d)], name=f"Vmin_le_{l}_{d}")

    # (12) Tighter constraint range for z value in LP relaxation stage
    dmin = compute_k_nearest_d(ids, Ddist, stops, k=1)  # k=1 또는 더 완화하려면 k=2~3

    for i in ids:
        for l in WEEKS:
            for d in DAYS:
                m.addConstr(
                    z[(i, l, d)] >= dmin[i] * y[(i, l, d)],
                    name=f"zmin_{i}_{l}_{d}"
                )

    # Solve
    m.optimize()
    if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Optimization ended with status {m.Status}")

    # extract chosen schedule per stop (tuple)
    chosen_tuple: Dict[int, SchedTuple] = {}
    for i in ids:
        for p in pi[i]:
            if x[(i, p)].X > 0.5:
                chosen_tuple[i] = p
                break

    return m, chosen_tuple, {i: int(round(c[i].X)) for i in ids}


# Main
def main():

    DATA_SET = "1042199"

    PARAMS_PATH = Path(f"{DATA_SET}/params.txt")

    params = load_params(PARAMS_PATH)
    timecycle = params["timecycle"]
    V_MAX = params["V_MAX"]
    G_MAX = params["G_MAX"]
    max_pct = params["MAX_PCT_DAY_CHANGES"]

    # 1) aggregate raw stops -> stop-level with BEF_* + frequency
    aggr = pd.read_csv(f"{DATA_SET}/{DATA_SET}_stops_aggregated.csv")

    pools = Pools()

    # 2) read darules (freq -> pooled daybits)
    darules_path = Path(f"{DATA_SET}/darules.txt")
    darules_map = load_darules(darules_path, pools)

    # 3) build Stop objects + Pi (tuple schedules) + baseline schedule tuple
    stops, Pi, baseline_sched = build_stops_and_pi(aggr, pools, darules_map)

    n = len(stops)
    C_MAX = int(round(max_pct / 100.0 * n))

    # 4) distances (Euclidean)
    Ddist = load_od_matrix(Path('od_info'))

    # 5) solve (new objective formulation)
    model, chosen_tuple, changed = solve_formulation(stops=stops, pi=Pi, baseline_sched=baseline_sched,
                                                     timecycle=timecycle, v_max=V_MAX, g_max=G_MAX, c_max=C_MAX,
                                                     Ddist=Ddist, w1=1.0, w2=1.0, time_limit=600, mip_gap=0.0)

    # 6) write chosen schedules back to Stop class (final layer = class)
    for s in stops.values():
        w_t, d_b = chosen_tuple[s.custno]
        s.chosen = ScheduleView(w_t, d_b)
        s.changed = changed[s.custno]

    # 7) export resultdetail-like
    rows = []
    for s in stops.values():
        assert s.chosen is not None
        rows.append({
            "custno": s.custno,
            "volume": s.volume,
            "weight": s.weight,
            "material_typ": s.material_typ,
            "frequency": s.frequency,
            "dowcd_BEF": s.dowcd,
            "week_BEF": s.baseline.week_key(),
            "day_BEF": s.baseline.day_key(),
            "changed": s.changed,
            "week_AFT": s.chosen.week_key(),
            "day_AFT": s.chosen.day_key(),
            **{f"AFT_{d}": int(s.chosen.day_bits[DAYS.index(d)]) for d in DAYS},
        })

    out = pd.DataFrame(rows)
    out.to_csv("resultdetail_gurobi_tuplepool.csv", index=False, encoding="utf-8-sig")
    print("Saved: resultdetail_gurobi_tuplepool.csv")


if __name__ == "__main__":
    main()
