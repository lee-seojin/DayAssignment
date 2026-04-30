from pathlib import Path
from typing import Dict, List, Tuple
import ast
import itertools

import pandas as pd

from helper_funcs import load_artifacts, a, get_dist, WEEKS, DAYS_5
from data_type import WeekTuple, DayBits, SchedTuple, TuplePools
from shapely.geometry import MultiPoint

# CSV parsing
def parse_week_value(x) -> WeekTuple:
    if x is None:
        return tuple()
    s = str(x).strip()
    if not s:
        return tuple()
    if s[0] in "([{" and s[-1] in ")]}":
        t = ast.literal_eval(s)
        return tuple(int(v) for v in t)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def parse_daybits_tuple_str(x) -> DayBits:
    t = ast.literal_eval(str(x))
    if len(t) != 7:
        raise ValueError(f"daybits must be length 7, got {t}")
    return tuple(int(v) for v in t)


def _dowcd_to_weektuple(x, pools: TuplePools, timecycle: int) -> WeekTuple:
    s = str(x).strip().upper()

    # 1-week compressed model
    if timecycle == 1:
        return pools.week([1])

    if s == "A":
        return pools.week([1, 2, 3, 4])
    if s == "O":
        return pools.week([1, 3])
    if s == "E":
        return pools.week([2, 4])
    if s in {"1", "2", "3", "4"}:
        return pools.week([int(s)])
    if "," in s:
        return pools.week([int(v.strip()) for v in s.split(",") if v.strip()])
    return tuple()


def build_p_from_result_csv(csv_path: Path, artifacts: dict) -> Dict[int, SchedTuple]:
    df = pd.read_csv(csv_path)

    if "STOP ID" not in df.columns:
        df = pd.read_csv(csv_path, skiprows=3)

    timecycle = int(artifacts["timecycle"])
    pools = TuplePools()

    if "STOP ID" in df.columns and "AFT_MON" in df.columns and "AFT_WK_CD" in df.columns:
        ALL_DAYS_7 = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        p: Dict[int, SchedTuple] = {}
        for _, r in df.iterrows():
            stop_id = int(r["STOP ID"])
            week_t = _dowcd_to_weektuple(r["AFT_WK_CD"], pools, timecycle)
            bits = tuple(int(r[f"AFT_{d}"]) for d in ALL_DAYS_7)
            day_b = pools.day(bits)
            p[stop_id] = pools.schedule(week_t, day_b)
        return p

    elif {"custno", "week_AFT", "day_AFT"}.issubset(df.columns):
        p = {}
        for _, r in df.iterrows():
            week_t = parse_week_value(r["week_AFT"])
            if timecycle == 1:
                week_t = pools.week([1])
            p[int(r["custno"])] = (
                week_t,
                parse_daybits_tuple_str(r["day_AFT"]),
            )
        return p

    elif {"stop_id", "chosen_week", "chosen_daybits"}.issubset(df.columns):
        p = {}
        for _, r in df.iterrows():
            week_t = parse_week_value(r["chosen_week"])
            if timecycle == 1:
                week_t = pools.week([1])
            p[int(r["stop_id"])] = (
                week_t,
                parse_daybits_tuple_str(r["chosen_daybits"]),
            )
        return p

    else:
        raise KeyError(f"Unknown result CSV format. columns={list(df.columns)}")


# Common preprocessing
def _get_visited(artifacts: dict, p: Dict[int, SchedTuple]) -> dict:
    ids = list(p.keys())
    timecycle = int(artifacts["timecycle"])
    weeks_local = list(range(1, timecycle + 1))

    visited = {(l, d): [] for l in weeks_local for d in DAYS_5}
    for i in ids:
        for l in weeks_local:
            for d in DAYS_5:
                if a(p[i], l, d) == 1:
                    visited[(l, d)].append(i)
    return visited


# Existing objective functions
def compute_nn_objective(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    stops, Ddist = artifacts["stops"], artifacts["Ddist"]
    visited = _get_visited(artifacts, p)

    total = 0.0
    for ids_cell in visited.values():
        if len(ids_cell) <= 1:
            continue
        for i in ids_cell:
            total += min(get_dist(i, j, Ddist, stops) for j in ids_cell if j != i)
    return total


def compute_mssc_objective(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    stops, Ddist = artifacts["stops"], artifacts["Ddist"]
    visited = _get_visited(artifacts, p)

    total = 0.0
    for ids_cell in visited.values():
        if len(ids_cell) <= 1:
            continue
        total += min(
            sum(get_dist(i, j, Ddist, stops) for i in ids_cell)
            for j in ids_cell
        )
    return total


# Rectangle model metrics
def compute_rectangle_size_term(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    stops = artifacts["stops"]
    visited = _get_visited(artifacts, p)

    total = 0.0
    for ids_cell in visited.values():
        if not ids_cell:
            continue

        xs = [float(stops[i].xcoord) for i in ids_cell]
        ys = [float(stops[i].ycoord) for i in ids_cell]

        x1_ld = min(xs)
        x2_ld = max(xs)
        y1_ld = min(ys)
        y2_ld = max(ys)

        total += (x2_ld - x1_ld) + (y2_ld - y1_ld)

    return total

def compute_rectangle_overlap_term(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    stops = artifacts["stops"]
    visited = _get_visited(artifacts, p)

    total = 0.0
    day_pairs = list(itertools.combinations(DAYS_5, 2))   # d < e only
    timecycle = int(artifacts["timecycle"])
    weeks_local = list(range(1, timecycle + 1))

    for l in weeks_local:
        rect = {}
        for d in DAYS_5:
            ids_cell = visited[(l, d)]
            if not ids_cell:
                rect[d] = None
                continue

            xs = [float(stops[i].xcoord) for i in ids_cell]
            ys = [float(stops[i].ycoord) for i in ids_cell]
            rect[d] = (min(xs), max(xs), min(ys), max(ys))   # x1, x2, y1, y2

        for d, e in day_pairs:
            rd = rect[d]
            re = rect[e]
            if rd is None or re is None:
                continue

            x1_ld, x2_ld, y1_ld, y2_ld = rd
            x1_le, x2_le, y1_le, y2_le = re

            wx = max(0.0, min(x2_ld, x2_le) - max(x1_ld, x1_le))
            wy = max(0.0, min(y2_ld, y2_le) - max(y1_ld, y1_le))

            wc = wx + wy if (wx > 0.0 and wy > 0.0) else 0.0
            total += wc

    return total


def compute_rectangle_objective(
    artifacts: dict,
    p: Dict[int, SchedTuple],
    w1: float = 1.0,
    w2: float = 1.0,
) -> Tuple[float, float, float]:
    rect_term = compute_rectangle_size_term(artifacts, p)
    overlap_term = compute_rectangle_overlap_term(artifacts, p)
    obj = w1 * rect_term + w2 * overlap_term

    return obj, rect_term, overlap_term

def compute_nho_metric(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    stops = artifacts["stops"]
    visited = _get_visited(artifacts, p)
    timecycle = int(artifacts["timecycle"])
    weeks_local = list(range(1, timecycle + 1))
    day_pairs = list(itertools.combinations(DAYS_5, 2))   # d < e only

    week_nho_values = []

    for l in weeks_local:
        hulls = {}
        active_days = []

        for d in DAYS_5:
            ids_cell = visited[(l, d)]
            if len(ids_cell) < 3:
                # 점이 2개 이하이면 polygon area가 0: NHO 계산 대상에서 제외
                hulls[d] = None
                continue

            pts = [(float(stops[i].xcoord), float(stops[i].ycoord)) for i in ids_cell]
            hull = MultiPoint(pts).convex_hull

            # convex_hull이 Polygon이 아니거나 면적이 0이면 제외
            if hull.geom_type != "Polygon" or hull.area <= 0:
                hulls[d] = None
                continue

            hulls[d] = hull
            active_days.append(d)

        n_active = len(active_days)
        if n_active < 2:
            week_nho_values.append(0.0)
            continue

        directed_overlap_sum = 0.0

        for d, e in day_pairs:
            hd = hulls.get(d)
            he = hulls.get(e)
            if hd is None or he is None:
                continue

            inter_area = hd.intersection(he).area
            if inter_area <= 0:
                continue

            directed_overlap_sum += inter_area / hd.area
            directed_overlap_sum += inter_area / he.area

        nho_l = directed_overlap_sum / (n_active * (n_active - 1))
        week_nho_values.append(nho_l)

    if not week_nho_values:
        return 0.0

    return sum(week_nho_values) / len(week_nho_values)


# Main
def main():
    ARTIFACTS_PATH = Path("baseline_data_store/1004812_artifacts.pkl")
    RESULT_CSV = Path("./results_optimal/1004812_20260428_140532_resultdetail.csv")

    W1 = 1.0
    W2 = 1.0

    artifacts = load_artifacts(ARTIFACTS_PATH)
    baseline_sched: Dict[int, SchedTuple] = artifacts["baseline_sched"]
    p_optimal = build_p_from_result_csv(RESULT_CSV, artifacts)

    for label, p in [("Baseline", baseline_sched), ("Optimal ", p_optimal)]:
        nn = compute_nn_objective(artifacts, p)
        print(f"[{label}]  NN              = {nn:>15,.2f}")

        mssc = compute_mssc_objective(artifacts, p)
        print(f"[{label}]  MSSC            = {mssc:>15,.2f}")

        nho = compute_nho_metric(artifacts, p)
        print(f"[{label}]  NHO             = {nho:>15,.6f}")

        rect_obj, rect_term, overlap_term = compute_rectangle_objective(
            artifacts, p, w1=W1, w2=W2
        )
        print(f"[{label}]  RECT_SIZE term  = {rect_term:>15,.6f}")
        print(f"[{label}]  OVERLAP term    = {overlap_term:>15,.6f}")
        print(f"[{label}]  RECT_OBJ        = {rect_obj:>15,.6f}")
        print()

    # improvements
    nn_b = compute_nn_objective(artifacts, baseline_sched)
    nn_o = compute_nn_objective(artifacts, p_optimal)
    if nn_b != 0:
        print(f"[NN improvement]          {(nn_b - nn_o) / nn_b * 100:.2f}%")

    mssc_b = compute_mssc_objective(artifacts, baseline_sched)
    mssc_o = compute_mssc_objective(artifacts, p_optimal)
    if mssc_b != 0:
        print(f"[MSSC improvement]        {(mssc_b - mssc_o) / mssc_b * 100:.2f}%")

    nho_b = compute_nho_metric(artifacts, baseline_sched)
    nho_o = compute_nho_metric(artifacts, p_optimal)
    if nho_b != 0:
        print(f"[NHO improvement]         {(nho_b - nho_o) / nho_b * 100:.2f}%")

    rect_b, rect_size_b, overlap_b = compute_rectangle_objective(artifacts, baseline_sched, w1=W1, w2=W2)
    rect_o, rect_size_o, overlap_o = compute_rectangle_objective(artifacts, p_optimal, w1=W1, w2=W2)

    if rect_size_b != 0:
        print(f"[RECT_SIZE improvement]   {(rect_size_b - rect_size_o) / rect_size_b * 100:.2f}%")
    if overlap_b != 0:
        print(f"[OVERLAP improvement]     {(overlap_b - overlap_o) / overlap_b * 100:.2f}%")
    if rect_b != 0:
        print(f"[RECT_OBJ improvement]    {(rect_b - rect_o) / rect_b * 100:.2f}%")


if __name__ == "__main__":
    main()