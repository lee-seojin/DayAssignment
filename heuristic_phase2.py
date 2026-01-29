from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from data_type import Stop, SchedTuple, Cell
from helper_funcs import WEEKS, DAYS_5, a, get_dist, try_apply_change


# Volume utilities
def compute_V_cell(stops: Dict[int, Stop], p: Dict[int, SchedTuple]) -> Dict[Cell, float]:
    V = {(l, d): 0.0 for l in WEEKS for d in DAYS_5}
    for sid, s in stops.items():
        vol = float(s.volume)
        sched = p[sid]
        for l in WEEKS:
            for d in DAYS_5:
                if a(sched, l, d):
                    V[(l, d)] += vol
    return V


def volume_balance_term(V: Dict[Cell, float]) -> float:
    tot = 0.0
    for l in WEEKS:
        vals = [V[(l, d)] for d in DAYS_5]
        tot += max(vals) - min(vals)
    return tot


def pick_week_with_max_imbalance(V: Dict[Cell, float]) -> int:
    return max(
        WEEKS,
        key=lambda l: max(V[(l, d)] for d in DAYS_5) - min(V[(l, d)] for d in DAYS_5)
    )


def pick_high_low_days(V: Dict[Cell, float], l: int) -> Tuple[str, str]:
    d_high = max(DAYS_5, key=lambda d: V[(l, d)])
    d_low = min(DAYS_5, key=lambda d: V[(l, d)])
    return d_high, d_low



# Density utilities (exact, local)
def build_visited_by_cell(stops: Dict[int, Stop], p: Dict[int, SchedTuple]) -> Dict[Cell, List[int]]:
    visited = {(l, d): [] for l in WEEKS for d in DAYS_5}
    for sid in stops:
        sched = p[sid]
        for l in WEEKS:
            for d in DAYS_5:
                if a(sched, l, d):
                    visited[(l, d)].append(sid)
    return visited


def cells_of_sched(s: SchedTuple) -> List[Cell]:
    return [(l, d) for l in WEEKS for d in DAYS_5 if a(s, l, d)]


def cell_density(ids: List[int], stops: Dict[int, Stop], Ddist) -> float:
    if len(ids) <= 1:
        return 0.0
    tot = 0.0
    for i in ids:
        best = float("inf")
        for j in ids:
            if i == j:
                continue
            d = get_dist(i, j, Ddist, stops)
            if d < best:
                best = d
        tot += best
    return tot


def delta_density(
    sid: int,
    old_sched: SchedTuple,
    new_sched: SchedTuple,
    visited: Dict[Cell, List[int]],
    stops: Dict[int, Stop],
    Ddist,
) -> float:
    old_cells = set(cells_of_sched(old_sched))
    new_cells = set(cells_of_sched(new_sched))
    affected = old_cells ^ new_cells

    delta = 0.0
    for cell in affected:
        before_ids = visited[cell]
        if cell in old_cells and cell not in new_cells:
            after_ids = [x for x in before_ids if x != sid]
        else:
            after_ids = before_ids + [sid]

        delta += cell_density(after_ids, stops, Ddist)
        delta -= cell_density(before_ids, stops, Ddist)

    return delta


def apply_to_visited(
    sid: int,
    old_sched: SchedTuple,
    new_sched: SchedTuple,
    visited: Dict[Cell, List[int]],
):
    old_cells = set(cells_of_sched(old_sched))
    new_cells = set(cells_of_sched(new_sched))

    for cell in old_cells - new_cells:
        visited[cell].remove(sid)
    for cell in new_cells - old_cells:
        visited[cell].append(sid)


# Heatmap
def plot_heatmap(
    V: Dict[Cell, float],
    title: str,
    savepath: Optional[str] = None,
):
    mat = np.zeros((len(WEEKS), len(DAYS_5)))
    for i, l in enumerate(WEEKS):
        for j, d in enumerate(DAYS_5):
            mat[i, j] = V[(l, d)]

    plt.figure(figsize=(8, 5))
    plt.imshow(mat, cmap="hot", aspect="auto")
    plt.colorbar(label="Volume")
    plt.xticks(range(len(DAYS_5)), DAYS_5)
    plt.yticks(range(len(WEEKS)), WEEKS)
    plt.title(title)

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# Density-aware Phase 2
def phase_2(
    artifacts: dict,
    p: Dict[int, SchedTuple],
    changed: Dict[int, int],
    C_used: int,
    *,
    max_iters: int = 500,
    patience: int = 30,
    max_candidates: int = 300,
    heatmap_prefix: Optional[str] = None,
) -> Tuple[Dict[int, SchedTuple], Dict[int, int], int]:

    stops: Dict[int, Stop] = artifacts["stops"]
    Ddist = artifacts["Ddist"]
    baseline = artifacts["baseline_sched"]
    sched_cache = artifacts["sched_cache"]
    priority_map = artifacts["priority_map"]
    C_max = int(artifacts["C_max"])

    V = compute_V_cell(stops, p)
    visited = build_visited_by_cell(stops, p)
    cur_vbal = volume_balance_term(V)

    eps_density = 10000.0  # density 악화 허용 상한
    K_low_days = 3  # to_cell 후보 개수
    min_vol_improve = 1e-6  # volume 개선 최소 기준

    # order by flexibility then volume
    all_ids = sorted(
        stops.keys(),
        key=lambda sid: (priority_map.get((stops[sid].dowcd, stops[sid].frequency), 999),
                         -float(stops[sid].volume))
    )

    no_improve = 0

    for it in range(max_iters):
        l0 = pick_week_with_max_imbalance(V)

        # from_cell: 가장 volume 큰 day
        d_high = max(DAYS_5, key=lambda d: V[(l0, d)])
        from_cell = (l0, d_high)

        # to_cell 후보: volume 작은 day 상위 K개
        low_days = sorted(DAYS_5, key=lambda d: V[(l0, d)])[:K_low_days]
        to_cells = [(l0, d) for d in low_days]

        from_ids = visited[from_cell]
        if len(from_ids) <= 1:
            no_improve += 1
            if no_improve >= patience:
                break
            continue

        candidates = [sid for sid in all_ids if sid in from_ids][:max_candidates]
        best = None
        # (sid, new_sched, delta_vbal, delta_den, newV)

        for sid in candidates:
            stop = stops[sid]
            old_sched = p[sid]
            vol = float(stop.volume)

            scheds = sched_cache[(stop.dowcd, stop.frequency)]
            if old_sched not in scheds:
                scheds = scheds + [old_sched]

            bw, bd = baseline[sid]
            if stop.dowlockcd:
                scheds = [s for s in scheds if s[1] == bd]
            if stop.wccd_flag:
                scheds = [s for s in scheds if s[0] == bw]

            for new_sched in scheds:
                if new_sched == old_sched:
                    continue
                if not a(old_sched, *from_cell):
                    continue
                if a(new_sched, *from_cell):
                    continue

                for to_cell in to_cells:
                    if not a(new_sched, *to_cell):
                        continue

                    # budget check
                    before = changed[sid]
                    after = 1 if new_sched != baseline[sid] else 0
                    if C_used - before + after > C_max:
                        continue

                    # volume update
                    newV = dict(V)
                    newV[from_cell] -= vol
                    newV[to_cell] += vol
                    new_vbal = volume_balance_term(newV)
                    delta_vbal = new_vbal - cur_vbal
                    if delta_vbal >= -min_vol_improve:
                        continue

                    # density delta (local exact)
                    delta_den = delta_density(
                        sid, old_sched, new_sched,
                        visited, stops, Ddist
                    )
                    if delta_den > eps_density:
                        continue

                    # 선택 기준:
                    # 1) volume 더 많이 개선
                    # 2) density 덜 악화
                    if best is None:
                        best = (sid, new_sched, delta_vbal, delta_den, newV)
                    else:
                        _, _, bv, bd, _ = best
                        if (delta_vbal < bv) or (
                                abs(delta_vbal - bv) < 1e-9 and delta_den < bd
                        ):
                            best = (sid, new_sched, delta_vbal, delta_den, newV)

        if best is None:
            no_improve += 1
            if no_improve >= patience:
                break
            continue

        # apply best move
        sid, new_sched, delta_vbal, delta_den, newV = best
        old_sched = p[sid]

        ok, C_used2 = try_apply_change(
            stop_id=sid,
            stops=stops,
            new_sched=new_sched,
            p=p,
            baseline_sched=baseline,
            changed=changed,
            C_used=C_used,
            C_max=C_max,
        )
        if not ok:
            no_improve += 1
            continue

        C_used = C_used2
        V = newV
        cur_vbal = volume_balance_term(V)
        apply_to_visited(sid, old_sched, new_sched, visited)
        no_improve = 0

    return p, changed, C_used