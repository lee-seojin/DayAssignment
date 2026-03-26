from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from data_type import Stop, SchedTuple, Cell
from helper_funcs import WEEKS, DAYS_5, a, get_dist, try_apply_change

def build_priority_groups(
    stops: Dict[int, Stop],
    priority_map: Dict[Tuple[str, int], int],
) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {}
    for sid, s in stops.items():
        pr = priority_map.get((s.dowcd, s.frequency))
        if pr is None:
            continue
        groups.setdefault(pr, []).append(sid)
    for pr in groups:
        groups[pr].sort(key=lambda i: (stops[i].frequency, i))
    return groups


def visited_by_cell(stops: Dict[int, Stop], p: Dict[int, SchedTuple]) -> Dict[Cell, List[int]]:
    out: Dict[Cell, List[int]] = {(l, d): [] for l in WEEKS for d in DAYS_5}
    for sid in stops:
        sched = p[sid]
        for l in WEEKS:
            for d in DAYS_5:
                if a(sched, l, d) == 1:
                    out[(l, d)].append(sid)
    return out


def stop_cells(sched: SchedTuple) -> List[Cell]:
    cells: List[Cell] = []
    for l in WEEKS:
        for d in DAYS_5:
            if a(sched, l, d) == 1:
                cells.append((l, d))
    return cells


def filter_candidates_by_locks(
    sid: int,
    stop: Stop,
    candidates: List[SchedTuple],
    baseline_sched: Dict[int, SchedTuple],
) -> List[SchedTuple]:
    base_w, base_d = baseline_sched[sid]
    out = candidates
    if getattr(stop, "dowlockcd", 0) == 1:
        out = [s for s in out if s[1] == base_d]
    if getattr(stop, "wccd_flag", 0) == 1:
        out = [s for s in out if s[0] == base_w]
    return out


def min_dist_to_set(
    sid: int,
    targets: List[int],
    stops: Dict[int, Stop],
    Ddist: Dict[Tuple[int, int], float],
) -> float:
    """min_j dist(sid, j). targets empty -> +inf"""
    if not targets:
        return float("inf")
    best = float("inf")
    for j in targets:
        if j == sid:
            continue
        d = get_dist(sid, j, Ddist, stops)
        if d < best:
            best = d
    return best


def core_proxy_cost_for_sched(
    sid: int,
    sched: SchedTuple,
    core_by_cell: Dict[Cell, List[int]],
    visited_by_cell_current: Dict[Cell, List[int]],
    stops: Dict[int, Stop],
    Ddist: Dict[Tuple[int, int], float],
    empty_cell_penalty: float = 1e3,
) -> float:
    """
    proxy cost = sum over visited cells:
      - if core exists in that cell: min dist to that cell's core
      - else: min dist to existing stops in that cell (so we don't jump into isolation)
      - if even that cell is empty: empty_cell_penalty
    """
    cost = 0.0
    for cell in stop_cells(sched):
        cores = core_by_cell.get(cell, [])
        if cores:
            dmin = min_dist_to_set(sid, cores, stops, Ddist)
            cost += dmin
            continue

        # core 없는 셀: 그 셀에 이미 있는 방문자들에 붙는 비용 사용
        members = visited_by_cell_current.get(cell, [])
        dmin2 = min_dist_to_set(sid, members, stops, Ddist)
        if dmin2 == float("inf"):
            cost += empty_cell_penalty
        else:
            cost += dmin2

    return cost


def choose_outliers_in_cell(
    ids: List[int],
    core_ids: List[int],
    stops: Dict[int, Stop],
    Ddist: Dict[Tuple[int, int], float],
    top_ratio: float,
) -> List[int]:
    """
    Rank by distance-to-nearest-core (bigger = more outlier).
    Return top_ratio portion (at least 1).
    """
    if not ids or not core_ids:
        return []

    scored = []
    for sid in ids:
        # core stop itself is not a target to move
        if sid in core_ids:
            continue
        d = min_dist_to_set(sid, core_ids, stops, Ddist)
        scored.append((d, sid))
    if not scored:
        return []

    scored.sort(reverse=True)  # farthest first
    k = max(1, int(len(scored) * top_ratio))
    return [sid for _, sid in scored[:k]]


def phase_1(
    artifacts: dict,
    p: Dict[int, SchedTuple],
    changed: Dict[int, int],
    C_used: int
) -> Tuple[Dict[int, List[int]], Dict[int, int], Dict[int, SchedTuple], Dict[int, int], int]:
    """
    Phase1' (core-anchored incremental relocate)

    - core stops = pr == best_pr (fixed anchor)
    - start from current p (usually baseline)
    - iteratively pick far-from-core stops within a cell and try to relocate them
      to schedules that reduce "core-proxy cost"
    - accept only improving proxy moves
    - respects locks and budget

    Returns clusters/nucleus dummy outputs for compatibility.
    """

    stops: Dict[int, Stop] = artifacts["stops"]
    Ddist: Dict[Tuple[int, int], float] = artifacts["Ddist"]
    sched_cache: Dict[Tuple[str, int], List[SchedTuple]] = artifacts["sched_cache"]
    priority_map: Dict[Tuple[str, int], int] = artifacts["priority_map"]
    baseline_sched: Dict[int, SchedTuple] = artifacts["baseline_sched"]
    C_max: int = int(artifacts["C_max"])

    # params
    max_outer_iters = 300
    top_ratio = 0.20         # "셀 내에서 core로부터 먼 애" 상위 20%
    max_cells_per_iter = 20  # 한 바퀴에 최대 몇 cell 볼지 (20이면 전체)
    max_outliers_per_cell = 25
    eps_improve = 1e-9

    # 0) identify core stops
    priority_groups = build_priority_groups(stops, priority_map)
    if not priority_groups:
        # fallback: nothing to do
        clusters = {0: list(stops.keys())}
        nucleus = {0: clusters[0][0] if clusters[0] else -1}
        return clusters, nucleus, p, changed, C_used

    best_pr = min(priority_groups.keys())
    core_ids_all = set(priority_groups[best_pr])

    # 1) build core_by_cell (based on current p; core usually baseline-fixed)
    #    We'll rebuild periodically because p changes.
    def rebuild_core_by_cell() -> Dict[Cell, List[int]]:
        core_by_cell: Dict[Cell, List[int]] = {(l, d): [] for l in WEEKS for d in DAYS_5}
        for sid in core_ids_all:
            sched = p[sid]
            for l in WEEKS:
                for d in DAYS_5:
                    if a(sched, l, d) == 1:
                        core_by_cell[(l, d)].append(sid)
        return core_by_cell

    # main loop
    for _ in range(max_outer_iters):
        vbc = visited_by_cell(stops, p)
        core_by_cell = rebuild_core_by_cell()

        # rank cells by "how many non-core are far from core" proxy
        cell_rank = []
        for cell, ids in vbc.items():
            cores = core_by_cell.get(cell, [])
            if not cores:
                continue
            # compute max distance-to-core among non-core in this cell
            far = 0.0
            for sid in ids:
                if sid in core_ids_all:
                    continue
                d = min_dist_to_set(sid, cores, stops, Ddist)
                if d > far:
                    far = d
            if far > 0:
                cell_rank.append((far, cell))

        if not cell_rank:
            break

        cell_rank.sort(reverse=True)
        cells_to_check = [cell for _, cell in cell_rank[:max_cells_per_iter]]

        improved_any = False

        for cell in cells_to_check:
            ids = vbc[cell]
            cores = core_by_cell.get(cell, [])

            outliers = choose_outliers_in_cell(ids, cores, stops, Ddist, top_ratio=top_ratio)
            outliers = outliers[:max_outliers_per_cell]

            # among outliers, move "more flexible" first => higher pr value means less constrained
            outliers.sort(key=lambda sid: priority_map.get((stops[sid].dowcd, stops[sid].frequency), 999), reverse=True)

            for sid in outliers:
                if sid in core_ids_all:
                    continue

                stop = stops[sid]
                old_sched = p[sid]

                candidates = sched_cache.get((stop.dowcd, stop.frequency), []).copy()
                if old_sched not in candidates:
                    candidates.append(old_sched)

                candidates = filter_candidates_by_locks(sid, stop, candidates, baseline_sched)

                # must be currently visiting this cell
                l0, d0 = cell
                if a(old_sched, l0, d0) != 1:
                    continue

                old_cost = core_proxy_cost_for_sched(
                    sid, old_sched, core_by_cell, vbc, stops, Ddist, empty_cell_penalty=1e3
                )

                best_cost = old_cost
                best_sched: Optional[SchedTuple] = None

                for new_sched in candidates:
                    if new_sched == old_sched:
                        continue

                    # must leave the problematic cell
                    if a(new_sched, l0, d0) == 1:
                        continue

                    # budget check (exactly same as try_apply_change would do)
                    before = changed[sid]
                    after = 1 if new_sched != baseline_sched[sid] else 0
                    if (C_used - before + after) > C_max:
                        continue

                    new_cost = core_proxy_cost_for_sched(
                        sid, new_sched, core_by_cell, vbc, stops, Ddist, empty_cell_penalty=1e3)

                    if new_cost + eps_improve < best_cost:
                        best_cost = new_cost
                        best_sched = new_sched

                if best_sched is None:
                    continue

                ok, C_used = try_apply_change(
                    stop_id=sid,
                    stops=stops,
                    new_sched=best_sched,
                    p=p,
                    baseline_sched=baseline_sched,
                    changed=changed,
                    C_used=C_used,
                    C_max=C_max,
                )
                if ok:
                    improved_any = True
                    break  # rebuild vbc/core_by_cell and restart scanning

            if improved_any:
                break

        if not improved_any:
            break

    # compatibility returns (clusters/nucleus no longer meaningful here)
    clusters = {0: list(stops.keys())}
    nucleus = {0: next(iter(core_ids_all)) if core_ids_all else (clusters[0][0] if clusters[0] else -1)}

    return clusters, nucleus, p, changed, C_used
