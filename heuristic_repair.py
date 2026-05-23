from __future__ import annotations

from typing import Dict, List, Set, Optional
import random

from data_type import Stop, SchedTuple
from baseline_algorithm_result_test import compute_rectangle_overlap_nodes_term, compute_nho_metric
from optimal_utils import try_apply_change
from heuristic_utils import _capacity_feasible, _change_used, _changed_map

def _feasible_candidates(
    stops: Dict[int, Stop],
    p_work: Dict[int, SchedTuple],
    changed_work: Dict[int, int],
    c_used: int,
    sid: int,
    pi: Dict[int, List[SchedTuple]],
    baseline_sched: Dict[int, SchedTuple],
    timecycle: int,
    v_max: float,
    g_max: float,
    c_max: int,
) -> List[SchedTuple]:
    out = []

    for cand in pi[sid]:
        tmp_p = dict(p_work)
        tmp_changed = dict(changed_work)
        tmp_c_used = c_used

        ok, tmp_c_used = try_apply_change(
            stop_id=sid,
            stops=stops,
            new_sched=cand,
            p=tmp_p,
            baseline_sched=baseline_sched,
            changed=tmp_changed,
            C_used=tmp_c_used,
            C_max=c_max,
        )

        if not ok:
            continue

        if not _capacity_feasible(
            stops=stops,
            p=tmp_p,
            timecycle=timecycle,
            v_max=v_max,
            g_max=g_max,
        ):
            continue

        out.append(cand)

    return out



def _repair_greedy(
    artifacts: dict,
    stops: Dict[int, Stop],
    partial_p: Dict[int, SchedTuple],
    removed: Set[int],
    pi: Dict[int, List[SchedTuple]],
    baseline_sched: Dict[int, SchedTuple],
    timecycle: int,
    v_max: float,
    g_max: float,
    c_max: int,
) -> Optional[Dict[int, SchedTuple]]:
    p_work = dict(partial_p)
    changed_work = _changed_map(p_work, baseline_sched)
    c_used = _change_used(p_work, baseline_sched)

    remaining = list(removed)

    while remaining:
        best = None

        for sid in remaining:
            candidates = _feasible_candidates(
                stops=stops,
                p_work=p_work,
                changed_work=changed_work,
                c_used=c_used,
                sid=sid,
                pi=pi,
                baseline_sched=baseline_sched,
                timecycle=timecycle,
                v_max=v_max,
                g_max=g_max,
                c_max=c_max,
            )

            for cand in candidates:
                tmp_p = dict(p_work)
                tmp_changed = dict(changed_work)
                tmp_c_used = c_used

                ok, tmp_c_used = try_apply_change(
                    stop_id=sid,
                    stops=stops,
                    new_sched=cand,
                    p=tmp_p,
                    baseline_sched=baseline_sched,
                    changed=tmp_changed,
                    C_used=tmp_c_used,
                    C_max=c_max,
                )

                if not ok:
                    continue

                obj = compute_rectangle_overlap_nodes_term(artifacts, tmp_p)

                if best is None or obj < best[0]:
                    best = (obj, sid, cand)

        if best is None:
            return None

        _, sid, cand = best

        ok, c_used = try_apply_change(
            stop_id=sid,
            stops=stops,
            new_sched=cand,
            p=p_work,
            baseline_sched=baseline_sched,
            changed=changed_work,
            C_used=c_used,
            C_max=c_max,
        )

        if not ok:
            return None

        remaining.remove(sid)

    return p_work


def _repair_random(
    stops: Dict[int, Stop],
    partial_p: Dict[int, SchedTuple],
    removed: Set[int],
    pi: Dict[int, List[SchedTuple]],
    baseline_sched: Dict[int, SchedTuple],
    timecycle: int,
    v_max: float,
    g_max: float,
    c_max: int,
    rng: random.Random,
) -> Optional[Dict[int, SchedTuple]]:
    p_work = dict(partial_p)
    changed_work = _changed_map(p_work, baseline_sched)
    c_used = _change_used(p_work, baseline_sched)

    remaining = list(removed)
    rng.shuffle(remaining)

    for sid in remaining:
        candidates = _feasible_candidates(
            stops=stops,
            p_work=p_work,
            changed_work=changed_work,
            c_used=c_used,
            sid=sid,
            pi=pi,
            baseline_sched=baseline_sched,
            timecycle=timecycle,
            v_max=v_max,
            g_max=g_max,
            c_max=c_max,
        )

        if not candidates:
            return None

        cand = rng.choice(candidates)

        ok, c_used = try_apply_change(
            stop_id=sid,
            stops=stops,
            new_sched=cand,
            p=p_work,
            baseline_sched=baseline_sched,
            changed=changed_work,
            C_used=c_used,
            C_max=c_max,
        )

        if not ok:
            return None

    return p_work

def _repair_greedy_nho(
    artifacts: dict,
    stops: Dict[int, Stop],
    partial_p: Dict[int, SchedTuple],
    removed: Set[int],
    pi: Dict[int, List[SchedTuple]],
    baseline_sched: Dict[int, SchedTuple],
    timecycle: int,
    v_max: float,
    g_max: float,
    c_max: int,
) -> Optional[Dict[int, SchedTuple]]:
    p_work = dict(partial_p)
    changed_work = _changed_map(p_work, baseline_sched)
    c_used = _change_used(p_work, baseline_sched)

    remaining = list(removed)

    while remaining:
        best = None

        for sid in remaining:
            candidates = _feasible_candidates(
                stops=stops,
                p_work=p_work,
                changed_work=changed_work,
                c_used=c_used,
                sid=sid,
                pi=pi,
                baseline_sched=baseline_sched,
                timecycle=timecycle,
                v_max=v_max,
                g_max=g_max,
                c_max=c_max,
            )

            for cand in candidates:
                tmp_p = dict(p_work)
                tmp_changed = dict(changed_work)
                tmp_c_used = c_used

                ok, tmp_c_used = try_apply_change(
                    stop_id=sid,
                    stops=stops,
                    new_sched=cand,
                    p=tmp_p,
                    baseline_sched=baseline_sched,
                    changed=tmp_changed,
                    C_used=tmp_c_used,
                    C_max=c_max,
                )

                if not ok:
                    continue

                nho = compute_nho_metric(
                    artifacts,
                    tmp_p,
                )

                if best is None or nho < best[0]:
                    best = (nho, sid, cand)

        if best is None:
            return None

        _, sid, cand = best

        ok, c_used = try_apply_change(
            stop_id=sid,
            stops=stops,
            new_sched=cand,
            p=p_work,
            baseline_sched=baseline_sched,
            changed=changed_work,
            C_used=c_used,
            C_max=c_max,
        )

        if not ok:
            return None

        remaining.remove(sid)

    return p_work