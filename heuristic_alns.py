from __future__ import annotations

from typing import Dict, List
import random
import time

from data_type import Stop, SchedTuple
from baseline_algorithm_result_test import compute_rectangle_overlap_nodes_term, compute_nho_metric
from heuristic_utils import _changed_map
from heuristic_destroy import _destroy_random, _destroy_worst_pair, _destroy_overlap_driven, _destroy_worst_nho_pair
from heuristic_repair import _repair_random, _repair_greedy, _repair_greedy_nho

def select_operator(
    weights: Dict[str, float],
    rng: random.Random,
) -> str:
    names = list(weights.keys())
    total = sum(weights.values())

    r = rng.random() * total
    acc = 0.0

    for name in names:
        acc += weights[name]
        if acc >= r:
            return name

    return names[-1]


def update_weight(
    weights: Dict[str, float],
    name: str,
    score: float,
    reaction: float = 0.2,
    min_weight: float = 0.1,
) -> None:
    weights[name] = max(
        min_weight,
        (1.0 - reaction) * weights[name] + reaction * score,
    )


def alns_improve(
    stops: Dict[int, Stop],
    pi: Dict[int, List[SchedTuple]],
    baseline_sched: Dict[int, SchedTuple],
    p_initial: Dict[int, SchedTuple],
    timecycle: int,
    v_max: float,
    g_max: float,
    c_max: int,
    max_iters: int = 300,
    patience: int = 50,
    remove_ratio: float = 0.05,
    seed: int = 42,
):
    rng = random.Random(seed)

    # ------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------
    n_generated = 0
    n_failed_repair = 0
    n_accepted = 0
    n_best_update = 0

    # Initial solution
    p_cur = dict(p_initial)
    p_best = dict(p_initial)

    q = max(1, int(len(stops) * remove_ratio))

    artifacts_for_eval = {
        "stops": stops,
        "timecycle": timecycle,
    }

    # Current / best metrics
    cur_overlap = compute_rectangle_overlap_nodes_term(artifacts_for_eval, p_cur)
    cur_nho = compute_nho_metric(artifacts_for_eval, p_cur)

    best_overlap = cur_overlap
    best_nho = cur_nho
    best_obj = best_overlap

    initial_changed = _changed_map(p_initial, baseline_sched)

    # Operator weights
    destroy_weights = {
        # "overlap_driven": 1.0,
        # "worst_pair": 1.0,
        "worst_nho_pair": 1.0,
        # "random": 1.0,
    }

    repair_weights = {
        # "greedy": 1.0,
        #"greedy_nho": 1.0,
        "random": 1.0,
    }

    no_improve = 0

    print()
    print("[ALNS]")
    print(f"Initial #OVERLAP  = {cur_overlap:.0f}")
    print(f"Initial NHO       = {cur_nho:.6f}")
    print(f"Initial changed   = {sum(initial_changed.values())} / {c_max}")
    print(f"Destroy size      = {q}")

    t0 = time.perf_counter()

    # ALNS loop
    for it in range(1, max_iters + 1):
        destroy_name = select_operator(destroy_weights, rng)
        repair_name = select_operator(repair_weights, rng)

        # Destroy
        if destroy_name == "overlap_driven":
            removed = _destroy_overlap_driven(
                stops=stops,
                p=p_cur,
                timecycle=timecycle,
                q=q,
            )

        elif destroy_name == "worst_pair":
            removed = _destroy_worst_pair(
                stops=stops,
                p=p_cur,
                timecycle=timecycle,
                q=q,
                rng=rng,
            )

        elif destroy_name == "worst_nho_pair":
            removed = _destroy_worst_nho_pair(
                stops=stops,
                p=p_cur,
                timecycle=timecycle,
                q=q,
                rng=rng,
            )

        else:
            removed = _destroy_random(
                p=p_cur,
                q=q,
                rng=rng,
            )

        partial_p = {
            i: sched
            for i, sched in p_cur.items()
            if i not in removed
        }

        # Repair
        if repair_name == "greedy":
            p_new = _repair_greedy(
                artifacts=artifacts_for_eval,
                stops=stops,
                partial_p=partial_p,
                removed=removed,
                pi=pi,
                baseline_sched=baseline_sched,
                timecycle=timecycle,
                v_max=v_max,
                g_max=g_max,
                c_max=c_max,
            )

        elif repair_name == "greedy_nho":
            p_new = _repair_greedy_nho(
                artifacts=artifacts_for_eval,
                stops=stops,
                partial_p=partial_p,
                removed=removed,
                pi=pi,
                baseline_sched=baseline_sched,
                timecycle=timecycle,
                v_max=v_max,
                g_max=g_max,
                c_max=c_max,
            )

        else:
            p_new = _repair_random(
                stops=stops,
                partial_p=partial_p,
                removed=removed,
                pi=pi,
                baseline_sched=baseline_sched,
                timecycle=timecycle,
                v_max=v_max,
                g_max=g_max,
                c_max=c_max,
                rng=rng,
            )

        # Repair failed
        if p_new is None:
            n_failed_repair += 1
            no_improve += 1

            update_weight(destroy_weights, destroy_name, 0.5)
            update_weight(repair_weights, repair_name, 0.5)

            if no_improve >= patience:
                print(f"Stop by patience at iter={it}")
                break

            continue

        n_generated += 1

        # Evaluate new solution
        new_overlap = compute_rectangle_overlap_nodes_term(artifacts_for_eval, p_new)

        new_nho = compute_nho_metric(
            artifacts_for_eval,
            p_new,
        )

        overlap_tolerance = 0.02

        # If this iteration used an NHO-oriented operator,
        # accept/evaluate primarily by NHO.
        use_nho_acceptance = (
            destroy_name == "worst_nho_pair"
            or repair_name == "greedy_nho"
        )

        if use_nho_acceptance:
            # NHO-first:
            # 1) lower NHO is better
            # 2) if NHO ties, lower overlap is better
            is_best = (new_nho < best_nho)

            is_current = (new_nho < cur_nho)

        else:
            # Overlap-first:
            # 1) lower #NODE_OVERLAP is better
            # 2) if overlap ties, lower NHO is better
            is_best = (new_overlap < best_overlap)

            is_current = (new_overlap < cur_overlap)

        # Best update
        if is_best:
            n_best_update += 1
            n_accepted += 1

            p_best = dict(p_new)
            p_cur = dict(p_new)

            best_overlap = new_overlap
            best_nho = new_nho
            best_obj = best_overlap

            cur_overlap = new_overlap
            cur_nho = new_nho

            no_improve = 0

            update_weight(destroy_weights, destroy_name, 5.0)
            update_weight(repair_weights, repair_name, 5.0)

            print(
                f"\niter={it:4d}  "
                f"new best  "
                f"#OVERLAP={best_overlap:.0f}  "
                f"NHO={best_nho:.6f}  "
                f"acceptance={'NHO' if use_nho_acceptance else 'OVERLAP'}  "
                f"destroy={destroy_name}  "
                f"repair={repair_name}"
            )

        # Current update only
        elif is_current:
            n_accepted += 1

            p_cur = dict(p_new)

            cur_overlap = new_overlap
            cur_nho = new_nho

            no_improve += 1

            update_weight(destroy_weights, destroy_name, 2.0)
            update_weight(repair_weights, repair_name, 2.0)

        # Reject
        else:
            no_improve += 1

            update_weight(destroy_weights, destroy_name, 1.0)
            update_weight(repair_weights, repair_name, 1.0)

        if no_improve >= patience:
            print(f"Stop by patience at iter={it}")
            break

    t1 = time.perf_counter()

    # Final summary
    changed = _changed_map(p_best, baseline_sched)

    n_diff_from_initial = sum(
        int(p_best[i] != p_initial[i])
        for i in p_initial.keys()
    )

    print()
    print(f"Best #OVERLAP     = {best_overlap:.0f}")
    print(f"Best NHO          = {best_nho:.6f}")
    print(f"Changed stops     = {sum(changed.values())} / {c_max}")
    print(f"Diff from initial = {n_diff_from_initial}")

    print(f"Generated moves   = {n_generated}")
    print(f"Failed repairs    = {n_failed_repair}")
    print(f"Accepted moves    = {n_accepted}")
    print(f"Best updates      = {n_best_update}")

    print(f"\nRun time={t1 - t0:.3f}s")

    return p_best, changed, best_obj