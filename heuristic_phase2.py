from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from data_type import Stop, SchedTuple, DayBits, WEEKS, DAYS_5
from helper_funcs import a, compute_V, try_apply_change

def imbalance_between_weeks(V: Dict[Tuple[int, str], float]) -> float:
    week_totals = [sum(V[(l, d)] for d in DAYS_5) for l in WEEKS]
    return max(week_totals) - min(week_totals)

def imbalance_within_week(V: Dict[Tuple[int, str], float], l: int) -> float:
    vals = [V[(l, d)] for d in DAYS_5]
    return max(vals) - min(vals)

def imbalance_overall_cells(V: Dict[Tuple[int, str], float]) -> float:
    vals = V.values()
    return max(vals) - min(vals)

def metric_value(kind: str, V: Dict[Tuple[int, str], float]) -> float:
    if kind == "between_weeks":
        return imbalance_between_weeks(V)
    if kind == "within_week":
        return max(imbalance_within_week(V, l) for l in WEEKS)
    return imbalance_overall_cells(V)

def _daybits(s: SchedTuple) -> DayBits:
    return s[1]

def overlap_count(core: DayBits, cand: DayBits) -> int:
    # nucleus schedule과 후보 schedule이 겹치는(둘 다 1인) 요일 개수
    return sum(1 for k in range(7) if core[k] == 1 and cand[k] == 1)

def build_stop_to_cluster(clusters: Dict[int, List[int]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for cid, members in clusters.items():
        for sid in members:
            out[sid] = cid
    return out

def pick_high_low_days_in_week(V: Dict[Tuple[int, str], float], l: int) -> Tuple[str, str]:
    d_high = max(DAYS_5, key=lambda d: V[(l, d)])
    d_low = min(DAYS_5, key=lambda d: V[(l, d)])
    return d_high, d_low

def pick_week_high_low(V: Dict[Tuple[int, str], float]) -> Tuple[int, int]:
    week_total = {l: sum(V[(l, d)] for d in DAYS_5) for l in WEEKS}
    l_high = max(WEEKS, key=lambda l: week_total[l])
    l_low = min(WEEKS, key=lambda l: week_total[l])
    return l_high, l_low

def pick_high_low_cell(V: Dict[Tuple[int, str], float]) -> Tuple[Tuple[int, str], Tuple[int, str]]:
    high = max(V.keys(), key=lambda k: V[k])
    low = min(V.keys(), key=lambda k: V[k])
    return high, low


def _try_find_best_relocate(
    kind: str,
    V: Dict[Tuple[int, str], float],
    stops: Dict[int, Stop],
    p: Dict[int, SchedTuple],
    changed: Dict[int, int],
    C_used: int,
    C_max: int,
    baseline_sched: Dict[int, SchedTuple],
    sched_cache: Dict[Tuple[str, int], List[SchedTuple]],
    stop_to_cluster: Dict[int, int],
    nucleus: Dict[int, int],
    high_ids: List[int],
    # target cells
    from_cell: Tuple[int, str],
    to_cell: Tuple[int, str],
) -> Optional[Tuple[int, SchedTuple, float]]:
    """
    relocate-only: from_cell 방문을 끄고, to_cell 방문을 켜는 schedule change 찾기
    + nucleus overlap 악화는 금지
    + budget prune
    + best improvement 반환
    """
    before_val = metric_value(kind, V)

    l_from, d_from = from_cell
    l_to, d_to = to_cell

    best: Optional[Tuple[int, SchedTuple, float]] = None  # (stop_id, new_sched, after_metric)

    for stop_id in high_ids:
        old_sched = p[stop_id]
        vol_i = float(stops[stop_id].volume)
        cid = stop_to_cluster.get(stop_id)
        if cid is None:
            continue

        nuc_id = nucleus[cid]
        core_bits = _daybits(p[nuc_id])
        old_overlap = overlap_count(core_bits, _daybits(old_sched))

        # feasible options
        key = (stops[stop_id].dowcd, stops[stop_id].frequency)
        options = sched_cache.get(key, [])

        if old_sched not in options:
            options = options + [old_sched]

        for new_sched in options:
            if new_sched == old_sched:
                continue

            # relocate feasibility: from_cell off, to_cell on
            if a(old_sched, l_from, d_from) == 0:
                continue
            if a(new_sched, l_from, d_from) != 0:
                continue
            if a(new_sched, l_to, d_to) != 1:
                continue

            # nucleus 구조 보호
            new_overlap = overlap_count(core_bits, _daybits(new_sched))
            if new_overlap < old_overlap:
                continue

            # 변경 max 횟수 넘기지 않는 경우에만 진행
            before_c = changed[stop_id]
            after_c = 1 if new_sched != baseline_sched[stop_id] else 0
            if (C_used - before_c + after_c) > C_max:
                continue

            # virtual V update simulation
            V[(l_from, d_from)] -= vol_i
            V[(l_to, d_to)] += vol_i
            after_val = metric_value(kind, V)
            V[(l_from, d_from)] += vol_i
            V[(l_to, d_to)] -= vol_i

            if after_val < before_val and (best is None or after_val < best[2]):
                best = (stop_id, new_sched, after_val)

    return best


# Phase2 target selection per kind
def _targets_within_week(V: Dict[Tuple[int, str], float]) -> Tuple[Tuple[int, str], Tuple[int, str]]:
    # 가장 imbalance 큰 week 선택
    l0 = max(WEEKS, key=lambda l: imbalance_within_week(V, l))
    d_high, d_low = pick_high_low_days_in_week(V, l0)
    return (l0, d_high), (l0, d_low)

def _targets_between_weeks(V: Dict[Tuple[int, str], float]) -> Tuple[Tuple[int, str], Tuple[int, str]]:
    # week total 기준 high week / low week 선택
    l_high, l_low = pick_week_high_low(V)
    # high week에서 가장 큰 day, low week에서 가장 작은 day로 받기
    d_from = max(DAYS_5, key=lambda d: V[(l_high, d)])
    d_to = min(DAYS_5, key=lambda d: V[(l_low, d)])
    return (l_high, d_from), (l_low, d_to)

def _targets_overall_cells(V: Dict[Tuple[int, str], float]) -> Tuple[Tuple[int, str], Tuple[int, str]]:
    high, low = pick_high_low_cell(V)
    return high, low


def phase_2(
    artifacts: dict,
    p: Dict[int, SchedTuple],
    changed: Dict[int, int],
    C_used: int,
    clusters: Dict[int, List[int]],
    nucleus: Dict[int, int],
) -> Tuple[Dict[int, SchedTuple], Dict[int, int], int]:
    """
      1) between_weeks: 주차 총량 균형
      2) within_week: 주차 내 요일 균형
      3) overall_cells: 전체 (l,d) 셀 균형
    - nucleus overlap 악화 move 금지 (Phase1 구조 보호)
    """
    max_outer_iters = 200
    no_improve_patience = 5
    max_candidate_stops = 200

    stops: Dict[int, Stop] = artifacts["stops"]
    baseline_sched: Dict[int, SchedTuple] = artifacts["baseline_sched"]
    sched_cache: Dict[Tuple[str, int], List[SchedTuple]] = artifacts["sched_cache"]
    C_max: int = int(artifacts["C_max"])

    stop_to_cluster = build_stop_to_cluster(clusters)

    # init V
    V = compute_V(stops, p)

    # volume 큰 stop부터 후보로
    all_ids = list(stops.keys())
    all_ids.sort(key=lambda i: (-float(stops[i].volume), i))

    # kind별로 완전히 분리 실행
    for kind in ("between_weeks", "within_week", "overall_cells"):
        no_improve = 0

        for _ in range(max_outer_iters):
            before_val = metric_value(kind, V)

            # 타겟 셀 선택
            if kind == "between_weeks":
                from_cell, to_cell = _targets_between_weeks(V)
            elif kind == "within_week":
                from_cell, to_cell = _targets_within_week(V)
            else:
                from_cell, to_cell = _targets_overall_cells(V)

            l_from, d_from = from_cell

            # from_cell에 실제로 있는 stop 후보들 뽑기
            high_ids: List[int] = []
            for sid in all_ids[:max_candidate_stops]:
                if a(p[sid], l_from, d_from) == 1:
                    high_ids.append(sid)

            # relocate best 찾기
            best = _try_find_best_relocate(kind=kind, V=V, stops=stops, p=p, changed=changed, C_used=C_used,
                                           C_max=C_max, baseline_sched=baseline_sched, sched_cache=sched_cache,
                                           stop_to_cluster=stop_to_cluster, nucleus=nucleus, high_ids=high_ids,
                                           from_cell=from_cell, to_cell=to_cell)

            if best is None:
                no_improve += 1
                if no_improve >= no_improve_patience:
                    break
                continue

            # 적용
            stop_id, new_sched, _ = best
            ok, C_used = try_apply_change(
                stop_id=stop_id,
                new_sched=new_sched,
                p=p,
                baseline_sched=baseline_sched,
                changed=changed,
                C_used=C_used,
                C_max=C_max,
            )

            if ok:
                V = compute_V(stops, p)
                after_val = metric_value(kind, V)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= no_improve_patience:
                    break

    return p, changed, C_used
