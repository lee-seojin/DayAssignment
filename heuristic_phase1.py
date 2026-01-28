from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from data_type import Stop, DayBits, SchedTuple, WeekTuple, ScheduleView
from helper_funcs import get_dist, try_apply_change

def build_priority_groups(
    stops: Dict[int, Stop],
    priority_map: Dict[Tuple[str, int], int],
) -> Dict[int, List[int]]:
    # return: priority_groups[p] = [stop_id1, stop_id2, ...] - stop_id list는 (freq, stop_id) 기준으로 정렬됨

    groups = defaultdict(list)
    for stop_id, s in stops.items():
        key = (s.dowcd, s.frequency)
        pr = priority_map.get(key)
        if pr is None:
            continue
        groups[pr].append(stop_id)

    # 각 priority 그룹 내부 정렬 - 매번 같은 리스트가 생성되도록
    for pr, ids in groups.items():
        ids.sort(key=lambda i: (stops[i].frequency, i))

    return dict(groups)


def dbscan(
    core_ids: List[int],
    stops: Dict[int, Stop],
    Ddist: Dict[Tuple[int, int], float],
    eps: float,
    min_pts: int,
) -> Tuple[Dict[int, List[int]], Set[int]]:

    n = len(core_ids)
    if n == 1:
        return {0: [core_ids[0]]}, set()

    # precomputed distance matrix
    dist = np.zeros((n, n), dtype=float)
    for a in range(n):
        i = core_ids[a]
        for b in range(a+1, n):
            if a == b:
                continue
            j = core_ids[b]
            dist[a, b] = get_dist(i, j, Ddist, stops)
            dist[b, a] = get_dist(j, i, Ddist, stops)

    model = DBSCAN(eps=eps, min_samples=min_pts, metric="precomputed")
    labels = model.fit_predict(dist)

    clusters: Dict[int, List[int]] = {}
    noise: Set[int] = set()

    # sklearn label: -1 = noise, else cluster id
    for idx, label in enumerate(labels):
        stop_id = core_ids[idx]
        if label == -1:
            noise.add(stop_id)
        else:
            clusters.setdefault(int(label), []).append(stop_id)

    return clusters, noise


def choose_nucleus(
    members: List[int],
    stops: Dict[int, Stop],
    Ddist: Dict[Tuple[int, int], float],
) -> int:
    # nucleus = medoid (클러스터 내부 거리합 최소 stop)
    if len(members) == 1:
        return members[0]

    best = members[0]
    best_sum = float("inf")

    for i in members:
        s = 0.0
        for j in members:
            if i == j:
                continue
            s += get_dist(i, j, Ddist, stops)
        if s < best_sum:
            best_sum = s
            best = i

    return best

def nearest_k_nuclei(
    stop_id: int,
    nucleus: Dict[int, int],  # cid -> nucleus stop_id
    stops: Dict[int, Stop],
    Ddist: Dict[Tuple[int, int], float],
    k: int,
) -> List[int]:

    dlist = []
    for cid, nuc_id in nucleus.items():
        d = get_dist(stop_id, nuc_id, Ddist, stops)
        dlist.append((d, cid))

    dlist.sort(key=lambda x: x[0])
    return [cid for _, cid in dlist[:k]]

def overlap_score(core: DayBits, cand: DayBits, alpha: float = 0.2) -> float:
    overlap = 0
    extra = 0

    for k in range(7):
        if core[k] == 1 and cand[k] == 1:
            overlap += 1
        elif core[k] == 0 and cand[k] == 1:
            extra += 1

    return overlap - alpha * extra

def choose_best_sched(
    candidates: List[SchedTuple],
    core_week: WeekTuple,
    core_day: DayBits,
    alpha: float,
) -> Optional[SchedTuple]:
    if not candidates:
        return None

    # day 겹침 최우선
    scored = []
    best_day = -1e18
    for (w, d) in candidates:
        day_score = overlap_score(core_day, d, alpha=alpha)
        if day_score > best_day:
            best_day = day_score
        scored.append((w, d, day_score))

    # day_score 최대인 것만 추림, 1e-9 는 bit 연산에서 유래되는 오차를 수용하기 위한 기준으로, 사실상 ds = best_day인 것만 골라내려는 것
    top = [(w, d, ds) for (w, d, ds) in scored if abs(ds - best_day) <= 1e-9]

    # 주차 겹침 점수가 최대인 것 최종 선택
    core_set = set(core_week)
    best, best_score = None, -1e18

    for (w, d, ds) in top:
        week_overlap = len(set(w) & core_set)
        score = week_overlap
        if score > best_score:
            best_score = score
            best = (w, d)

    return best


def phase_1(
    artifacts: dict,
    p: Dict[int, SchedTuple],
    changed: Dict[int, int],
    C_used: int
) -> Tuple[Dict[int, List[int]], Dict[int, int], Dict[int, SchedTuple], Dict[int, int], int]:

    # artifacts에서 참조 데이터 로드
    stops: Dict[int, Stop] = artifacts["stops"]
    Ddist: Dict[Tuple[int, int], float] = artifacts["Ddist"]
    schedrules_map: Dict[Tuple[str, int], List[SchedTuple]] = artifacts["sched_cache"]
    priority_map: Dict[Tuple[str, int], int] = artifacts["priority_map"]
    baseline_sched: Dict[int, SchedTuple] = artifacts["baseline_sched"]
    C_max: int = int(artifacts["C_max"])

    eps: float = 2000.0
    min_pts: int = 3
    k_nearest: int = 3
    alpha: float = 0.2

    # priority 그룹 만들기
    priority_groups = build_priority_groups(stops, priority_map)
    if not priority_groups:
        # priority_map에 매칭되는 stop이 아예 없는 경우 방어
        clusters = {0: list(stops.keys())}
        nucleus = {0: choose_nucleus(clusters[0], stops, Ddist)}
        return clusters, nucleus, p, changed, C_used

    best_pr = min(priority_groups.keys())
    core_ids = priority_groups[best_pr]

    # 1) core_ids로 DBSCAN
    clusters, noise = dbscan(core_ids, stops, Ddist, eps=eps, min_pts=min_pts)

    # noise core -> singleton cluster 승격
    next_cid = max(clusters.keys(), default=-1) + 1
    for stop_id in noise:
        clusters[next_cid] = [stop_id]
        next_cid += 1

    # 2) nucleus 계산
    nucleus: Dict[int, int] = {}
    for cid, members in clusters.items():
        nucleus[cid] = choose_nucleus(members, stops, Ddist)

    # 3) 나머지 stop들을 priority 순서대로 편입
    for pr in sorted(priority_groups.keys()):
        if pr == best_pr:
            continue

        for stop_id in priority_groups[pr]:
            current_stop = stops[stop_id]
            candidates = schedrules_map.get((current_stop.dowcd, current_stop.frequency), []).copy()

            base_w, base_d = baseline_sched[stop_id]
            if current_stop.dowlockcd:
                candidates = [sched for sched in candidates if sched[1] == base_d]
            if current_stop.wccd_flag:
                candidates = [sched for sched in candidates if sched[0] == base_w]

            # 후보 클러스터 (가까운 nucleus 기준)
            cand_cids = nearest_k_nuclei(stop_id, nucleus, stops, Ddist, k=k_nearest)

            # nucleus가 없거나 후보가 없으면 singleton cluster 생성
            if not cand_cids:
                cid = next_cid
                next_cid += 1
                clusters[cid] = [stop_id]
                nucleus[cid] = stop_id
                continue

            # (1) 현재 stop이 할당될 수 있는 후보 클러스터별 평가를 1번만 계산해서 저장
            fit_sched: Dict[int, SchedTuple] = {}    # cid -> (core_week, best_day)
            need_change: Dict[int, bool] = {}        # cid -> p[stop_id] 변경 필요 여부
            dist_to: Dict[int, float] = {}           # cid -> dist(stop_id, nucleus[cid])

            for cid in cand_cids:
                nuc_id = nucleus[cid]
                dist_to[cid] = get_dist(stop_id, nuc_id, Ddist, stops)
                core_week, core_day = p[nuc_id]

                best_sched = choose_best_sched(candidates, core_week, core_day, alpha=alpha)
                if best_sched is None:
                    continue
                fit_sched[cid] = best_sched
                need_change[cid] = (p[stop_id] != best_sched)

            usable_cids = [cid for cid in cand_cids if cid in fit_sched]
            # cid 후보와 유의미하게 겹쳐질 수 있는 schedule이 feasible schedule 중에 없을 때 (현재 코드 구조에서는 거의 발생 x) 그냥 아무거나 반환
            if not usable_cids:
                chosen_cid = cand_cids[0]
                clusters[chosen_cid].append(stop_id)
                continue

            # (2) 변경 없이 가능하면: 그 중 가장 가까운 클러스터로
            no_change_cids = [cid for cid in usable_cids if not need_change[cid]]
            if no_change_cids:
                chosen_cid = min(no_change_cids, key=lambda cid: dist_to[cid])
                clusters[chosen_cid].append(stop_id)
                continue

            # (3) 전부 변경 필요하면: budget 되면 가장 가까운 곳으로 변경 후 편입
            if C_used + 1 <= C_max:
                chosen_cid = min(usable_cids, key=lambda cid: dist_to[cid])
                new_sched = fit_sched[chosen_cid]

                # C_used/C_max 반영해서 상태 dict만 업데이트
                ok, C_used = try_apply_change(
                    stop_id=stop_id,
                    stops=stops,
                    new_sched=new_sched,
                    p=p,
                    baseline_sched=baseline_sched,
                    changed=changed,
                    C_used=C_used,
                    C_max=C_max,
                )
                if ok:
                    clusters[chosen_cid].append(stop_id)
                    continue

            # budget 불가(or 적용 실패): baseline 유지 + 가장 가까운 곳으로 편입
            chosen_cid = min(usable_cids, key=lambda cid: dist_to[cid])
            clusters[chosen_cid].append(stop_id)

    # Phase 1 끝: nucleus 재계산(한 번만)
    for cid, members in clusters.items():
        nucleus[cid] = choose_nucleus(members, stops, Ddist)

    return clusters, nucleus, p, changed, C_used