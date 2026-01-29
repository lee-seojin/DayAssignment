from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from data_type import WEEKS, DAYS_5

from data_type import Stop, SchedTuple, WEEKS, DAYS_5, Cell
from helper_funcs import a, compute_V, try_apply_change

def _cells_toggled(old_sched: SchedTuple, new_sched: SchedTuple) -> List[Tuple[int, str, int]]:
    """
    old->new로 바꿀 때 방문 여부가 달라지는 셀들만 반환.
    각 원소: (l, d, delta_visit) where delta_visit ∈ {-1, +1}
    """
    out: List[Tuple[int, str, int]] = []
    for l in WEEKS:
        for d in DAYS_5:
            old = a(old_sched, l, d)
            new = a(new_sched, l, d)
            if old != new:
                out.append((l, d, new - old))
    return out


def _apply_delta_to_V(
    V: Dict[Cell, float],
    delta_cells: List[Tuple[int, str, int]],
    vol: float,
    sign: int,
) -> None:
    """
    sign=+1이면 delta 적용, sign=-1이면 rollback.
    """
    for l, d, dv in delta_cells:
        V[(l, d)] += sign * dv * vol

def _range_metric(V: Dict[Cell, float]) -> float:
    # range와 동시에 0 셀 개수 줄이기
    vals = list(V.values())
    mu = sum(vals) / len(vals)
    return sum(abs(v - mu) for v in vals)

def _argmax_cell(V: Dict[Cell, float]) -> Cell:
    return max(V.keys(), key=lambda k: V[k])


def _argmin_cell(V: Dict[Cell, float]) -> Cell:
    return min(V.keys(), key=lambda k: V[k])


def _filter_options_with_locks(
    stop: Stop,
    options: List[SchedTuple],
    baseline_sched: Dict[int, SchedTuple],
    stop_id: int,
) -> List[SchedTuple]:
    """
    baseline 기준 lock 반영
    - dowlockcd==1: daybits 고정 (baseline daybits만 허용)
    - wccd_flag==1: weektuple 고정 (baseline weektuple만 허용)
    """
    base_w, base_d = baseline_sched[stop_id]

    out = options
    if stop.dowlockcd == 1:
        out = [s for s in out if s[1] == base_d]
    if stop.wccd_flag == 1:
        out = [s for s in out if s[0] == base_w]
    return out


def save_calendar_heatmap(
    V: Dict[Cell, float],
    tag: str,
    out_dir: str = "phase2_heatmaps",
) -> None:
    """
    4주 x 5일 달력형 heatmap 저장
    """
    os.makedirs(out_dir, exist_ok=True)

    grid = [[V[(l, d)] for d in DAYS_5] for l in WEEKS]

    fig, ax = plt.subplots()
    im = ax.imshow(grid)

    ax.set_xticks(range(len(DAYS_5)))
    ax.set_xticklabels(DAYS_5)
    ax.set_yticks(range(len(WEEKS)))
    ax.set_yticklabels([f"W{l}" for l in WEEKS])

    for i, l in enumerate(WEEKS):
        for j, d in enumerate(DAYS_5):
            ax.text(j, i, f"{grid[i][j]:.0f}", ha="center", va="center", fontsize=8)

    ax.set_title(tag)
    fig.colorbar(im, ax=ax)

    fname = f"{tag}.png".replace(" ", "_")
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)


def phase_2(
    artifacts: dict,
    p: Dict[int, SchedTuple],
    changed: Dict[int, int],
    C_used: int,
    clusters: Dict[int, List[int]],   # 안 쓰지만 시그니처 유지
    nucleus: Dict[int, int],
) -> Tuple[Dict[int, SchedTuple], Dict[int, int], int]:
    """
    max(V)-min(V) (range)만 직접 줄이는 greedy relocate.
    - 매 iteration마다 현재 V를 보고 max cell / min cell 갭을 줄이는 move를 찾음
    - to_cell을 고정하지 않음: stop이 갈 수 있는 feasible schedule 중 range를 가장 줄이는 걸 선택
    - lock/dowcd/freq/sched_cache 기반 feasible schedule만 사용
    """

    stops: Dict[int, Stop] = artifacts["stops"]
    baseline_sched: Dict[int, SchedTuple] = artifacts["baseline_sched"]
    sched_cache: Dict[Tuple[str, int], List[SchedTuple]] = artifacts["sched_cache"]
    C_max: int = int(artifacts["C_max"])

    # 튜닝 파라미터 (원하는 만큼 올릴 수 있음)
    max_iters = 2000
    max_stop_candidates = 400         # max cell에 있는 stop들 중 상위 N개만 보자 (너무 크면 느려짐)

    # 초기 V
    V = compute_V(stops, p)
    save_calendar_heatmap(V, tag="phase2_start")
    cur_metric = _range_metric(V)

    # max cell에 있는 stop들 추출
    def stops_in_cell(cell: Cell) -> List[int]:
        l, d = cell
        return [sid for sid in stops.keys() if a(p[sid], l, d) == 1]

    # stop 우선순위: 자유도(후보 많음) + 효과(볼륨 큼)로 정렬
    def stop_priority_key(sid: int, opt_count: int) -> Tuple[int, float, int]:
        # opt_count 클수록 우선, volume 클수록 우선
        return (-opt_count, -float(stops[sid].volume), sid)

    no_improve = 0
    patience = 200  # 이만큼 연속으로 개선 없으면 종료

    for it in range(max_iters):
        max_cell = _argmax_cell(V)

        # max cell에 있는 stop 후보들
        cand_ids = stops_in_cell(max_cell)

        # 각 stop의 feasible option 개수를 빠르게 계산해서 우선순위 정렬
        scored_ids: List[Tuple[int, int]] = []
        for sid in cand_ids:
            s = stops[sid]
            key = (s.dowcd, s.frequency)
            options = sched_cache.get(key, [])
            # old_sched 포함
            old_sched = p[sid]
            if old_sched not in options:
                options = options + [old_sched]
            options = _filter_options_with_locks(s, options, baseline_sched, sid)
            scored_ids.append((sid, len(options)))

        scored_ids.sort(key=lambda x: stop_priority_key(x[0], x[1]))
        cand_ids = [sid for sid, _ in scored_ids[:max_stop_candidates]]

        best_move: Optional[Tuple[int, SchedTuple, float]] = None  # (sid, new_sched, new_metric)

        # move 탐색
        for sid in cand_ids:
            s = stops[sid]
            vol = float(s.volume)
            old_sched = p[sid]

            # feasible schedules
            key = (s.dowcd, s.frequency)
            options = sched_cache.get(key, [])
            if old_sched not in options:
                options = options + [old_sched]
            options = _filter_options_with_locks(s, options, baseline_sched, sid)

            # delta를 미리 계산해두면 빠름
            for new_sched in options:
                if new_sched == old_sched:
                    continue

                # budget check
                before_c = changed.get(sid, 0)
                after_c = 1 if new_sched != baseline_sched[sid] else 0
                if (C_used - before_c + after_c) > C_max:
                    continue

                # 실제로 max_cell에서 빠지지 않으면 의미 없음(최소 조건)
                lmax, dmax = max_cell
                if a(new_sched, lmax, dmax) != 0:
                    continue

                delta_cells = _cells_toggled(old_sched, new_sched)
                if not delta_cells:
                    continue

                # V에 임시 적용 → metric 평가 → rollback
                _apply_delta_to_V(V, delta_cells, vol, sign=+1)
                new_metric = _range_metric(V)
                _apply_delta_to_V(V, delta_cells, vol, sign=-1)

                if new_metric < cur_metric:
                    if best_move is None or new_metric < best_move[2]:
                        best_move = (sid, new_sched, new_metric)

        # 개선 move 없으면 종료 또는 계속
        if best_move is None:
            no_improve += 1
            if no_improve >= patience:
                break
            continue

        # best move 적용
        sid, new_sched, new_metric = best_move
        ok, C_used2 = try_apply_change(
            stop_id=sid,
            stops=stops,
            new_sched=new_sched,
            p=p,
            baseline_sched=baseline_sched,
            changed=changed,
            C_used=C_used,
            C_max=C_max,
        )
        if not ok:
            # 이론상 여기까지 오면 거의 ok여야 하는데, 그래도 안전하게 처리
            no_improve += 1
            if no_improve >= patience:
                break
            continue

        C_used = C_used2
        V = compute_V(stops, p)
        cur_metric = _range_metric(V)
        no_improve = 0

        save_calendar_heatmap(V, tag=f"iter_{it}_metric_{cur_metric:.0f}")

    save_calendar_heatmap(V, tag="phase2_final")

    return p, changed, C_used
