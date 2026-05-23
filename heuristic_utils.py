from __future__ import annotations

from typing import Dict, List

from data_type import Stop, SchedTuple, DAYS_5
from optimal_utils import a, _get_visited_by_cell

def _weeks(timecycle: int) -> List[int]:
    return list(range(1, int(timecycle) + 1))


def _capacity_feasible(
    stops: Dict[int, Stop],
    p: Dict[int, SchedTuple],
    timecycle: int,
    v_max: float,
    g_max: float,
) -> bool:
    for l in _weeks(timecycle):
        for d in DAYS_5:
            vol = 0.0
            weight = 0.0

            for i, sched in p.items():
                if a(sched, l, d) == 1:
                    vol += float(stops[i].volume)
                    weight += float(stops[i].weight)

            if vol > v_max + 1e-9:
                return False
            if weight > g_max + 1e-9:
                return False

    return True


def _changed_map(
    p: Dict[int, SchedTuple],
    baseline_sched: Dict[int, SchedTuple],
) -> Dict[int, int]:
    return {
        i: int(i in p and p[i] != baseline_sched[i])
        for i in baseline_sched.keys()
    }


def _change_used(
    p: Dict[int, SchedTuple],
    baseline_sched: Dict[int, SchedTuple],
) -> int:
    return sum(
        int(p[i] != baseline_sched[i])
        for i in p.keys()
    )


### only for rectangle method ###
def _rectangles(
    stops: Dict[int, Stop],
    p: Dict[int, SchedTuple],
    timecycle: int,
):
    visited = _get_visited_by_cell( {"timecycle": timecycle},p)
    rect = {}

    for l in _weeks(timecycle):
        for d in DAYS_5:
            ids = visited[(l, d)]

            if not ids:
                rect[(l, d)] = None
                continue

            xs = [float(stops[i].xcoord) for i in ids]
            ys = [float(stops[i].ycoord) for i in ids]

            rect[(l, d)] = (
                min(xs),
                max(xs),
                min(ys),
                max(ys),
            )

    return rect, visited


def _inside_rect(stop: Stop, rect) -> bool:
    x1, x2, y1, y2 = rect
    x = float(stop.xcoord)
    y = float(stop.ycoord)

    return x1 <= x <= x2 and y1 <= y <= y2

