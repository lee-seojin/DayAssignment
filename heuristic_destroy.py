from __future__ import annotations

from typing import Dict, List, Set
import random
import itertools
from shapely.geometry import MultiPoint, Point

from data_type import Stop, SchedTuple, DAYS_5
from heuristic_utils import _weeks, _inside_rect, _rectangles
from optimal_utils import _get_visited_by_cell

def _stop_overlap_contribution(
    stops: Dict[int, Stop],
    p: Dict[int, SchedTuple],
    timecycle: int,
) -> Dict[int, float]:
    rect, visited = _rectangles(stops, p, timecycle)

    contrib = {
        i: 0.0
        for i in p.keys()
    }

    for l in _weeks(timecycle):
        for d in DAYS_5:
            for e in DAYS_5:
                if d == e:
                    continue

                re = rect[(l, e)]
                if re is None:
                    continue

                for i in visited[(l, d)]:
                    if _inside_rect(stops[i], re):
                        contrib[i] += 1.0

    return contrib


def _worst_pair_nodes(
    stops: Dict[int, Stop],
    p: Dict[int, SchedTuple],
    timecycle: int,
) -> List[int]:
    rect, visited = _rectangles(stops, p, timecycle)

    best_nodes = []

    for l in _weeks(timecycle):
        for d in DAYS_5:
            for e in DAYS_5:
                if d == e:
                    continue

                re = rect[(l, e)]
                if re is None:
                    continue

                nodes = [
                    i for i in visited[(l, d)]
                    if _inside_rect(stops[i], re)
                ]

                if len(nodes) > len(best_nodes):
                    best_nodes = nodes

    return best_nodes

def _worst_nho_pair_nodes(
    stops: Dict[int, Stop],
    p: Dict[int, SchedTuple],
    timecycle: int,
) -> List[int]:

    artifacts_for_eval = {
        "stops": stops,
        "timecycle": timecycle,
    }

    visited = _get_visited_by_cell(artifacts_for_eval, p)
    day_pairs = list(itertools.combinations(DAYS_5, 2))

    best_score = 0.0
    best_nodes: List[int] = []

    for l in _weeks(timecycle):
        hulls = {}
        hull_areas = {}

        for d in DAYS_5:
            ids_cell = visited[(l, d)]

            if len(ids_cell) < 3:
                hulls[d] = None
                hull_areas[d] = 0.0
                continue

            pts = [
                (float(stops[i].xcoord), float(stops[i].ycoord))
                for i in ids_cell
            ]

            hull = MultiPoint(pts).convex_hull

            if hull.geom_type != "Polygon" or hull.area <= 0:
                hulls[d] = None
                hull_areas[d] = 0.0
                continue

            hulls[d] = hull
            hull_areas[d] = float(hull.area)

        for d, e in day_pairs:
            hd = hulls.get(d)
            he = hulls.get(e)

            if hd is None or he is None:
                continue

            inter = hd.intersection(he)

            if inter.is_empty or inter.area <= 0:
                continue

            area_d = hull_areas[d]
            area_e = hull_areas[e]

            score = 0.0
            if area_d > 0:
                score += inter.area / area_d
            if area_e > 0:
                score += inter.area / area_e

            if score <= best_score:
                continue

            candidate_nodes = []

            for i in visited[(l, d)] + visited[(l, e)]:
                pt = Point(float(stops[i].xcoord), float(stops[i].ycoord))

                if inter.covers(pt):
                    candidate_nodes.append(i)

            if not candidate_nodes:
                candidate_nodes = list(set(visited[(l, d)] + visited[(l, e)]))

            best_score = score
            best_nodes = candidate_nodes

    return best_nodes


def _destroy_random(
    p: Dict[int, SchedTuple],
    q: int,
    rng: random.Random,
) -> Set[int]:
    ids = list(p.keys())
    return set(rng.sample(ids, min(q, len(ids))))


def _destroy_overlap_driven(
    stops: Dict[int, Stop],
    p: Dict[int, SchedTuple],
    timecycle: int,
    q: int,
) -> Set[int]:
    contrib = _stop_overlap_contribution(stops, p, timecycle)

    ranked = sorted(
        contrib.items(),
        key=lambda x: (-x[1], x[0]),
    )

    selected = [
        i for i, value in ranked
        if value > 0
    ][:q]

    if len(selected) < q:
        rest = [
            i for i, _ in ranked
            if i not in selected
        ]
        selected.extend(rest[: q - len(selected)])

    return set(selected)


def _destroy_worst_pair(
    stops: Dict[int, Stop],
    p: Dict[int, SchedTuple],
    timecycle: int,
    q: int,
    rng: random.Random,
) -> Set[int]:
    nodes = _worst_pair_nodes(stops, p, timecycle)

    if not nodes:
        return _destroy_random(p, q, rng)

    if len(nodes) <= q:
        return set(nodes)

    return set(rng.sample(nodes, q))

def _destroy_worst_nho_pair(
    stops: Dict[int, Stop],
    p: Dict[int, SchedTuple],
    timecycle: int,
    q: int,
    rng: random.Random,
) -> Set[int]:
    nodes = _worst_nho_pair_nodes(
        stops=stops,
        p=p,
        timecycle=timecycle,
    )

    if not nodes:
        return _destroy_random(p, q, rng)

    if len(nodes) <= q:
        return set(nodes)

    return set(rng.sample(nodes, q))
