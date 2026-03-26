from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from data_type import DAYS_5 as DAYS
from helper_funcs import a, get_dist

def find_components(edges):
    parent = {}
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        parent[find(a)] = find(b)

    nodes = set()
    for i, j in edges:
        nodes.add(i)
        nodes.add(j)

    for n in nodes:
        parent[n] = n

    for i, j in edges:
        union(i, j)

    comp = {}
    for n in nodes:
        root = find(n)
        comp.setdefault(root, []).append(n)

    return list(comp.values())

def solve_formulation_medoid(
    stops,
    pi,
    baseline_sched,
    timecycle,
    v_max,
    g_max,
    c_max,
    Ddist,
    w1=1.0,
    w2=1.0,
    time_limit=600,
    mip_gap=0.0,
):

    WEEKS_local = list(range(1, timecycle + 1))
    ids = list(stops.keys())

    m = gp.Model("Medoid_MST")

    m.setParam(GRB.Param.TimeLimit, time_limit)
    m.setParam(GRB.Param.MIPGap, mip_gap)

    # 변수
    x = {(i, p): m.addVar(vtype=GRB.BINARY) for i in ids for p in pi[i]}
    y = {(i, l, d): m.addVar(vtype=GRB.BINARY) for i in ids for l in WEEKS_local for d in DAYS}
    c = {i: m.addVar(vtype=GRB.BINARY) for i in ids}

    # medoid
    m_medoid = {(j, l, d): m.addVar(vtype=GRB.BINARY)
                for j in ids for l in WEEKS_local for d in DAYS}

    # edge set
    K = 20  # 또는 15~30 사이 추천
    E = set()

    for i in ids:
        dists = sorted(
            [(get_dist(i, j, Ddist, stops), j) for j in ids if j != i],
            key=lambda x: x[0]
        )[:K]

        for _, j in dists:
            if i < j:
                E.add((i, j))
            else:
                E.add((j, i))

    E = list(E)

    e = {(i, j, l, d): m.addVar(vtype=GRB.BINARY)
         for (i, j) in E for l in WEEKS_local for d in DAYS}

    # MST length
    w = {(l, d): m.addVar(lb=0.0) for l in WEEKS_local for d in DAYS}

    # volume
    Vday = {(l, d): m.addVar(lb=0.0) for l in WEEKS_local for d in DAYS}
    Vmax = {l: m.addVar(lb=0.0) for l in WEEKS_local}
    Vmin = {l: m.addVar(lb=0.0) for l in WEEKS_local}

    m.update()

    # Objective
    term1 = gp.quicksum(
        get_dist(i, j, Ddist, stops) * y[(i, l, d)] * m_medoid[(j, l, d)]
        for i in ids for j in ids
        for l in WEEKS_local for d in DAYS
    )

    m.setObjective(
        w1 * term1 +
        w2 * gp.quicksum(Vmax[l] - Vmin[l] for l in WEEKS_local),
        GRB.MINIMIZE
    )

    # Constraints
    # (1)
    for i in ids:
        m.addConstr(gp.quicksum(x[(i, p)] for p in pi[i]) == 1)

    # (2)
    for i in ids:
        for l in WEEKS_local:
            for d in DAYS:
                m.addConstr(
                    y[(i, l, d)] == gp.quicksum(a(p, l, d) * x[(i, p)] for p in pi[i])
                )

    # (3)(4)
    for l in WEEKS_local:
        for d in DAYS:
            vol = gp.quicksum(stops[i].volume * y[(i, l, d)] for i in ids)
            m.addConstr(Vday[(l, d)] == vol)
            m.addConstr(vol <= v_max)

            m.addConstr(
                gp.quicksum(stops[i].weight * y[(i, l, d)] for i in ids) <= g_max
            )

    # (5)(6)
    for i in ids:
        p0 = baseline_sched[i]
        m.addConstr(c[i] + x[(i, p0)] == 1)

    m.addConstr(gp.quicksum(c[i] for i in ids) <= c_max)

    # medoid
    for l in WEEKS_local:
        for d in DAYS:
            m.addConstr(gp.quicksum(m_medoid[(j, l, d)] for j in ids) == 1)

    for j in ids:
        for l in WEEKS_local:
            for d in DAYS:
                m.addConstr(m_medoid[(j, l, d)] <= y[(j, l, d)])

    # edge
    for (i, j) in E:
        for l in WEEKS_local:
            for d in DAYS:
                m.addConstr(e[(i, j, l, d)] <= y[(i, l, d)])
                m.addConstr(e[(i, j, l, d)] <= y[(j, l, d)])

    # 🔥 edge count (수정 완료 버전)
    for l in WEEKS_local:
        for d in DAYS:
            m.addConstr(
                gp.quicksum(e[(i, j, l, d)] for (i, j) in E)
                <= gp.quicksum(y[(i, l, d)] for i in ids) - 1
            )

    # MST length
    for l in WEEKS_local:
        for d in DAYS:
            m.addConstr(
                w[(l, d)] ==
                gp.quicksum(get_dist(i, j, Ddist, stops) * e[(i, j, l, d)]
                            for (i, j) in E)
            )

    # balancing
    for l in WEEKS_local:
        for d in DAYS:
            m.addConstr(Vmax[l] >= w[(l, d)])
            m.addConstr(Vmin[l] <= w[(l, d)])

    for l in WEEKS_local:
        for d in DAYS:
            m.addConstr(Vmax[l] >= Vday[(l, d)])
            m.addConstr(Vmin[l] <= Vday[(l, d)])

    # Lazy constraint
    def find_components(edges):
        parent = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            parent[find(a)] = find(b)

        nodes = set()
        for i, j in edges:
            nodes.add(i)
            nodes.add(j)

        for n in nodes:
            parent[n] = n

        for i, j in edges:
            union(i, j)

        comp = {}
        for n in nodes:
            root = find(n)
            comp.setdefault(root, []).append(n)

        return list(comp.values())

    def mst_callback(model, where):
        if where == GRB.Callback.MIPSOL:

            vals = model.cbGetSolution(model._e)
            yvals = model.cbGetSolution(model._y)

            for l in model._L:
                for d in model._D:

                    active_nodes = [i for i in model._ids if yvals[(i, l, d)] > 0.5]

                    edges = [(i, j) for (i, j) in model._E
                             if vals[(i, j, l, d)] > 0.5]

                    if len(active_nodes) <= 1:
                        continue

                    # ❗ connectivity 체크
                    if len(edges) < len(active_nodes) - 1:
                        model.cbLazy(
                            gp.quicksum(
                                model._e[(i, j, l, d)] for (i, j) in model._E
                            )
                            >= len(active_nodes) - 1
                        )

                    comps = find_components(edges)

                    for comp in comps:
                        if len(comp) <= 1:
                            continue

                        edge_count = sum(
                            vals[(i, j, l, d)]
                            for (i, j) in model._E
                            if i in comp and j in comp
                        )

                        if edge_count >= len(comp):
                            model.cbLazy(
                                gp.quicksum(
                                    model._e[(i, j, l, d)]
                                    for (i, j) in model._E
                                    if i in comp and j in comp
                                )
                                <= len(comp) - 1
                            )

    # attach
    m._e = e
    m._y = y
    m._E = E
    m._ids = ids
    m._L = WEEKS_local
    m._D = DAYS

    m.Params.LazyConstraints = 1

    # solve
    m.optimize(mst_callback)

    if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Optimization ended with status {m.Status}")

    chosen_tuple = {}
    for i in ids:
        for p in pi[i]:
            if x[(i, p)].X > 0.5:
                chosen_tuple[i] = p
                break

    return m, chosen_tuple, {i: int(round(c[i].X)) for i in ids}