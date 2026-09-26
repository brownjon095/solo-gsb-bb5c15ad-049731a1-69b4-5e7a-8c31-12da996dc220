"""Cross-source event alignment core.

Given two groups of timestamped events, build candidate edges whose
absolute time difference is within a tolerance window and find a
one-to-one matching that lexicographically minimises:

1. the number of unmatched real events (maximum cardinality);
2. the total absolute time difference (microseconds);
3. a stable event-ID tie-break: with left events sorted by ID and right
   events ranked by ID, compare the assignment vector
   ``(partner_rank_or_UNMATCHED for each left event in ID order)``
   lexicographically, where UNMATCHED sorts after every right rank.

All three layers are encoded into a single exact-integer maximum-weight
assignment:

* a legal candidate edge (absolute difference ``d``) gets a strictly
  positive weight

      W(i, j) = C - d * S + (nr - j) * B**(nl - 1 - i)

  with ``B = nr + 1`` and ``S = B**nl``. The tie term follows from
  rewriting the base-B assignment-vector value (whose digit is the
  matched right rank or ``nr`` for an unmatched row):

      t(M) = nr * sum_i B**(nl-1-i)
             - sum over matched (i, j) of (nr - j) * B**(nl-1-i)

  so minimising ``t`` is exactly maximising the sum of ``(nr - j) *
  B**(nl-1-i)`` over the matched edges. ``C`` is large enough that every
  legal pair out-earns leaving events unmatched. Maximising the total
  weight therefore first maximises cardinality (layer 1), then minimises
  total ``d`` (layer 2, multiples of S never carry into the cardinality
  layer), then minimises the base-B value of the assignment vector
  (layer 3). The encoding is independent of the feasible set, so the
  same weights are reused for the capacitated solver below.

Without further constraints the maximum is found by one Hungarian run on
a square matrix padded with dummy rows/columns (dummies represent
unmatched events; illegal edges and edges touching a dummy get weight 0).

Optional source-pair capacities
-------------------------------
Callers may additionally declare, per ``(left source_key, right
source_key)`` combination, the maximum number of matched pairs that
combination may contribute; undeclared combinations stay unlimited. The
objective is unchanged -- the lexicographic optimum is now taken over
capacity-feasible matchings only.

Capacitated b-matching is not a pure assignment problem (its constraint
matrix is not totally unimodular and the problem is NP-hard in general),
so it cannot be solved by a single Hungarian run. The solver therefore
works in two steps:

1. Fast path: compute the (unique) unconstrained optimum with the
   Hungarian algorithm. If it already satisfies every capacity, it is
   the constrained global optimum as well -- this covers every request
   whose capacities do not bind, including all legacy requests.
2. Otherwise an exact branch-and-bound search over left events
   maximises the same weight function under the capacity constraints.
   Branches are pruned with a capacity-aware upper bound: the number of
   additional pairs is bounded by a max-flow relaxation (source -> left
   events -> combo nodes -> right events -> sink, combo nodes carrying
   the remaining capacities), times the largest edge weight.

The search is exact (global optimum) but exponential in the worst case;
keep capacity-constrained jobs to modest sizes.

Only the Python standard library is used, which keeps the module easy to
unit-test outside the containerised web stack.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MatchedPair:
    """A matched pair of events."""

    left_id: str
    right_id: str
    diff_us: int          # signed: left_time - right_time
    abs_diff_us: int      # abs(diff_us), also the primary cost


@dataclass(frozen=True)
class AlignmentResult:
    """Outcome of an alignment run."""

    tolerance_us: int
    pairs: list[MatchedPair]
    unmatched_left: list[str]
    unmatched_right: list[str]
    candidate_edge_count: int
    total_abs_diff_us: int

    @property
    def pair_count(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True)
class SourcePairStat:
    """Match statistics for one ``(left source, right source)`` combination."""

    left_source_key: str
    right_source_key: str
    capacity: int | None      # declared max pairs; None = unlimited
    candidate_edges: int
    matched_pairs: int
    rejected_edges: int       # candidate edges denied by the capacity


# Sentinel for "no candidate yet" during the Hungarian search. It only
# needs to exceed any reduced cost that can appear; the value is scaled
# dynamically inside the solver because edge weights can be large
# exact integers.
_DEFAULT_INF = 10 ** 60


def _hungarian_min(cost: Sequence[Sequence[int]]):
    """Hungarian algorithm (Jonker-Volgenant / Kuhn-Munkres potentials).

    ``cost`` is a square ``n x n`` matrix; a finite perfect matching must
    exist. Returns ``(assignment, u, v)`` where
    ``assignment[row1based] = col1based`` and ``(u, v)`` are feasible
    dual potentials indexed by row/column.
    """
    n = len(cost)
    # The "no candidate" sentinel must dominate every reduced cost that
    # can arise; potentials can accumulate up to O(n^2) edge-weight
    # magnitudes, so scale by the largest absolute entry.
    max_abs = 0
    for row in cost:
        for value in row:
            if value < 0:
                value = -value
            if value > max_abs:
                max_abs = value
    inf = n * n * max_abs + 1 if max_abs else _DEFAULT_INF

    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)          # p[col] = row matched to that column
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [0] * (n + 1)
    for j in range(1, n + 1):
        assignment[p[j]] = j
    return assignment, u, v


def _lexicographic_layers(nl: int, nr: int, tolerance_us: int):
    """Weight layers encoding the three optimisation criteria.

    Returns ``(tie_span, c_const, tie_coef)`` such that the edge weight
    ``W(i, j) = c_const - d * tie_span + tie_coef[i][j]`` (``d`` = absolute
    time difference) lexicographically orders every matching by
    (cardinality, -total difference, -assignment-vector value). See the
    module docstring for the derivation.
    """
    base = nr + 1
    tie_span = base ** nl                 # every tie value is in [0, tie_span)
    powers = [base ** (nl - 1 - i) for i in range(nl)]
    # Per-edge tie coefficient: matching row i to right-rank j gains
    # (nr - j) * p_i in the maximisation; j in [0, nr-1].
    tie_coef = [[(nr - j) * powers[i] for j in range(nr)] for i in range(nl)]

    # Cardinality layer: one extra pair must out-earn the maximum possible
    # loss in all lower layers (diff layer span + tie span).
    size = max(nl, nr)
    max_lower_loss = tolerance_us * tie_span + tie_span - 1
    c_const = size * max_lower_loss + 1
    return tie_span, c_const, tie_coef


def _hungarian_matching(
    nl: int,
    nr: int,
    edge_diff: dict[int, dict[int, int]],
    tie_span: int,
    c_const: int,
    tie_coef: list[list[int]],
) -> list[tuple[int, int]]:
    """Max-weight matching ignoring capacities (one Hungarian run).

    Returns the matching as a list of ``(left_index, right_index)`` pairs
    in canonical (ID-sorted) indices.
    """
    size = max(nl, nr)
    cost: list[list[int]] = [[0] * size for _ in range(size)]
    for i in range(nl):
        row = edge_diff.get(i)
        if not row:
            continue
        coefs = tie_coef[i]
        cost_row = cost[i]
        for j, d in row.items():
            cost_row[j] = -(c_const - d * tie_span + coefs[j])
        # illegal real-real edges and dummy-touching edges stay at 0

    assignment, _, _ = _hungarian_min(cost)

    matching: list[tuple[int, int]] = []
    for i in range(nl):
        j = assignment[i + 1] - 1
        if 0 <= j < nr and j in edge_diff.get(i, {}):
            matching.append((i, j))
    return matching


def _max_flow(arc_caps: dict[tuple[int, int], int], source: int, sink: int,
              node_count: int) -> int:
    """Edmonds-Karp maximum flow for small integer-capacity networks.

    ``arc_caps`` maps ``(u, v) -> capacity`` (directed arcs). Only used for
    the branch-and-bound upper bound on tiny graphs, so the simple
    BFS augmenting-path implementation is plenty.
    """
    residual: list[dict[int, int]] = [dict() for _ in range(node_count)]
    for (u, v), c in arc_caps.items():
        if c <= 0:
            continue
        residual[u][v] = residual[u].get(v, 0) + c
        residual[v].setdefault(u, 0)

    flow = 0
    while True:
        parent = [-1] * node_count
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] < 0:
            u = queue.popleft()
            for v, c in residual[u].items():
                if c > 0 and parent[v] < 0:
                    parent[v] = u
                    queue.append(v)
        if parent[sink] < 0:
            return flow
        # Bottleneck of the augmenting path.
        bottleneck = 0
        v = sink
        while v != source:
            u = parent[v]
            c = residual[u][v]
            if bottleneck == 0 or c < bottleneck:
                bottleneck = c
            v = u
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= bottleneck
            residual[v][u] = residual[v].get(u, 0) + bottleneck
            v = u
        flow += bottleneck


def _remaining_cardinality_bound(
    start: int,
    nl: int,
    nr: int,
    left_edges: list[list[tuple[int, int, int]]],
    used_right: list[bool],
    caps_remaining: list[int],
) -> int:
    """Upper bound on how many more pairs can still be formed.

    Relaxes the exact per-edge pairing inside each source combo (a
    relaxation, hence a valid upper bound) into a small flow network:

        source -> remaining left events (cap 1)
        left event -> combo node / right event directly (cap 1)
        combo node -> right events (throughput = remaining capacity)
        right events -> sink (cap 1)

    Every capacity-feasible completion maps to a flow of the same value,
    so the max-flow value bounds the number of additional pairs.
    """
    n_combos = len(caps_remaining)
    src = 0
    left0 = 1
    right0 = left0 + (nl - start)
    combo0 = right0 + nr
    sink = combo0 + 2 * n_combos

    arcs: dict[tuple[int, int], int] = {}

    def add(u: int, v: int, c: int) -> None:
        arcs[(u, v)] = arcs.get((u, v), 0) + c

    for a in range(start, nl):
        node_a = left0 + (a - start)
        has_edge = False
        for (j, _w, c) in left_edges[a]:
            if used_right[j]:
                continue
            if c >= 0:
                if caps_remaining[c] <= 0:
                    continue
                # Duplicate arcs per (left, combo) / (combo, right) pair
                # are harmless: unit capacities at both ends dominate.
                add(node_a, combo0 + 2 * c, 1)
                add(combo0 + 2 * c + 1, right0 + j, 1)
            else:
                add(node_a, right0 + j, 1)
            has_edge = True
        if has_edge:
            add(src, node_a, 1)
    for j in range(nr):
        if not used_right[j]:
            add(right0 + j, sink, 1)
    for c in range(n_combos):
        if caps_remaining[c] > 0:
            add(combo0 + 2 * c, combo0 + 2 * c + 1, caps_remaining[c])
    if not arcs:
        return 0
    return _max_flow(arcs, src, sink, sink + 1)


def _capacitated_matching(
    nl: int,
    nr: int,
    edge_diff: dict[int, dict[int, int]],
    edge_combo: dict[tuple[int, int], int],
    caps: list[int],
    tie_span: int,
    c_const: int,
    tie_coef: list[list[int]],
) -> list[tuple[int, int]]:
    """Exact max-weight matching under source-pair capacity constraints.

    ``edge_combo`` maps each candidate edge to its capacity-combo index
    (``-1`` = unlimited combination); ``caps`` holds the per-combo maximum
    number of pairs. Returns the lexicographically optimal
    capacity-feasible matching as ``(left_index, right_index)`` pairs.
    """
    # Fast path: the unconstrained optimum is unique (edge weights make
    # distinct matchings differ in total weight), so if it already
    # satisfies every capacity it is the constrained optimum as well.
    matching = _hungarian_matching(nl, nr, edge_diff, tie_span, c_const, tie_coef)
    usage = [0] * len(caps)
    feasible = True
    for (i, j) in matching:
        c = edge_combo[(i, j)]
        if c >= 0:
            usage[c] += 1
            if usage[c] > caps[c]:
                feasible = False
                break
    if feasible:
        return matching

    # Exact branch and bound over left events (canonical ID order),
    # maximising the same lexicographic weight under the capacities.
    weights: dict[tuple[int, int], int] = {}
    max_weight = 0
    for i, row in edge_diff.items():
        coefs = tie_coef[i]
        for j, d in row.items():
            w = c_const - d * tie_span + coefs[j]
            weights[(i, j)] = w
            if w > max_weight:
                max_weight = w

    # Per-left candidate edges, best weight first (finds strong
    # incumbents early, which sharpens pruning).
    left_edges: list[list[tuple[int, int, int]]] = []
    for i in range(nl):
        row = edge_diff.get(i)
        edges: list[tuple[int, int, int]] = []
        if row:
            for j in row:
                edges.append((j, weights[(i, j)], edge_combo[(i, j)]))
            edges.sort(key=lambda e: (-e[1], e[0]))
        left_edges.append(edges)

    used_right = [False] * nr
    quota_used = [0] * len(caps)
    current = [-1] * nl
    best_assignment = [-1] * nl
    best_weight = 0                     # the empty matching is always feasible

    def dfs(i: int, weight_so_far: int) -> None:
        nonlocal best_weight, best_assignment
        if i == nl:
            if weight_so_far > best_weight:
                best_weight = weight_so_far
                best_assignment = current.copy()
            return
        # Cheap static bound first, then the tighter capacity-aware
        # max-flow bound. Distinct matchings have distinct total weights,
        # so a branch that can at most tie the incumbent is prunable.
        if weight_so_far + (nl - i) * max_weight <= best_weight:
            return
        caps_remaining = [caps[c] - quota_used[c] for c in range(len(caps))]
        extra = _remaining_cardinality_bound(
            i, nl, nr, left_edges, used_right, caps_remaining
        )
        if weight_so_far + extra * max_weight <= best_weight:
            return
        for (j, w, c) in left_edges[i]:
            if used_right[j] or (c >= 0 and quota_used[c] >= caps[c]):
                continue
            used_right[j] = True
            if c >= 0:
                quota_used[c] += 1
            current[i] = j
            dfs(i + 1, weight_so_far + w)
            current[i] = -1
            if c >= 0:
                quota_used[c] -= 1
            used_right[j] = False
        dfs(i + 1, weight_so_far)       # leave event i unmatched

    dfs(0, 0)
    return [(i, best_assignment[i]) for i in range(nl) if best_assignment[i] >= 0]


def align_events(
    left_ids: Sequence[str],
    left_times: Sequence[int],
    right_ids: Sequence[str],
    right_times: Sequence[int],
    tolerance_us: int,
    *,
    left_sources: Sequence[str] | None = None,
    right_sources: Sequence[str] | None = None,
    capacities: Mapping[tuple[str, str], int] | None = None,
) -> AlignmentResult:
    """Align two event groups.

    Parameters
    ----------
    left_ids, left_times:
        Event IDs and timestamps (integer microseconds since the epoch,
        e.g. aware-UTC datetimes converted by the caller) of group A.
        Events are reordered internally by ID, so callers may pass them in
        any order.
    right_ids, right_times:
        Same for group B.
    tolerance_us:
        Non-negative window. A candidate edge exists only when
        ``abs(t_left - t_right) <= tolerance_us``.
    left_sources, right_sources:
        Optional per-event source keys (same order as the IDs). Required
        when ``capacities`` is given.
    capacities:
        Optional mapping ``(left_source_key, right_source_key) -> max
        pairs`` limiting how many matched pairs each source combination
        may contribute. Combinations without an entry are unlimited. When
        omitted or empty, the alignment is the unconstrained Hungarian
        optimum (legacy behaviour).

    IDs must be unique within each group.
    """
    nl = len(left_ids)
    nr = len(right_ids)
    if len(left_times) != nl or len(right_times) != nr:
        raise ValueError("ids and times must have equal length")
    if tolerance_us < 0:
        raise ValueError("tolerance_us must be non-negative")
    if nl != len(set(left_ids)) or nr != len(set(right_ids)):
        raise ValueError("event ids must be unique within each side")
    if capacities:
        if left_sources is None or right_sources is None:
            raise ValueError("source keys are required when capacities are declared")
        if len(left_sources) != nl or len(right_sources) != nr:
            raise ValueError("sources and ids must have equal length")
        for key, cap in capacities.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(isinstance(k, str) and k for k in key)
            ):
                raise ValueError(
                    "capacity keys must be (left_source_key, right_source_key) "
                    "pairs of non-empty strings"
                )
            if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
                raise ValueError("capacity values must be non-negative integers")

    if nl == 0 and nr == 0:
        return AlignmentResult(tolerance_us, [], [], [], 0, 0)

    # Canonical order = ID order, so results are independent of input order.
    left_order = sorted(range(nl), key=lambda i: left_ids[i])
    right_order = sorted(range(nr), key=lambda j: right_ids[j])
    l_ids = [left_ids[i] for i in left_order]
    l_times = [int(left_times[i]) for i in left_order]
    r_ids = [right_ids[j] for j in right_order]
    r_times = [int(right_times[j]) for j in right_order]
    l_sources = [left_sources[i] for i in left_order] if left_sources is not None else None
    r_sources = [right_sources[j] for j in right_order] if right_sources is not None else None

    # Candidate edges in canonical indices: row -> {column: abs diff}.
    edge_diff: dict[int, dict[int, int]] = {}
    edge_count = 0
    for i, tid in enumerate(l_times):
        row: dict[int, int] = {}
        for j, tjd in enumerate(r_times):
            d = abs(tid - tjd)
            if d <= tolerance_us:
                row[j] = d
        if row:
            edge_diff[i] = row
            edge_count += len(row)

    # Lexicographic weight encoding (see module docstring); independent
    # of the feasible set, so both solver paths share it.
    tie_span, c_const, tie_coef = _lexicographic_layers(nl, nr, tolerance_us)

    if capacities:
        combo_keys = sorted(capacities)
        combo_index = {key: idx for idx, key in enumerate(combo_keys)}
        caps = [int(capacities[key]) for key in combo_keys]
        edge_combo: dict[tuple[int, int], int] = {}
        for i, row in edge_diff.items():
            ls = l_sources[i]
            for j in row:
                edge_combo[(i, j)] = combo_index.get((ls, r_sources[j]), -1)
        matching = _capacitated_matching(
            nl, nr, edge_diff, edge_combo, caps, tie_span, c_const, tie_coef
        )
    else:
        matching = _hungarian_matching(
            nl, nr, edge_diff, tie_span, c_const, tie_coef
        )

    pairs: list[MatchedPair] = []
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    total_abs = 0
    for i, j in matching:
        diff = l_times[i] - r_times[j]
        pairs.append(
            MatchedPair(
                left_id=l_ids[i],
                right_id=r_ids[j],
                diff_us=diff,
                abs_diff_us=abs(diff),
            )
        )
        matched_left.add(i)
        matched_right.add(j)
        total_abs += abs(diff)

    unmatched_left = sorted(
        (l_ids[i] for i in range(nl) if i not in matched_left)
    )
    unmatched_right = sorted(
        (r_ids[j] for j in range(nr) if j not in matched_right)
    )
    # Deterministic, ID-stable presentation order.
    pairs.sort(key=lambda p: (p.left_id, p.right_id))

    return AlignmentResult(
        tolerance_us=tolerance_us,
        pairs=pairs,
        unmatched_left=unmatched_left,
        unmatched_right=unmatched_right,
        candidate_edge_count=edge_count,
        total_abs_diff_us=total_abs,
    )


def source_pair_stats(
    left_ids: Sequence[str],
    left_times: Sequence[int],
    left_sources: Sequence[str],
    right_ids: Sequence[str],
    right_times: Sequence[int],
    right_sources: Sequence[str],
    tolerance_us: int,
    pairs: Sequence[tuple[str, str]],
    capacities: Mapping[tuple[str, str], int] | None,
) -> list[SourcePairStat]:
    """Per-(left source, right source) matching statistics.

    ``pairs`` holds the matched ``(left_id, right_id)`` pairs of a finished
    alignment; ``capacities`` maps ``(left_source_key, right_source_key) ->
    max pairs`` and may be empty/None (all combinations unlimited).

    One entry is returned for every combination that has at least one
    candidate edge, at least one matched pair, or a declared capacity,
    sorted by ``(left_source_key, right_source_key)``. ``rejected_edges``
    counts candidate edges that the declared capacity alone rules out,
    i.e. ``max(0, candidate_edges - capacity)``; it is 0 for combinations
    without a declared capacity.
    """
    caps = capacities or {}
    candidate: dict[tuple[str, str], int] = {}
    for a in range(len(left_ids)):
        ta = int(left_times[a])
        ls = left_sources[a]
        for b in range(len(right_ids)):
            if abs(ta - int(right_times[b])) <= tolerance_us:
                key = (ls, right_sources[b])
                candidate[key] = candidate.get(key, 0) + 1

    left_source_of = dict(zip(left_ids, left_sources))
    right_source_of = dict(zip(right_ids, right_sources))
    matched: dict[tuple[str, str], int] = {}
    for left_id, right_id in pairs:
        key = (left_source_of[left_id], right_source_of[right_id])
        matched[key] = matched.get(key, 0) + 1

    stats: list[SourcePairStat] = []
    for key in sorted(set(candidate) | set(matched) | set(caps)):
        cap = caps.get(key)
        cand = candidate.get(key, 0)
        rejected = max(0, cand - cap) if cap is not None else 0
        stats.append(
            SourcePairStat(
                left_source_key=key[0],
                right_source_key=key[1],
                capacity=cap,
                candidate_edges=cand,
                matched_pairs=matched.get(key, 0),
                rejected_edges=rejected,
            )
        )
    return stats
