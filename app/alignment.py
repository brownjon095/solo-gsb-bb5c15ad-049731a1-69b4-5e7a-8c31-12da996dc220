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

Optional **source-pair capacities** let the caller cap how many matched
pairs may join events of a given ``(left source_key, right source_key)``
combination. Combinations that are not declared stay unlimited. The
three optimisation layers are unchanged: they are now optimised *over
the matchings that respect every capacity* (a global optimum, not a
greedy repair).

Solver layout
-------------

* Without capacities, all three layers are encoded into a single
  exact-integer maximum-weight assignment solved by one Hungarian run:

  * the problem is padded to a square matrix (dummy rows/columns
    represent unmatched events), illegal edges and edges touching a
    dummy get weight 0;
  * a legal candidate edge (absolute difference ``d``) gets a strictly
    positive weight

      W(i, j) = C - d * S + (nr - j) * B**(nl - 1 - i)

    with ``B = nr + 1`` and ``S = B**nl``. The tie term follows from
    rewriting the base-B assignment-vector value (whose digit is the
    matched right rank or ``nr`` for an unmatched row):

      t(M) = nr * sum_i B**(nl-1-i)
             - sum over matched (i, j) of (nr - j) * B**(nl-1-i)

    so minimising ``t`` is exactly maximising the sum of ``(nr - j) *
    B**(nl-1-i)`` over the matched edges. ``C`` is large enough that
    every legal pair out-earns leaving events unmatched.

* With capacities the feasible matchings are the intersection of three
  partition matroids (left events, right events and per-combo quotas),
  which cannot be expressed as a plain assignment. An exact
  branch-and-bound search is used instead: rows are expanded in a
  most-constrained-first order and pruned with (a) augmenting-path
  cardinality bounds over the residual graph, clipped by the remaining
  quotas, and (b) a per-row minimum-difference lower bound. The
  Hungarian optimum is evaluated first; whenever it already respects
  every capacity it is reused verbatim, so the search only runs when a
  capacity actually binds.

Only the Python standard library is used, which keeps the module easy to
unit-test outside the containerised web stack.
"""

from __future__ import annotations

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
class SourceComboStat:
    """Actual usage / capacity outcome for one source-key combination."""

    left_source: str
    right_source: str
    capacity: int | None          # None = combination was not declared
    candidate_edges: int
    matched_pairs: int
    # Edges of the (unique) unconstrained optimum inside this combination
    # that the capacity-constrained result does not contain. Always 0 for
    # undeclared combinations and whenever no capacity binds.
    rejected_by_capacity: int


@dataclass(frozen=True)
class AlignmentResult:
    """Outcome of an alignment run."""

    tolerance_us: int
    pairs: list[MatchedPair]
    unmatched_left: list[str]
    unmatched_right: list[str]
    candidate_edge_count: int
    total_abs_diff_us: int
    source_combos: list[SourceComboStat]

    @property
    def pair_count(self) -> int:
        return len(self.pairs)


# Sentinel for "no candidate yet" during the Hungarian search. It only
# needs to exceed any reduced cost that can appear; the value is scaled
# dynamically inside the solver because edge weights can be large
# exact integers.
_DEFAULT_INF = 10 ** 60

# Capacity-constrained requests are solved by an exact search. The
# underlying problem is matching under per-edge-group quotas (an
# intersection of three partition matroids), which is NP-hard at scale,
# so the exact search is gated to small candidate graphs; larger inputs
# receive a clear 422 instead of a stalled worker. The limit only
# applies when at least one capacity is declared; 60 edges stays well
# within a second or two on measured random instances.
_MAX_CONSTRAINED_CANDIDATE_EDGES = 60
_SEARCH_NODE_LIMIT = 2_000_000


class ConstraintSearchLimit(RuntimeError):
    """Raised when a capacity-constrained instance exceeds the search budget."""


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


def _hungarian_matching(
    nl: int,
    nr: int,
    size: int,
    edge_diff: dict[int, dict[int, int]],
    tolerance_us: int,
) -> set[tuple[int, int]]:
    """Unconstrained global optimum as a set of canonical ``(i, j)`` edges."""
    # Lexicographic encoding (see module docstring).
    base = nr + 1
    tie_span = base ** nl                 # every tie value is in [0, tie_span)
    powers = [base ** (nl - 1 - i) for i in range(nl)]
    # Per-edge tie coefficient: matching row i to right-rank j gains
    # (nr - j) * p_i in the maximisation; j in [0, nr-1].
    tie_coef = [[(nr - j) * powers[i] for j in range(nr)] for i in range(nl)]

    # Cardinality layer: one extra pair must out-earn the maximum possible
    # loss in all lower layers (diff layer span + tie span).
    w_real_min = 1
    max_lower_loss = tolerance_us * tie_span + tie_span - 1
    c_const = size * max_lower_loss + w_real_min

    cost: list[list[int]] = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i < nl and j < nr:
                d = edge_diff.get(i, {}).get(j)
                if d is not None:
                    weight = c_const - d * tie_span + tie_coef[i][j]
                    cost[i][j] = -weight
            # illegal real-real edges and dummy-touching edges stay at 0

    assignment, _, _ = _hungarian_min(cost)

    matching: set[tuple[int, int]] = set()
    for i in range(nl):
        j = assignment[i + 1] - 1
        if 0 <= j < nr and j in edge_diff.get(i, {}):
            matching.add((i, j))
    return matching


def _matching_key(
    matching: set[tuple[int, int]],
    diff_of: dict[tuple[int, int], int],
    nl: int,
    nr: int,
) -> tuple[int, int, int]:
    """Lexicographic comparison key (smaller is better).

    ``(-cardinality, total abs diff, base-B assignment-vector value)``.
    """
    digits = [nr] * nl
    total = 0
    for i, j in matching:
        digits[i] = j
        total += diff_of[(i, j)]
    vec = 0
    for digit in digits:
        vec = vec * (nr + 1) + digit
    return (-len(matching), total, vec)


def _constrained_matching(
    nl: int,
    nr: int,
    edges_by_row: Sequence[Sequence[tuple[int, int, tuple[str, str]]]],
    caps: dict[tuple[str, str], int],
    diff_of: dict[tuple[int, int], int],
    seed: set[tuple[int, int]],
    combo_of: dict[tuple[int, int], tuple[str, str]],
) -> set[tuple[int, int]]:
    """Exact optimum under per-source-combo capacities (branch and bound).

    ``edges_by_row[i]`` lists ``(j, d, combo)`` per candidate of row i,
    already ordered by right rank. ``seed`` is the unconstrained
    Hungarian optimum used as the first incumbent; the search proves it
    optimal or replaces it.
    """

    def greedy(order: Sequence[int]) -> set[tuple[int, int]]:
        right_taken = [False] * nr
        combo_used: dict[tuple[str, str], int] = {}
        picked: set[tuple[int, int]] = set()
        for i in order:
            for j, _d, combo in edges_by_row[i]:
                if right_taken[j]:
                    continue
                if combo in caps and combo_used.get(combo, 0) >= caps[combo]:
                    continue
                right_taken[j] = True
                combo_used[combo] = combo_used.get(combo, 0) + 1
                picked.add((i, j))
                break
        return picked

    def repair(matching: set[tuple[int, int]]) -> set[tuple[int, int]]:
        # Drop the seed edges that violate combo quotas (row/right
        # uniqueness of the seed is already guaranteed), then greedily
        # re-fill the rows in canonical order.
        right_taken = [False] * nr
        combo_used: dict[tuple[str, str], int] = {}
        kept: set[tuple[int, int]] = set()
        for i, j in sorted(matching):
            combo = combo_of[(i, j)]
            if combo in caps and combo_used.get(combo, 0) >= caps[combo]:
                continue
            right_taken[j] = True
            combo_used[combo] = combo_used.get(combo, 0) + 1
            kept.add((i, j))
        for i in range(nl):
            if any(ii == i for ii, _ in kept):
                continue
            for j, _d, combo in edges_by_row[i]:
                if right_taken[j]:
                    continue
                if combo in caps and combo_used.get(combo, 0) >= caps[combo]:
                    continue
                right_taken[j] = True
                combo_used[combo] = combo_used.get(combo, 0) + 1
                kept.add((i, j))
                break
        return kept

    # --- incumbent: repaired Hungarian seed plus plain greedy variants ---
    incumbent = repair(seed)
    inc_key = _matching_key(incumbent, diff_of, nl, nr)
    for order in (
        list(range(nl)),
        sorted(range(nl), key=lambda i: (len(edges_by_row[i]), i)),
    ):
        cand = greedy(order)
        key = _matching_key(cand, diff_of, nl, nr)
        if key < inc_key:
            incumbent, inc_key = cand, key
    best_card = -inc_key[0]

    def residual_bound(
        rem: Sequence[int],
        avail: Sequence[bool],
        combo_used: dict[tuple[str, str], int],
    ) -> int:
        """Upper bound on extra matches from the residual rows.

        Kuhn augmenting matches are computed three times over the
        residual graph: all edges, undeclared-combo edges only and
        declared-combo edges only. The declared-class count is clipped
        by the total remaining quota; the undeclared and clipped
        declared halves are additive (their edge sets are disjoint).
        """

        def kuhn(declared: bool | None) -> int:
            match_r = [-1] * nr

            def try_augment(i: int, seen: list[bool]) -> bool:
                for j, _d, combo in edges_by_row[i]:
                    if declared is not None and (combo in caps) is not declared:
                        continue
                    if not avail[j] or seen[j]:
                        continue
                    seen[j] = True
                    if match_r[j] < 0 or try_augment(match_r[j], seen):
                        match_r[j] = i
                        return True
                return False

            count = 0
            for i in rem:
                if try_augment(i, [False] * nr):
                    count += 1
            return count

        k_all = kuhn(None)
        k_unc = kuhn(False)
        k_con = kuhn(True)
        remaining_quota = sum(
            caps[c] - combo_used.get(c, 0)
            for c in caps
            if combo_used.get(c, 0) < caps[c]
        )
        return min(k_all, k_unc + k_con, k_unc + remaining_quota)

    def relaxed_bound(
        rem: Sequence[int], avail: Sequence[bool], need: int
    ) -> tuple[int, int] | None:
        """Relaxed optimum (quota caps ignored) for exactly ``need`` edges.

        Min-cost flow of exactly ``need`` units on the residual bipartite
        DAG ``S -> rows -> free columns -> T`` with arc cost
        ``d * S - tie_gain`` (so the flow jointly minimises added total
        diff and, on ties, maximises the stable-ID tie gain). Ignoring
        the combo caps can only improve any subtree, hence the returned
        ``(added_diff, added_tie_gain)`` bounds every feasible
        completion. None when fewer than ``need`` residual edges exist.
        """
        if need <= 0:
            return 0, 0
        rows = list(rem)
        cols = [j for j in range(nr) if avail[j]]
        c_idx = {j: b for b, j in enumerate(cols)}
        R, C = len(rows), len(cols)
        s = R + C
        t = s + 1
        nverts = t + 1
        graph: list[list[list[int]]] = [[] for _ in range(nverts)]

        def add_edge(u, v, cap, cost):
            graph[u].append([v, cap, cost, len(graph[v])])
            graph[v].append([u, 0, -cost, len(graph[u]) - 1])

        span = (nr + 1) ** nl
        for a, i in enumerate(rows):
            add_edge(s, a, 1, 0)
            for j, d, _combo in edges_by_row[i]:
                if j in c_idx:
                    b = c_idx[j]
                    tie_gain = (nr - j) * (nr + 1) ** (nl - 1 - i)
                    add_edge(a, R + b, 1, d * span - tie_gain)
        for b in range(C):
            add_edge(R + b, t, 1, 0)

        flow = 0
        inf = 10 ** 100
        while flow < need:
            dist = [inf] * nverts
            in_q = [False] * nverts
            prev_v = [-1] * nverts
            prev_e = [-1] * nverts
            dist[s] = 0
            queue = [s]
            in_q[s] = True
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                in_q[u] = False
                for ei, (v, cap, cost, _rev) in enumerate(graph[u]):
                    if cap > 0 and dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        prev_v[v] = u
                        prev_e[v] = ei
                        if not in_q[v]:
                            queue.append(v)
                            in_q[v] = True
            if dist[t] == inf:
                return None
            v = t
            while v != s:
                u = prev_v[v]
                e = graph[u][prev_e[v]]
                e[1] -= 1
                rev = graph[e[0]][e[3]]
                rev[1] += 1
                v = u
            flow += 1

        # Extract the final flow: a row->column forward edge carries
        # one unit exactly when its residual capacity is 0.
        chosen = []
        for a, i in enumerate(rows):
            for e in graph[a]:
                v, cap, _cost, _rev = e
                if R <= v < R + C and cap == 0:
                    chosen.append((i, cols[v - R]))

        add_d = sum(diff_of[(i, j)] for i, j in chosen)
        add_tie_gain = sum(
            (nr - j) * (nr + 1) ** (nl - 1 - i) for i, j in chosen
        )
        return add_d, add_tie_gain

    # --- search state (backtracked scalars/arrays) ---
    avail = [True] * nr
    combo_used: dict[tuple[str, str], int] = {}
    chosen: set[tuple[int, int]] = set()
    nodes = 0

    def feasible_options(i: int) -> list[tuple[int, int, tuple[str, str]]]:
        return [
            (j, d, combo)
            for j, d, combo in edges_by_row[i]
            if avail[j]
            and (combo not in caps or combo_used.get(combo, 0) < caps[combo])
        ]

    def search(rem: list[int], card: int, total: int) -> None:
        nonlocal nodes, incumbent, inc_key, best_card
        nodes += 1
        if nodes > _SEARCH_NODE_LIMIT:
            raise ConstraintSearchLimit(
                "capacity-constrained alignment exceeded the search budget"
            )

        # Rows that currently have no feasible option are forced
        # unmatched; options only disappear deeper in the tree.
        forced: list[int] = []
        k = 0
        while k < len(rem):
            if feasible_options(rem[k]):
                k += 1
            else:
                forced.append(rem.pop(k))

        if not rem:
            digits = [nr] * nl
            for i, j in chosen:
                digits[i] = j
            vec = 0
            for digit in digits:
                vec = vec * (nr + 1) + digit
            key = (-card, total, vec)
            if key < inc_key:
                incumbent, inc_key = set(chosen), key
                best_card = card
            rem.extend(forced)
            return

        bound = card + residual_bound(rem, avail, combo_used)
        if bound < best_card:
            rem.extend(forced)
            return

        # If this subtree can at best tie the incumbent cardinality,
        # relax the caps and solve the residual lexicographic assignment
        # (diff, then stable-ID tie); prune when even the relaxation
        # cannot beat the incumbent at any layer. A cheap per-row sum is
        # checked first to avoid the O(n^3) relaxation on obvious misses.
        if bound == best_card:
            need = best_card - card
            cheapest = sorted(
                min(d for _j, d, _c in feasible_options(r))
                for r in rem
                if feasible_options(r)
            )
            if len(cheapest) < need:
                rem.extend(forced)
                return
            if total + sum(cheapest[:need]) > inc_key[1]:
                rem.extend(forced)
                return
            relaxed = relaxed_bound(rem, avail, need)
            if relaxed is None:
                rem.extend(forced)
                return
            add_d, add_tie_gain = relaxed
            if total + add_d > inc_key[1]:
                rem.extend(forced)
                return
            if total + add_d == inc_key[1]:
                # Assignment-vector value of the fixed prefix plus the
                # relaxation's maximal tie gain (all other rows stay
                # unmatched), compared against the incumbent vector.
                base = nr + 1
                prefix_gain = sum(
                    (nr - j) * base ** (nl - 1 - i) for i, j in chosen
                )
                prefix_vec = nr * sum(base ** k for k in range(nl))
                lb_vec = prefix_vec - prefix_gain - add_tie_gain
                if lb_vec >= inc_key[2]:
                    rem.extend(forced)
                    return

        # Most-constrained remaining row; try the cheapest edges first.
        i = min(rem, key=lambda r: (len(feasible_options(r)), r))
        rem.remove(i)

        for j, d, combo in sorted(feasible_options(i), key=lambda o: (o[1], o[0])):
            avail[j] = False
            combo_used[combo] = combo_used.get(combo, 0) + 1
            chosen.add((i, j))
            search(rem, card + 1, total + d)
            chosen.remove((i, j))
            combo_used[combo] -= 1
            avail[j] = True

        # Leave row i unmatched.
        search(rem, card, total)

        rem.append(i)
        rem.extend(forced)

    search(list(range(nl)), 0, 0)
    return incumbent


def align_events(
    left_ids: Sequence[str],
    left_times: Sequence[int],
    right_ids: Sequence[str],
    right_times: Sequence[int],
    tolerance_us: int,
    left_sources: Sequence[str] | None = None,
    right_sources: Sequence[str] | None = None,
    source_capacities: Sequence[tuple[str, str, int]] | None = None,
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
        ``source_key`` of each event, in the same order as the ids.
        Defaults to a single empty source for every event when omitted.
    source_capacities:
        Optional ``(left_source, right_source, max_matches)`` triples;
        pairs joining the named combination are capped at ``max_matches``
        (a non-negative integer, ``0`` forbids the combination).
        Combinations that are not declared are unlimited; declaring the
        same combination more than once is rejected.

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

    if left_sources is None:
        left_sources = [""] * nl
    if right_sources is None:
        right_sources = [""] * nr
    if len(left_sources) != nl or len(right_sources) != nr:
        raise ValueError("sources must line up with ids one-to-one")

    cap_map: dict[tuple[str, str], int] = {}
    if source_capacities:
        for ls, rs, cap in source_capacities:
            if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
                raise ValueError("source capacity must be a non-negative integer")
            key = (ls, rs)
            if key in cap_map:
                raise ValueError(
                    f"duplicate capacity for source combination ({ls!r}, {rs!r})"
                )
            cap_map[key] = cap

    if nl == 0 and nr == 0:
        combos = [
            SourceComboStat(ls, rs, cap, 0, 0, 0)
            for (ls, rs), cap in sorted(cap_map.items())
        ]
        return AlignmentResult(tolerance_us, [], [], [], 0, 0, combos)

    # Canonical order = ID order, so results are independent of input order.
    left_order = sorted(range(nl), key=lambda i: left_ids[i])
    right_order = sorted(range(nr), key=lambda j: right_ids[j])
    l_ids = [left_ids[i] for i in left_order]
    l_times = [int(left_times[i]) for i in left_order]
    l_src = [left_sources[i] for i in left_order]
    r_ids = [right_ids[j] for j in right_order]
    r_times = [int(right_times[j]) for j in right_order]
    r_src = [right_sources[j] for j in right_order]

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

    if cap_map and edge_count > _MAX_CONSTRAINED_CANDIDATE_EDGES:
        raise ConstraintSearchLimit(
            f"capacity-constrained alignment supports at most "
            f"{_MAX_CONSTRAINED_CANDIDATE_EDGES} candidate edges "
            f"(got {edge_count}); narrow tolerance_us or split the request"
        )

    size = max(nl, nr)
    m0_matching = _hungarian_matching(nl, nr, size, edge_diff, tolerance_us)

    # Edges annotated with their source combination.
    combo_of: dict[tuple[int, int], tuple[str, str]] = {}
    edges_by_row: list[list[tuple[int, int, tuple[str, str]]]] = [
        [] for _ in range(nl)
    ]
    diff_of: dict[tuple[int, int], int] = {}
    for i, row in edge_diff.items():
        for j, d in row.items():
            combo = (l_src[i], r_src[j])
            combo_of[(i, j)] = combo
            diff_of[(i, j)] = d
            edges_by_row[i].append((j, d, combo))  # j ascending

    matching = m0_matching
    binds = False
    if cap_map:
        used_in_m0: dict[tuple[str, str], int] = {}
        for i, j in m0_matching:
            c = combo_of[(i, j)]
            used_in_m0[c] = used_in_m0.get(c, 0) + 1
        binds = any(used_in_m0.get(c, 0) > cap for c, cap in cap_map.items())
        if binds:
            matching = _constrained_matching(
                nl, nr, edges_by_row, cap_map, diff_of, m0_matching, combo_of
            )

    pairs: list[MatchedPair] = []
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    total_abs = 0
    matched_by_combo: dict[tuple[str, str], int] = {}
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
        c = combo_of[(i, j)]
        matched_by_combo[c] = matched_by_combo.get(c, 0) + 1

    unmatched_left = sorted(
        (l_ids[i] for i in range(nl) if i not in matched_left)
    )
    unmatched_right = sorted(
        (r_ids[j] for j in range(nr) if j not in matched_right)
    )
    # Deterministic, ID-stable presentation order.
    pairs.sort(key=lambda p: (p.left_id, p.right_id))

    # Per-combination statistics: every combination that either has
    # candidate edges or was explicitly declared, sorted by source keys.
    candidate_by_combo: dict[tuple[str, str], int] = {}
    for i, row in edge_diff.items():
        for j in row:
            c = (l_src[i], r_src[j])
            candidate_by_combo[c] = candidate_by_combo.get(c, 0) + 1

    rejected_by_combo: dict[tuple[str, str], int] = {}
    if cap_map and binds:
        # How many pairs the (unique) unconstrained optimum placed in
        # each declared combination versus what the capacity-constrained
        # optimum keeps. The shortfall is the candidate edge count
        # refused by that combination's cap.
        for c, unc in used_in_m0.items():
            if c in cap_map:
                shortfall = unc - matched_by_combo.get(c, 0)
                if shortfall > 0:
                    rejected_by_combo[c] = shortfall

    all_combos = sorted(set(candidate_by_combo) | set(cap_map))
    source_combos = [
        SourceComboStat(
            left_source=ls,
            right_source=rs,
            capacity=cap_map.get((ls, rs)),
            candidate_edges=candidate_by_combo.get((ls, rs), 0),
            matched_pairs=matched_by_combo.get((ls, rs), 0),
            rejected_by_capacity=rejected_by_combo.get((ls, rs), 0),
        )
        for ls, rs in all_combos
    ]

    return AlignmentResult(
        tolerance_us=tolerance_us,
        pairs=pairs,
        unmatched_left=unmatched_left,
        unmatched_right=unmatched_right,
        candidate_edge_count=edge_count,
        total_abs_diff_us=total_abs,
        source_combos=source_combos,
    )
