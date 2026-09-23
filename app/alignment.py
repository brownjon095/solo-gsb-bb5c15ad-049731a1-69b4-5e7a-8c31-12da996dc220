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
assignment solved by one Hungarian run:

* the problem is padded to a square matrix (dummy rows/columns represent
  unmatched events), illegal edges and edges touching a dummy get weight
  0;
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
  (layer 3).

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


def align_events(
    left_ids: Sequence[str],
    left_times: Sequence[int],
    right_ids: Sequence[str],
    right_times: Sequence[int],
    tolerance_us: int,
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

    if nl == 0 and nr == 0:
        return AlignmentResult(tolerance_us, [], [], [], 0, 0)

    # Canonical order = ID order, so results are independent of input order.
    left_order = sorted(range(nl), key=lambda i: left_ids[i])
    right_order = sorted(range(nr), key=lambda j: right_ids[j])
    l_ids = [left_ids[i] for i in left_order]
    l_times = [int(left_times[i]) for i in left_order]
    r_ids = [right_ids[j] for j in right_order]
    r_times = [int(right_times[j]) for j in right_order]

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

    size = max(nl, nr)

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

    pairs: list[MatchedPair] = []
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    total_abs = 0
    for i in range(nl):
        j = assignment[i + 1] - 1
        if 0 <= j < nr and j in edge_diff.get(i, {}):
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
