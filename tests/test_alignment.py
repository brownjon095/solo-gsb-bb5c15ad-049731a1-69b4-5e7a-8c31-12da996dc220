"""Unit tests for the alignment core (stdlib only)."""

from __future__ import annotations

import itertools
import random
import unittest

from app.alignment import align_events, source_pair_stats


def _ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _brute_force(l_times, r_times, tol):
    """Reference optimum matching the API's exact lexicographic rule.

    Order of criteria:
      1. maximum cardinality;
      2. minimum total absolute difference;
      3. minimum base-(nr+1) value of the assignment vector read across
         left IDs (digit = matched right ID-rank, nr = unmatched).
    """
    nl, nr = len(l_times), len(r_times)
    base = nr + 1
    best = None

    def consider(matching):
        nonlocal best
        left_seen: set[int] = set()
        right_seen: set[int] = set()
        total = 0
        vector = [nr] * nl
        for i, j in matching:
            if i in left_seen or j in right_seen:
                return
            if abs(l_times[i] - r_times[j]) > tol:
                return
            left_seen.add(i)
            right_seen.add(j)
            total += abs(l_times[i] - r_times[j])
            vector[i] = j
        tie = 0
        for digit in vector:
            tie = tie * base + digit
        score = (-len(matching), total, tie)
        if best is None or score < best[0]:
            best = (score, frozenset(matching))

    for size in range(0, min(nl, nr) + 1):
        for li in itertools.combinations(range(nl), size):
            for rj in itertools.combinations(range(nr), size):
                for perm in itertools.permutations(rj):
                    consider(list(zip(li, perm)))
    if best is None:
        # empty matching is always considered (size 0), so this cannot happen
        raise AssertionError
    return best[1]


def _brute_force_capacitated(l_times, r_times, tol, l_srcs, r_srcs, caps):
    """Reference optimum under source-pair capacities (same lexicographic
    rule as `_brute_force`, restricted to capacity-feasible matchings)."""
    nl, nr = len(l_times), len(r_times)
    base = nr + 1
    best = None

    def consider(matching):
        nonlocal best
        left_seen: set[int] = set()
        right_seen: set[int] = set()
        usage: dict[tuple[str, str], int] = {}
        total = 0
        vector = [nr] * nl
        for i, j in matching:
            if i in left_seen or j in right_seen:
                return
            if abs(l_times[i] - r_times[j]) > tol:
                return
            key = (l_srcs[i], r_srcs[j])
            usage[key] = usage.get(key, 0) + 1
            if usage[key] > caps.get(key, float("inf")):
                return
            left_seen.add(i)
            right_seen.add(j)
            total += abs(l_times[i] - r_times[j])
            vector[i] = j
        tie = 0
        for digit in vector:
            tie = tie * base + digit
        score = (-len(matching), total, tie)
        if best is None or score < best[0]:
            best = (score, frozenset(matching))

    for size in range(0, min(nl, nr) + 1):
        for li in itertools.combinations(range(nl), size):
            for rj in itertools.combinations(range(nr), size):
                for perm in itertools.permutations(rj):
                    consider(list(zip(li, perm)))
    if best is None:
        raise AssertionError
    return best[1]


class AlignmentTests(unittest.TestCase):
    def test_identity_matching(self):
        res = align_events(
            _ids("a", 2), [0, 1000], _ids("x", 2), [0, 1000], 2000
        )
        self.assertEqual(
            {(p.left_id, p.right_id) for p in res.pairs},
            {("a0", "x0"), ("a1", "x1")},
        )
        self.assertEqual(res.total_abs_diff_us, 0)
        self.assertEqual(res.unmatched_left, [])
        self.assertEqual(res.unmatched_right, [])
        self.assertEqual(res.candidate_edge_count, 4)

    def test_global_optimum_under_crossing_temptation(self):
        # left a0=0, a1=100 ; right b0=40, b1=60 ; tol 100
        # crossing  a0-b1 (60) + a1-b0 (60) = 120
        # non-cross a0-b0 (40) + a1-b1 (40) = 80  <-- optimum
        res = align_events(
            ["a0", "a1"], [0, 100], ["b0", "b1"], [40, 60], 100
        )
        self.assertEqual(
            {(p.left_id, p.right_id) for p in res.pairs},
            {("a0", "b0"), ("a1", "b1")},
        )
        self.assertEqual(res.total_abs_diff_us, 80)

    def test_boundary_tolerance_inclusive(self):
        res = align_events(["a"], [0], ["b"], [1000], 1000)
        self.assertEqual(len(res.pairs), 1)
        self.assertEqual(res.pairs[0].abs_diff_us, 1000)
        res2 = align_events(["a"], [0], ["b"], [1001], 1000)
        self.assertEqual(res2.pairs, [])
        self.assertEqual(res2.unmatched_left, ["a"])
        self.assertEqual(res2.unmatched_right, ["b"])

    def test_unmatched_events_are_kept(self):
        res = align_events(
            ["a0", "a1", "a2"], [0, 1_000_000, 2_000_000],
            ["b0"], [1_000_050], 500,
        )
        self.assertEqual(
            {(p.left_id, p.right_id) for p in res.pairs}, {("a1", "b0")}
        )
        self.assertEqual(res.unmatched_left, ["a0", "a2"])
        self.assertEqual(res.unmatched_right, [])

    def test_cardinality_outranks_total_diff(self):
        # a0=0 can only reach b0(0); a1=100 reaches both b0 and b1(100).
        # A cost-only view "saves" nothing by dropping pairs, but assert
        # the service still forces the maximum number of pairs even when
        # the extra pair uses the whole tolerance budget.
        res = align_events(
            ["a0", "a1"], [0, 100], ["b0", "b1"], [0, 100], 100
        )
        self.assertEqual(len(res.pairs), 2)
        self.assertEqual(res.total_abs_diff_us, 0)

        # 2 left vs 1 right, second pair impossible: report both unmatched.
        res2 = align_events(
            ["a0", "a1"], [0, 5_000_000], ["b0"], [0], 100
        )
        self.assertEqual(len(res2.pairs), 1)
        self.assertEqual(res2.unmatched_left, ["a1"])

    def test_tie_break_strict_by_left_event_id(self):
        # Two perfect matchings, total diff 110 in both:
        #   M1: a0-b0(100) + a1-b1(10)   vector (b0,b1) = (0,1)
        #   M2: a0-b1(110) + a1-b0(0)    vector (b1,b0) = (1,0)
        # M1 is lexicographically smaller -> stable winner.
        def run(order_l, order_r):
            lt, rt = [0, 100], [100, 110]
            l_ids = [["a0", "a1"][i] for i in order_l]
            r_ids = [["b0", "b1"][j] for j in order_r]
            l_times = [lt[i] for i in order_l]
            r_times = [rt[j] for j in order_r]
            res = align_events(l_ids, l_times, r_ids, r_times, 110)
            return {(p.left_id, p.right_id) for p in res.pairs}

        expected = {("a0", "b0"), ("a1", "b1")}
        for ol in itertools.permutations(range(2)):
            for orr in itertools.permutations(range(2)):
                self.assertEqual(run(ol, orr), expected, (ol, orr))

    def test_tie_break_picks_smaller_left_id_when_only_one_can_match(self):
        # Both a0 and a1 occur at t=100, single right b0 at t=100.
        # Either matching has cost 0; smaller left ID wins.
        res = align_events(
            ["a1", "a0"], [100, 100], ["b0"], [100], 0
        )
        self.assertEqual(
            {(p.left_id, p.right_id) for p in res.pairs}, {("a0", "b0")}
        )
        self.assertEqual(res.unmatched_left, ["a1"])

    def test_tie_break_when_unmatched_row_set_differs(self):
        # Regression: two max-cardinality (3 pairs) matchings with equal
        # total diff 341, but they match DIFFERENT left rows, so the
        # unmatched-row set differs. The strict ID vector decides.
        lt = [198, 193, 198, 169]
        rt = [79, 28, 112]
        res = align_events(
            [f"a{i}" for i in range(4)], lt,
            [f"b{j}" for j in range(3)], rt, 150,
        )
        got = {(p.left_id, p.right_id) for p in res.pairs}
        # a0(198)->b0(79)=119, a1(193)->b2(112)=81, a3(169)->b1(28)=141
        self.assertEqual(got, {("a0", "b0"), ("a1", "b2"), ("a3", "b1")})
        self.assertEqual(res.total_abs_diff_us, 341)
        self.assertEqual(res.unmatched_left, ["a2"])

    def test_tie_break_single_column_prefers_smaller_id_on_equal_diff(self):
        # nl=2, nr=1; both rows at the same distance; smaller ID wins and
        # the larger one survives as unmatched.
        res = align_events(["a1", "a0"], [100, 100], ["b0"], [0], 100)
        self.assertEqual(
            {(p.left_id, p.right_id) for p in res.pairs}, {("a0", "b0")}
        )
        self.assertEqual(res.unmatched_left, ["a1"])

    def test_signed_diff_and_explanation_values(self):
        res = align_events(["a"], [1500], ["b"], [1000], 1000)
        p = res.pairs[0]
        self.assertEqual(p.diff_us, 500)
        self.assertEqual(p.abs_diff_us, 500)
        res2 = align_events(["a"], [1000], ["b"], [1500], 1000)
        self.assertEqual(res2.pairs[0].diff_us, -500)
        res3 = align_events(["a"], [1000], ["b"], [1000], 0)
        self.assertEqual(res3.pairs[0].diff_us, 0)

    def test_empty_groups(self):
        res = align_events([], [], [], [], 10)
        self.assertEqual(res.pairs, [])
        self.assertEqual(res.candidate_edge_count, 0)

    def test_no_edges(self):
        res = align_events(["a"], [0], ["b"], [10**9], 5)
        self.assertEqual(res.pairs, [])
        self.assertEqual(res.unmatched_left, ["a"])
        self.assertEqual(res.unmatched_right, ["b"])
        self.assertEqual(res.total_abs_diff_us, 0)

    def test_rectangular_more_rights(self):
        res = align_events(
            _ids("a", 1), [100], _ids("b", 3), [0, 100, 200], 100
        )
        self.assertEqual(len(res.pairs), 1)
        self.assertEqual(res.pairs[0].right_id, "b1")
        self.assertEqual(res.unmatched_right, ["b0", "b2"])

    def test_rectangular_more_lefts_no_edge(self):
        # 3 left, 1 right, no candidate: no finite perfect matching exists,
        # the solver must still return all events unmatched.
        res = align_events(
            _ids("a", 3), [0, 100, 200], ["b0"], [9_000_000], 100
        )
        self.assertEqual(res.pairs, [])
        self.assertEqual(res.unmatched_left, ["a0", "a1", "a2"])
        self.assertEqual(res.unmatched_right, ["b0"])

    def test_randomised_against_brute_force(self):
        rng = random.Random(20260922)
        checked = 0
        for _ in range(400):
            nl = rng.randint(0, 3)
            nr = rng.randint(0, 3)
            if nl == 0 and nr == 0:
                continue
            l_times = [rng.randint(0, 500) for _ in range(nl)]
            r_times = [rng.randint(0, 500) for _ in range(nr)]
            tol = rng.choice([0, 50, 100, 500])
            expected = _brute_force(l_times, r_times, tol)

            # Feed shuffled orders to prove input order independence.
            lo = list(range(nl))
            ro = list(range(nr))
            rng.shuffle(lo)
            rng.shuffle(ro)
            l_ids = [f"a{i}" for i in lo]
            r_ids = [f"b{j}" for j in ro]
            res = align_events(
                l_ids, [l_times[i] for i in lo],
                r_ids, [r_times[j] for j in ro],
                tol,
            )
            idx_l = {f"a{i}": i for i in range(nl)}
            idx_r = {f"b{j}": j for j in range(nr)}
            got = {
                (idx_l[p.left_id], idx_r[p.right_id]) for p in res.pairs
            }
            self.assertEqual(got, expected, (l_times, r_times, tol, got))
            checked += 1
        self.assertGreater(checked, 200)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            align_events(["a", "a"], [0, 1], ["b"], [0], 10)

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            align_events(["a"], [0], ["b"], [0], -1)


class CapacityTests(unittest.TestCase):
    """Source-pair capacity constraints on the alignment core."""

    def test_capacity_limits_source_pair(self):
        # 2 lefts (source A), 2 rights (source X), all simultaneous;
        # without a cap both pairs match, cap 1 keeps exactly one.
        res = align_events(
            ["l0", "l1"], [0, 0], ["r0", "r1"], [0, 0], 0,
            left_sources=["A", "A"], right_sources=["X", "X"],
            capacities={("A", "X"): 1},
        )
        self.assertEqual(len(res.pairs), 1)
        # ID tie-break: smallest left ID with smallest right ID.
        self.assertEqual(res.pairs[0].left_id, "l0")
        self.assertEqual(res.pairs[0].right_id, "r0")
        self.assertEqual(res.unmatched_left, ["l1"])
        self.assertEqual(res.unmatched_right, ["r1"])

    def test_capacity_zero_forbids_combo(self):
        # l0 can reach r0 (combo A/X) or r1 (combo A/Y); A/X is banned.
        res = align_events(
            ["l0"], [0], ["r0", "r1"], [0, 0], 0,
            left_sources=["A"], right_sources=["X", "Y"],
            capacities={("A", "X"): 0},
        )
        self.assertEqual(
            [(p.left_id, p.right_id) for p in res.pairs], [("l0", "r1")]
        )

    def test_undeclared_combo_unlimited(self):
        # Capacity declared for a combo with no events: no effect.
        res = align_events(
            ["l0", "l1"], [0, 0], ["r0", "r1"], [0, 0], 0,
            left_sources=["A", "A"], right_sources=["X", "X"],
            capacities={("A", "absent"): 1},
        )
        self.assertEqual(len(res.pairs), 2)

    def test_generous_capacity_matches_unconstrained_result(self):
        # Non-binding capacities must reproduce the legacy optimum.
        base = align_events(
            ["a0", "a1"], [0, 100], ["b0", "b1"], [40, 60], 100
        )
        capped = align_events(
            ["a0", "a1"], [0, 100], ["b0", "b1"], [40, 60], 100,
            left_sources=["A", "A"], right_sources=["X", "X"],
            capacities={("A", "X"): 2},
        )
        self.assertEqual(
            {(p.left_id, p.right_id) for p in base.pairs},
            {(p.left_id, p.right_id) for p in capped.pairs},
        )
        self.assertEqual(base.total_abs_diff_us, capped.total_abs_diff_us)

    def test_capacity_global_optimum_not_greedy(self):
        # l0(A,t=0), l1(A,t=100); r0(X,t=0), r1(X,t=100), r2(Y,t=100).
        # Unconstrained: l0-r0 + l1-r1 (total 0), but cap (A,X)=1 forbids
        # it. Global optimum: l0-r0 (A/X) + l1-r2 (A/Y), still total 0.
        res = align_events(
            ["l0", "l1"], [0, 100], ["r0", "r1", "r2"], [0, 100, 100], 0,
            left_sources=["A", "A"], right_sources=["X", "X", "Y"],
            capacities={("A", "X"): 1},
        )
        got = {(p.left_id, p.right_id) for p in res.pairs}
        self.assertEqual(got, {("l0", "r0"), ("l1", "r2")})
        self.assertEqual(res.total_abs_diff_us, 0)

    def test_capacity_cardinality_still_dominates_total_diff(self):
        # Edges: l0-r0 (500), l1-r0 (500), l1-r1 (0); cap (A,X)=1.
        # Two pairs with total 500 beat one pair with total 0.
        res = align_events(
            ["l0", "l1"], [0, 1000], ["r0", "r1"], [500, 1000], 500,
            left_sources=["A", "A"], right_sources=["X", "Y"],
            capacities={("A", "X"): 1},
        )
        got = {(p.left_id, p.right_id) for p in res.pairs}
        self.assertEqual(got, {("l0", "r0"), ("l1", "r1")})
        self.assertEqual(res.total_abs_diff_us, 500)

    def test_capacity_tie_break_stays_id_stable(self):
        # Two capacity-feasible 2-pair matchings with equal total diff;
        # the ID vector decides, regardless of input order.
        def run(order_l, order_r):
            lt, rt = [0, 100], [100, 110]
            l_ids = [["a0", "a1"][i] for i in order_l]
            r_ids = [["b0", "b1"][j] for j in order_r]
            res = align_events(
                l_ids, [lt[i] for i in order_l],
                r_ids, [rt[j] for j in order_r], 110,
                left_sources=["A"] * 2, right_sources=["X"] * 2,
                capacities={("A", "X"): 2},
            )
            return {(p.left_id, p.right_id) for p in res.pairs}

        expected = {("a0", "b0"), ("a1", "b1")}
        for ol in itertools.permutations(range(2)):
            for orr in itertools.permutations(range(2)):
                self.assertEqual(run(ol, orr), expected, (ol, orr))

    def test_capacity_requires_sources(self):
        with self.assertRaises(ValueError):
            align_events(["a"], [0], ["b"], [0], 0,
                         capacities={("A", "X"): 1})

    def test_negative_capacity_rejected(self):
        with self.assertRaises(ValueError):
            align_events(["a"], [0], ["b"], [0], 0,
                         left_sources=["A"], right_sources=["X"],
                         capacities={("A", "X"): -1})

    def test_randomised_capacitated_against_brute_force(self):
        rng = random.Random(20260926)
        checked = 0
        for _ in range(300):
            nl = rng.randint(0, 4)
            nr = rng.randint(0, 4)
            if nl == 0 and nr == 0:
                continue
            l_times = [rng.randint(0, 300) for _ in range(nl)]
            r_times = [rng.randint(0, 300) for _ in range(nr)]
            l_srcs = [rng.choice("AB") for _ in range(nl)]
            r_srcs = [rng.choice("XY") for _ in range(nr)]
            tol = rng.choice([0, 50, 150, 300])
            caps = {}
            for ls in "AB":
                for rs in "XY":
                    if rng.random() < 0.5:
                        caps[(ls, rs)] = rng.randint(0, 2)
            expected = _brute_force_capacitated(
                l_times, r_times, tol, l_srcs, r_srcs, caps
            )

            # Feed shuffled orders to prove input order independence.
            lo = list(range(nl))
            ro = list(range(nr))
            rng.shuffle(lo)
            rng.shuffle(ro)
            res = align_events(
                [f"a{i}" for i in lo], [l_times[i] for i in lo],
                [f"b{j}" for j in ro], [r_times[j] for j in ro],
                tol,
                left_sources=[l_srcs[i] for i in lo],
                right_sources=[r_srcs[j] for j in ro],
                capacities=caps or None,
            )
            idx_l = {f"a{i}": i for i in range(nl)}
            idx_r = {f"b{j}": j for j in range(nr)}
            got = {
                (idx_l[p.left_id], idx_r[p.right_id]) for p in res.pairs
            }
            self.assertEqual(got, expected, (l_times, r_times, tol, caps, got))
            checked += 1
        self.assertGreater(checked, 150)


class SourcePairStatTests(unittest.TestCase):
    def test_stats_with_capacity(self):
        stats = source_pair_stats(
            ["l0", "l1"], [0, 0], ["A", "A"],
            ["r0", "r1"], [0, 0], ["X", "X"],
            0, [("l0", "r0")], {("A", "X"): 1},
        )
        self.assertEqual(len(stats), 1)
        s = stats[0]
        self.assertEqual((s.left_source_key, s.right_source_key), ("A", "X"))
        self.assertEqual(s.capacity, 1)
        self.assertEqual(s.candidate_edges, 4)
        self.assertEqual(s.matched_pairs, 1)
        # 4 candidate edges, capacity 1 -> 3 denied by the capacity.
        self.assertEqual(s.rejected_edges, 3)

    def test_stats_unlimited_combo_has_zero_rejected(self):
        stats = source_pair_stats(
            ["l0"], [0], ["A"], ["r0"], [0], ["X"], 0, [("l0", "r0")], None
        )
        self.assertEqual(len(stats), 1)
        s = stats[0]
        self.assertIsNone(s.capacity)
        self.assertEqual(s.matched_pairs, 1)
        self.assertEqual(s.rejected_edges, 0)

    def test_stats_include_declared_combo_without_candidates(self):
        stats = source_pair_stats(
            ["l0"], [0], ["A"], ["r0"], [10**9], ["X"], 5, [],
            {("A", "X"): 2, ("cam", "gate"): 1},
        )
        by_key = {(s.left_source_key, s.right_source_key): s for s in stats}
        self.assertEqual(set(by_key), {("A", "X"), ("cam", "gate")})
        self.assertEqual(by_key[("A", "X")].candidate_edges, 0)
        self.assertEqual(by_key[("A", "X")].rejected_edges, 0)
        self.assertEqual(by_key[("cam", "gate")].matched_pairs, 0)

    def test_stats_sorted_by_source_keys(self):
        stats = source_pair_stats(
            ["l0", "l1"], [0, 0], ["B", "A"],
            ["r0"], [0], ["Y"],
            0, [("l0", "r0"), ("l1", "r0")][:1], None,
        )
        keys = [(s.left_source_key, s.right_source_key) for s in stats]
        self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()
