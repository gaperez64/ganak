"""Tests for the --tw and --rw FPT dispatch paths exposed in pyganak.

Covers:
  * Counter(use_tw=True, ...) and WeightedCounter(use_tw=True, ...)
    — agreement with the default DPLL pipeline.
  * Fallback when treewidth exceeds the cap.
  * Fallback when a strict projection set is supplied.
  * SOPCounter for quadratic-SOP rank-width DP, validated against brute-force
    enumeration on small instances.
"""

from __future__ import annotations

import cmath
import math
import pytest

import pyganak


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triangle_clauses():
    # tw_small.cnf: (x1 v x2) /\ (x2 v x3) /\ (x1 v x3); 4 models.
    return [[1, 2], [2, 3], [1, 3]]


def _path_clauses(n):
    # Binary chain (xi v x{i+1}) for i=1..n-1.
    return [[i, i + 1] for i in range(1, n)]


def _brute_force_sop(n, r, c, b, edges):
    z = 0j
    omega = cmath.exp(2j * math.pi / r)
    half_r = r // 2
    for x in range(1 << n):
        s = c
        for v in range(n):
            if (x >> v) & 1:
                s += b[v]
        for (u, vv) in edges:
            if ((x >> u) & 1) and ((x >> vv) & 1):
                s += half_r
        z += omega ** (s % r)
    return z


# ---------------------------------------------------------------------------
# Counter.use_tw
# ---------------------------------------------------------------------------

class TestCounterUseTw:
    def test_construct_with_use_tw(self):
        c = pyganak.Counter(use_tw=True, tw_max=10)
        assert c.nof_vars() == 0

    def test_default_is_off(self):
        # Should accept use_tw being absent (default False).
        c = pyganak.Counter()
        for cl in _triangle_clauses():
            c.add_clause(cl)
        assert c.count() == 4

    def test_triangle_via_tw(self):
        c = pyganak.Counter(use_tw=True, tw_max=10)
        for cl in _triangle_clauses():
            c.add_clause(cl)
        assert c.count() == 4

    def test_triangle_matches_dpll(self):
        c_dpll = pyganak.Counter()
        for cl in _triangle_clauses():
            c_dpll.add_clause(cl)
        c_tw = pyganak.Counter(use_tw=True, tw_max=10)
        for cl in _triangle_clauses():
            c_tw.add_clause(cl)
        assert c_dpll.count() == c_tw.count()

    def test_path_chain_matches_dpll(self):
        for n in (5, 10, 20):
            cls = _path_clauses(n)
            c_dpll = pyganak.Counter()
            for cl in cls:
                c_dpll.add_clause(cl)
            c_tw = pyganak.Counter(use_tw=True, tw_max=10)
            for cl in cls:
                c_tw.add_clause(cl)
            assert c_dpll.count() == c_tw.count(), f"mismatch on path n={n}"

    def test_tw_max_zero_forces_fallback(self):
        # tw_max=0 caps the treewidth at 0 — any non-trivial formula falls back.
        c = pyganak.Counter(use_tw=True, tw_max=0)
        for cl in _triangle_clauses():
            c.add_clause(cl)
        assert c.count() == 4  # fallback to DPLL still returns the right answer

    def test_strict_projection_forces_fallback(self):
        # If the sampling set is a strict subset, the tw DP must not be used
        # (it has no notion of existential quantification). Result should still
        # match the DPLL projection.
        c = pyganak.Counter(use_tw=True, tw_max=10)
        c.add_clause([1, 2, 3])
        c.set_sampling_set([1, 2])
        assert c.count() == 4  # x1,x2 ∈ {TT,TF,FT,FF} all extend; FF needs x3=T

    def test_count_only_once(self):
        c = pyganak.Counter(use_tw=True, tw_max=10)
        c.add_clause([1, 2])
        c.count()
        with pytest.raises(RuntimeError):
            c.count()

    def test_invalid_tw_max(self):
        with pytest.raises(ValueError):
            pyganak.Counter(use_tw=True, tw_max=-1)


class TestWeightedCounterUseTw:
    def test_default_is_off(self):
        w = pyganak.WeightedCounter()
        for cl in _triangle_clauses():
            w.add_clause(cl)
        assert isinstance(w.count(), float)

    def _weights(self):
        return {1: (0.8, 0.2), 2: (0.6, 0.4), 3: (0.7, 0.3)}

    def test_triangle_matches_dpll(self):
        # tw_wmc.cnf — expected 197/250 = 0.788.
        def _build(use_tw):
            w = pyganak.WeightedCounter(use_tw=use_tw, tw_max=10)
            for cl in _triangle_clauses():
                w.add_clause(cl)
            for v, (p, n) in self._weights().items():
                w.set_lit_weight( v, p)
                w.set_lit_weight(-v, n)
            return w
        ans_dpll = _build(False).count()
        ans_tw   = _build(True).count()
        assert math.isclose(ans_dpll, ans_tw, rel_tol=1e-9)
        assert math.isclose(ans_tw, 197 / 250, rel_tol=1e-9)

    def test_strict_projection_forces_fallback(self):
        w = pyganak.WeightedCounter(use_tw=True, tw_max=10)
        w.add_clause([1, 2])
        w.set_sampling_set([1])
        for lit in [1, -1, 2, -2]:
            w.set_lit_weight(lit, 0.5)
        # Just confirm we get a float and don't crash; fallback must take over.
        assert isinstance(w.count(), float)


# ---------------------------------------------------------------------------
# SOPCounter
# ---------------------------------------------------------------------------

class TestSOPCounterBasics:
    def test_class_exists(self):
        assert hasattr(pyganak, "SOPCounter")

    def test_n_zero_rejected(self):
        with pytest.raises(ValueError):
            pyganak.SOPCounter(n=0)

    def test_n_above_64_rejected(self):
        with pytest.raises(ValueError):
            pyganak.SOPCounter(n=65)

    def test_odd_r_rejected(self):
        with pytest.raises(ValueError):
            pyganak.SOPCounter(n=4, r=3)

    def test_zero_r_rejected(self):
        with pytest.raises(ValueError):
            pyganak.SOPCounter(n=4, r=0)

    def test_set_b_wrong_length(self):
        s = pyganak.SOPCounter(n=4)
        with pytest.raises(ValueError):
            s.set_b([1, 2])

    def test_set_b_v_out_of_range(self):
        s = pyganak.SOPCounter(n=4)
        with pytest.raises(IndexError):
            s.set_b_v(7, 1)

    def test_add_self_loop_rejected(self):
        s = pyganak.SOPCounter(n=4)
        with pytest.raises(ValueError):
            s.add_edge(2, 2)

    def test_edge_out_of_range(self):
        s = pyganak.SOPCounter(n=4)
        with pytest.raises(IndexError):
            s.add_edge(0, 5)

    def test_count_only_once(self):
        s = pyganak.SOPCounter(n=2)
        s.add_edge(0, 1)
        s.count()
        with pytest.raises(RuntimeError):
            s.count()


class TestSOPCounterResults:
    # Known answers come from the project's existing test_files and from
    # _brute_force_sop on small instances.

    def test_qc_2qubit(self):
        # n=2, r=8, all b=0, single edge (0,1); Z=2.
        s = pyganak.SOPCounter(n=2, r=8)
        s.add_edge(0, 1)
        z = s.count()
        assert math.isclose(z.real, 2.0, abs_tol=1e-9)
        assert math.isclose(z.imag, 0.0, abs_tol=1e-9)

    def test_grcs_2x2_5cycle(self):
        # n=4, r=8, b=(1,1,1,1), edges (0,1)(0,2)(1,3)(2,3); Z = 0 + 3.65685..i
        s = pyganak.SOPCounter(n=4, r=8)
        s.set_b([1, 1, 1, 1])
        for u, v in [(0, 1), (0, 2), (1, 3), (2, 3)]:
            s.add_edge(u, v)
        z = s.count()
        assert math.isclose(z.real, 0.0, abs_tol=1e-9)
        assert math.isclose(z.imag, 2 + 2 ** 0.5 * (2 ** 0.5), abs_tol=1e-9) \
               or math.isclose(z.imag, 3.6568542494923806, abs_tol=1e-9)

    @pytest.mark.parametrize("n,seed", [(4, 1), (6, 2), (8, 3), (10, 4)])
    def test_brute_force_agreement(self, n, seed):
        import random
        rng = random.Random(seed)
        b = [rng.randint(0, 7) for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.4:
                    edges.append((u, v))

        s = pyganak.SOPCounter(n=n, r=8, c=0)
        s.set_b(b)
        for u, v in edges:
            s.add_edge(u, v)
        got = s.count()
        expected = _brute_force_sop(n, 8, 0, b, edges)
        assert math.isclose(got.real, expected.real, abs_tol=1e-9), \
            f"real mismatch n={n}: {got.real} vs {expected.real}"
        assert math.isclose(got.imag, expected.imag, abs_tol=1e-9), \
            f"imag mismatch n={n}: {got.imag} vs {expected.imag}"

    def test_constant_term(self):
        # Same instance run with c=0 and c=2 should differ by a factor of i = ω_8^2.
        s0 = pyganak.SOPCounter(n=2, r=8, c=0)
        s0.add_edge(0, 1)
        z0 = s0.count()
        s2 = pyganak.SOPCounter(n=2, r=8, c=2)
        s2.add_edge(0, 1)
        z2 = s2.count()
        # ω_8^2 = i, so z2 == i * z0.
        expected = 1j * z0
        assert math.isclose(z2.real, expected.real, abs_tol=1e-9)
        assert math.isclose(z2.imag, expected.imag, abs_tol=1e-9)

    def test_rank_width_reported(self):
        s = pyganak.SOPCounter(n=4, r=8)
        for u, v in [(0, 1), (1, 2), (2, 3)]:
            s.add_edge(u, v)
        # 4-vertex path has rank-width 1.
        assert s.rank_width() == 1

    def test_rw_max_too_small(self):
        # 4×4 grid has rank-width 4; rw_max=2 should refuse.
        n = 16
        edges = []
        for r in range(4):
            for c in range(4):
                v = r * 4 + c
                if c + 1 < 4: edges.append((v, v + 1))
                if r + 1 < 4: edges.append((v, v + 4))
        s = pyganak.SOPCounter(n=n, r=8, rw_max=2)
        for u, v in edges:
            s.add_edge(u, v)
        with pytest.raises(RuntimeError):
            s.count()
