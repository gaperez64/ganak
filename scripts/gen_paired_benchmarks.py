#!/usr/bin/env python3
"""Generate paired benchmark CNFs for the --tw and --rw structured-dispatch
paths in ganak.

Treewidth benchmarks
====================
Output: weighted CNFs over structured graphs (paths, cycles, ladders, grids,
trees, random sparse). The same file is run with --tw 1 (incidence-graph DP)
and --tw 0 (DPLL). Comparison is apples-to-apples: both paths compute the
same WMC value.

Each variable v gets a positive weight 0.55 + 0.4*sin(v) and a negative
weight 1 - that_value (kept in [0.15, 0.95]). Clauses are arity-2 OR-of-
literals on the graph's edges; polarities pseudo-random per edge.

Rank-width benchmarks
=====================
Output: SOP-style CNFs over structured graphs with r=8 (Z_8). The file
carries BOTH (a) the `c rw_sop` / `c rw_edge` metadata used by --rw 1, and
(b) an equivalent weighted CNF encoding (auxiliary y_e = x_u AND x_v) for
--rw 0 (= DPLL). Both paths compute the same complex amplitude.

For each instance:
  Z = sum_{x in {0,1}^n} omega_8^{sum_v b_v x_v + 4 * sum_{(u,v) in E} x_u x_v}

WMC encoding for --rw 0:
  Variables: x_1, ..., x_n, y_1, ..., y_m  (m = number of edges)
  Clauses (for each edge e=(u,v)):
    (~y_e | x_u)          # y_e -> x_u
    (~y_e | x_v)          # y_e -> x_v
    (y_e | ~x_u | ~x_v)   # x_u AND x_v -> y_e
  Weights:
    w(x_v = 1) = omega_8^{b_v},   w(x_v = 0) = 1
    w(y_e = 1) = omega_8^4 = -1,  w(y_e = 0) = 1

Run mode 6 (complex MPFR). Constant c is taken as 0 to avoid an
extra scaling variable.
"""

from __future__ import annotations

import argparse
import cmath
import math
import os
import random
import sys


# --------------------------------------------------------------------------- #
# Graph families
# --------------------------------------------------------------------------- #

def path_edges(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def cycle_edges(n: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def ladder_edges(k: int) -> tuple[int, list[tuple[int, int]]]:
    # 2 x k ladder.  Variables: top row 0..k-1, bottom row k..2k-1.
    n = 2 * k
    edges: list[tuple[int, int]] = []
    for i in range(k - 1):
        edges.append((i, i + 1))                    # top
        edges.append((k + i, k + i + 1))            # bottom
    for i in range(k):
        edges.append((i, k + i))                    # rungs
    return n, edges


def grid_edges(rows: int, cols: int) -> tuple[int, list[tuple[int, int]]]:
    n = rows * cols
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    return n, edges


def tree_edges(n: int, rng: random.Random) -> list[tuple[int, int]]:
    return [(rng.randint(0, i - 1), i) for i in range(1, n)]


def random_sparse_edges(n: int, deg: int,
                        rng: random.Random) -> list[tuple[int, int]]:
    """Roughly-deg-regular random graph via random edges (no self loops, no dup)."""
    edges: set[tuple[int, int]] = set()
    target = n * deg // 2
    tries = 0
    while len(edges) < target and tries < target * 20:
        u, v = rng.randint(0, n - 1), rng.randint(0, n - 1)
        if u == v:
            tries += 1
            continue
        edges.add((min(u, v), max(u, v)))
        tries += 1
    return sorted(edges)


# --------------------------------------------------------------------------- #
# Treewidth benchmark generation
# --------------------------------------------------------------------------- #

def _tw_weight(v: int) -> tuple[float, float]:
    """Deterministic per-variable weight in (0.15, 0.95)."""
    w1 = 0.55 + 0.4 * math.sin(v * 0.9 + 0.3)
    w1 = max(0.15, min(0.95, w1))
    return w1, 1.0 - w1


def write_tw_cnf(path: str, n: int, edges: list[tuple[int, int]],
                 seed: int, *, mode: int = 1, twmaxk: int = 30,
                 family_label: str = "") -> None:
    rng = random.Random(seed)
    # Clauses: for each edge, an OR-of-literals with random polarity.
    clauses: list[tuple[int, int]] = []
    for (u, v) in edges:
        l1 = (u + 1) if rng.random() < 0.5 else -(u + 1)
        l2 = (v + 1) if rng.random() < 0.5 else -(v + 1)
        clauses.append((l1, l2))

    with open(path, "w") as f:
        f.write(f"c Treewidth benchmark — {family_label}, n={n}, m={len(clauses)}, seed={seed}\n")
        f.write(f"c c RUN: %solver --mode {mode} --tw 1 --twmaxk {twmaxk} %s | %OutputCheck %s\n")
        f.write(f"p cnf {n} {len(clauses)}\n")
        f.write("c t wmc\n")
        for (l1, l2) in clauses:
            f.write(f"{l1} {l2} 0\n")
        for v in range(n):
            w_pos, w_neg = _tw_weight(v)
            f.write(f"c p weight {v+1} {w_pos:.10f} 0\n")
            f.write(f"c p weight -{v+1} {w_neg:.10f} 0\n")


# --------------------------------------------------------------------------- #
# Rank-width benchmark generation
# --------------------------------------------------------------------------- #

R_MOD = 8           # working in Z_8 (H + T + CZ gate set)
OMEGA = cmath.exp(2j * math.pi / R_MOD)
SQRT2_FRAC = "6369051672525773/9007199254740992"  # IEEE-double sqrt(2)/2


def omega_power(k: int) -> complex:
    return OMEGA ** (k % R_MOD)


def omega_power_exact_str(k: int) -> str:
    """Emit omega_8^k as an exact rational complex weight string.

    omega_8 = sqrt(2)/2 + i sqrt(2)/2. Powers 0..7 are well-known
    +/-1, +/-i, +/-sqrt(2)/2 +/- i sqrt(2)/2. Mode 6 uses MPFR floats
    so we emit the float representation; mode 1 uses rationals.

    Format must be parseable by parse_complex_mpq: avoid the
    "real -imag" pitfall (parser misreads "-" as "minus imag i").
    Emit "real + imag i" or "real - |imag| i" when imag is nonzero.
    """
    k = k % R_MOD
    # Exact powers of omega_8.
    table = {
        0: ("1", "0"),
        1: (SQRT2_FRAC, SQRT2_FRAC),
        2: ("0", "1"),
        3: ("-" + SQRT2_FRAC, SQRT2_FRAC),
        4: ("-1", "0"),
        5: ("-" + SQRT2_FRAC, "-" + SQRT2_FRAC),
        6: ("0", "-1"),
        7: (SQRT2_FRAC, "-" + SQRT2_FRAC),
    }
    re, im = table[k]
    if im == "0":
        return f"{re} 0"
    if im.startswith("-"):
        return f"{re} - {im[1:]}i"
    return f"{re} + {im}i"


def write_rw_cnf(path: str, n: int, edges: list[tuple[int, int]],
                 b: list[int], seed: int, *,
                 mode: int = 6, rwmaxk: int = 20,
                 family_label: str = "") -> None:
    # SOP variables are 1..n (x_v). Auxiliary y_e variables follow as n+1..n+m.
    m = len(edges)
    nvars_total = n + m

    clauses: list[tuple[int, ...]] = []
    for e_idx, (u, v) in enumerate(edges):
        xu = u + 1
        xv = v + 1
        ye = n + 1 + e_idx
        clauses.append((-ye, xu))
        clauses.append((-ye, xv))
        clauses.append((ye, -xu, -xv))

    with open(path, "w") as f:
        f.write(f"c Rank-width benchmark — {family_label}, n={n}, edges={m}, seed={seed}\n")
        f.write(f"c c RUN: %solver --mode {mode} --rw 1 --rwmaxk {rwmaxk} %s | %OutputCheck %s\n")
        f.write(f"p cnf {nvars_total} {len(clauses)}\n")
        f.write("c t wmc\n")

        # SOP metadata (for --rw 1).
        f.write(f"c rw_sop {n} {R_MOD} 0 " + " ".join(str(bi) for bi in b) + "\n")
        for (u, v) in edges:
            f.write(f"c rw_edge {u} {v}\n")

        # Clauses for the WMC encoding (for --rw 0).
        for cl in clauses:
            f.write(" ".join(str(l) for l in cl) + " 0\n")

        # Per-x_v weights: w(x_v=1) = omega_8^{b_v}, w(x_v=0) = 1.
        for v in range(n):
            f.write(f"c p weight {v+1} {omega_power_exact_str(b[v])} 0\n")
            f.write(f"c p weight -{v+1} 1 0 0\n")

        # Per-y_e weights: w(y_e=1) = omega_8^4 = -1, w(y_e=0) = 1.
        for e_idx in range(m):
            ye = n + 1 + e_idx
            f.write(f"c p weight {ye} -1 0 0\n")
            f.write(f"c p weight -{ye} 1 0 0\n")


# --------------------------------------------------------------------------- #
# Reference computation (brute force) for small instances
# --------------------------------------------------------------------------- #

def brute_force_sop(n: int, edges: list[tuple[int, int]],
                    b: list[int]) -> complex:
    z = 0j
    for x in range(1 << n):
        s = 0
        for v in range(n):
            if (x >> v) & 1:
                s += b[v]
        for (u, vv) in edges:
            if ((x >> u) & 1) and ((x >> vv) & 1):
                s += R_MOD // 2
        z += omega_power(s)
    return z


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/benchmarks",
                   help="Output directory (created if missing)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify", action="store_true",
                   help="Brute-force verify SOP instances (n<=18)")
    args = p.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- Treewidth families ------------------------------------------------
    tw_specs: list[tuple[str, int, list[tuple[int, int]]]] = []

    for n in [40, 160, 640, 2560]:
        tw_specs.append((f"tw_path_n{n:05d}", n, path_edges(n)))
    for n in [40, 160, 640, 2560]:
        tw_specs.append((f"tw_cycle_n{n:05d}", n, cycle_edges(n)))
    for k in [20, 80, 320, 1280]:
        nn, ee = ladder_edges(k)
        tw_specs.append((f"tw_ladder_k{k:04d}", nn, ee))
    for rows, cols in [(3, 16), (4, 24), (5, 40), (6, 60), (8, 80)]:
        nn, ee = grid_edges(rows, cols)
        tw_specs.append((f"tw_grid_{rows}x{cols:03d}", nn, ee))
    for n in [80, 320, 1280, 5120]:
        tw_specs.append((f"tw_tree_n{n:05d}", n, tree_edges(n, rng)))

    for name, n, edges in tw_specs:
        path = os.path.join(out_dir, name + ".cnf")
        write_tw_cnf(path, n, edges, seed=hash(name) & 0xFFFFFFFF,
                     family_label=name)
        print(f"  wrote {path} (n={n}, m={len(edges)})")

    # ---- Rank-width families -----------------------------------------------
    rw_specs: list[tuple[str, int, list[tuple[int, int]]]] = []

    for n in [10, 20, 30, 40, 50, 60]:
        rw_specs.append((f"rw_path_n{n:03d}", n, path_edges(n)))
    for n in [10, 20, 30, 40, 50, 60]:
        rw_specs.append((f"rw_cycle_n{n:03d}", n, cycle_edges(n)))
    for rows, cols in [(2, 5), (2, 10), (2, 20), (2, 30),
                       (3, 6), (3, 10), (3, 15), (3, 20),
                       (4, 8), (4, 12), (4, 15),
                       (5, 8), (5, 12),
                       (6, 10)]:
        nn, ee = grid_edges(rows, cols)
        rw_specs.append((f"rw_grid_{rows}x{cols:02d}", nn, ee))
    for n in [16, 32, 48, 60]:
        rw_specs.append((f"rw_tree_n{n:03d}", n, tree_edges(n, rng)))
    # Random sparse graphs (avg degree 3) — typically have higher rank-width.
    for n in [20, 30, 40, 50, 60]:
        rw_specs.append((f"rw_rnd3_n{n:03d}", n, random_sparse_edges(n, 3, rng)))

    for name, n, edges in rw_specs:
        if n > 60:
            print(f"  skip {name}: n>60 (SOP DP requires n<=64 and adj fits in 64-bit mask)")
            continue
        b = [rng.randint(0, R_MOD - 1) for _ in range(n)]
        path = os.path.join(out_dir, name + ".cnf")
        write_rw_cnf(path, n, edges, b, seed=hash(name) & 0xFFFFFFFF,
                     family_label=name)
        ref = ""
        if args.verify and n <= 18:
            z = brute_force_sop(n, edges, b)
            ref = f"  Z={z.real:+.6e}{z.imag:+.6e}i"
        print(f"  wrote {path} (n={n}, m={len(edges)}){ref}")


if __name__ == "__main__":
    main()
