# pyganak — Python bindings for Ganak

`pyganak` provides Python bindings for [Ganak](https://github.com/meelgroup/ganak),
a high-performance **exact model counter** for CNF formulas.  It exposes two
classes:

- **`Counter`** — unweighted (and projected) model counting.  Returns an
  arbitrary-precision Python `int`.
- **`WeightedCounter`** — weighted model counting with per-literal float
  weights.  Uses MPFR internally; returns a Python `float`.

Arjun preprocessing is applied automatically before each count, just as the
command-line tool does.

## Installation

```bash
pip install pyganak
```

Pre-built wheels are available for Linux (x86-64, ARM64) and macOS (Apple
Silicon, Intel).

## Quick start

### Unweighted counting

```python
from pyganak import Counter

c = Counter()
c.add_clause([1, 2])      # x1 OR x2
c.add_clause([-1, 2])     # NOT x1 OR x2
print(c.count())          # → 2  (exact Python int)
```

### Weighted counting

```python
from pyganak import WeightedCounter

c = WeightedCounter()
c.add_clause([1, 2])           # x1 OR x2

# Set weights for both polarities of each variable.
c.set_lit_weight( 1, 0.3)     # P(x1 = True)  = 0.3
c.set_lit_weight(-1, 0.7)     # P(x1 = False) = 0.7
c.set_lit_weight( 2, 0.4)     # P(x2 = True)  = 0.4
c.set_lit_weight(-2, 0.6)     # P(x2 = False) = 0.6

# Models: (T,T)=0.12  (T,F)=0.18  (F,T)=0.28  → total=0.58
print(c.count())               # → 0.58  (Python float)
```

---

## `Counter` API

### `Counter(verbose=0, seed=0, use_tw=False, tw_max=20)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `int` | `0` | Verbosity level (0 = silent) |
| `seed`    | `int` | `0` | Random seed for the solver |
| `use_tw`  | `bool`| `False` | Attempt the FPT junction-tree DP on the incidence-graph tree decomposition before falling back to DPLL.  Only triggers when the projection set covers all variables.  The DP is double-precision real, so #SAT counts above 2<sup>53</sup> lose exact-integer precision via this path. |
| `tw_max`  | `int` | `20` | Maximum treewidth at which the DP is used; on instances with larger treewidth the DP is skipped and the standard pipeline runs. |

### `add_clause(clause)`

Add a single clause — iterable of nonzero signed integers (1-indexed
variables; positive = positive literal, negative = negated literal).

```python
c.add_clause([1, -2, 3])   # x1 OR NOT x2 OR x3
```

### `add_clauses(clauses)`

Add multiple clauses at once.

```python
c.add_clauses([[1, 2], [-1, 3]])
```

### `set_sampling_set(vars)`

Set the *projection set* (independent support).  Only the given variables
are counted; all others are existentially quantified out.  If never called,
all variables are included.

```python
c.add_clause([1, 2, 3])
c.set_sampling_set([1, 2])
print(c.count())   # → 4
```

### `count() → int`

Run Arjun preprocessing + Ganak and return the exact model count as a Python
`int`.  May only be called **once** per instance.

### `new_vars(n)` / `nof_vars() → int` / `nof_clauses() → int`

Declare extra variables, or query the current variable / clause count.

---

## `WeightedCounter` API

### `WeightedCounter(verbose=0, seed=0, prec=128, use_tw=False, tw_max=20)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `int` | `0` | Verbosity level |
| `seed`    | `int` | `0` | Random seed |
| `prec`    | `int` | `128` | MPFR precision in bits (≥ 2) |
| `use_tw`  | `bool`| `False` | Attempt the FPT junction-tree DP first (double precision; *ignores `prec`*). Triggers only when the projection set covers all variables. Per-literal weights are forwarded into the DP. |
| `tw_max`  | `int` | `20` | Treewidth cap for the DP; above this the DP is skipped and the MPFR DPLL pipeline runs. |

The internal arithmetic is performed with MPFR at `prec` bits of precision
(default 128 ≈ 38 significant decimal digits).  The final result is returned
as a Python `float` (double).

> **Floating-point caveat:** Because floating-point arithmetic is **not
> associative**, the result is a high-precision approximation rather than an
> exact value — the order in which partial sums are accumulated can affect the
> last few bits.  For exact rational weighted counting, use the command-line
> tool with `--mode 1` (GMP `mpq` rationals).

### `set_lit_weight(lit, weight)`

Set the weight of a literal.

- `lit` — nonzero signed integer (1-indexed; positive = positive literal).
- `weight` — Python `float` (double).  Stored internally as an MPFR value at
  the precision given to the constructor.

Always set weights for **both** a literal and its negation to get
well-defined weighted counting.  Literals whose weights are not set default
to weight `1.0`.

```python
c.set_lit_weight( 1, 0.3)   # weight of  x1 = 0.3
c.set_lit_weight(-1, 0.7)   # weight of ¬x1 = 0.7
```

### `add_clause(clause)` / `add_clauses(clauses)` / `set_sampling_set(vars)` / `new_vars(n)` / `nof_vars()` / `nof_clauses()`

Identical to the `Counter` versions.

### `count() → float`

Run Arjun preprocessing + Ganak weighted model counting.  Returns the
weighted count as a Python `float`.  The internal computation uses MPFR at
the configured precision.  May only be called **once** per instance.

---

## `SOPCounter` API

Rank-width FPT DP for *quadratic sum-of-powers* (the structured form that
arises from Clifford+T quantum simulation via the Feynman path integral).
The input is a graph (the interaction graph), not a CNF, so it has its own
class rather than being grafted onto `Counter` / `WeightedCounter`.

`SOPCounter` computes

$$Z = \sum_{x \in \{0,1\}^n} \omega_r^{f(x)}, \qquad f(x) = c + \sum_v b_v\, x_v + \tfrac{r}{2}\!\!\sum_{(u,v)\in E}\! x_u\, x_v \pmod{r}$$

via Algorithm 1 (Fourier-mode, *a = 1*) of "WMC for Quantum Simulation Also
Breaks the Treewidth Barrier".  Running time is `O(n · r · 4^k · poly(n))`
where `k` is the rank-width of the interaction graph.

### `SOPCounter(n, r=8, c=0, rw_max=12, verbose=0)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n`       | `int` | — | Number of SOP variables.  Internal adjacency uses a 64-bit mask per vertex, so `1 ≤ n ≤ 64`. |
| `r`       | `int` | `8` | Modulus (must be a positive even integer; `r = 8` matches the H + T + CZ gate set). |
| `c`       | `int` | `0` | Constant term in `f` (taken mod `r`). |
| `rw_max`  | `int` | `12` | Maximum rank-width at which `count()` will run; above this it raises `RuntimeError`. |
| `verbose` | `int` | `0` | Verbosity level. |

### `set_b(coefs)` / `set_b_v(v, b_v)`

Set per-vertex coefficients `b_v` (taken mod `r`).  `set_b` takes an iterable
of exactly `n` integers; `set_b_v` updates a single vertex.

### `add_edge(u, v)`

Add an undirected edge `{u, v}` to the interaction graph.  Both endpoints
must be in `[0, n)` and `u != v`.

### `rank_width() → int`

Return the rank-width of the current interaction graph (uses the same greedy
bipartition as `count()`).  Useful for deciding whether `count()` will fit
inside `rw_max` before paying for the DP.

### `count() → complex`

Run the Fourier-mode rank-width DP and return the amplitude `Z` as a Python
`complex`.  Raises `RuntimeError` if the rank-width exceeds `rw_max`.  May
only be called **once** per instance.

```python
from pyganak import SOPCounter

# qc_2qubit:  H⊗H · CZ · H⊗H — n=2, r=8, all b=0, edge (0,1) — Z = 2.
s = SOPCounter(n=2, r=8)
s.add_edge(0, 1)
print(s.count())          # (2+0j)

# grcs_2x2_5cycle:  4 T-gates, two layers of CZ.  Z = 0 + 3.6568542…i
g = SOPCounter(n=4, r=8)
g.set_b([1, 1, 1, 1])
for u, v in [(0, 1), (0, 2), (1, 3), (2, 3)]:
    g.add_edge(u, v)
print(g.count())          # ≈ 3.6568542j
```

---

## Examples

### Plain model counting

```python
from pyganak import Counter

# (x1 XOR x2): 2 models
c = Counter()
c.add_clause([1, 2])
c.add_clause([-1, -2])
print(c.count())   # 2
```

### Projected model counting

```python
from pyganak import Counter

# (x1 OR x2 OR x3) projected onto {x1, x2}: 4 models
c = Counter()
c.add_clause([1, 2, 3])
c.set_sampling_set([1, 2])
print(c.count())   # 4
```

### Large count (arbitrary precision)

```python
from pyganak import Counter

# 30 unconstrained variables → 2^30 models
c = Counter()
c.new_vars(30)
print(c.count())   # 1073741824
```

### Weighted counting

```python
from pyganak import WeightedCounter

# (x1 OR x2) with independent Bernoulli weights
c = WeightedCounter()
c.add_clause([1, 2])
c.set_lit_weight( 1, 0.3);  c.set_lit_weight(-1, 0.7)
c.set_lit_weight( 2, 0.4);  c.set_lit_weight(-2, 0.6)
print(c.count())   # 0.58
```

### Weighted counting with higher MPFR precision

```python
from pyganak import WeightedCounter

# Use 256-bit MPFR precision for the internal computation.
c = WeightedCounter(prec=256)
c.add_clause([1, 2, 3])
c.set_lit_weight( 1, 0.1);  c.set_lit_weight(-1, 0.9)
c.set_lit_weight( 2, 0.2);  c.set_lit_weight(-2, 0.8)
c.set_lit_weight( 3, 0.5);  c.set_lit_weight(-3, 0.5)
print(c.count())
```

### Treewidth dispatch (`use_tw`)

```python
from pyganak import Counter, WeightedCounter

# Same triangle CNF runs ~10× faster via the FPT DP on large structured
# instances; on this tiny one it just produces the same answer.
c = Counter(use_tw=True, tw_max=10)
c.add_clause([1, 2]); c.add_clause([2, 3]); c.add_clause([1, 3])
print(c.count())                  # 4

w = WeightedCounter(use_tw=True, tw_max=10)
w.add_clause([1, 2]); w.add_clause([2, 3]); w.add_clause([1, 3])
for v, (p, n) in [(1, (.8, .2)), (2, (.6, .4)), (3, (.7, .3))]:
    w.set_lit_weight( v, p)
    w.set_lit_weight(-v, n)
print(w.count())                  # 0.788 = 197/250
```

### Rank-width SOP counting

```python
from pyganak import SOPCounter

# A 4-vertex grid (rank-width 1) with all b=1 — equivalent CNF formulation
# of a small GRCS-style circuit.  Run with no rank-width cap.
s = SOPCounter(n=4, r=8, rw_max=12)
s.set_b([1, 1, 1, 1])
for u, v in [(0, 1), (0, 2), (1, 3), (2, 3)]:
    s.add_edge(u, v)
print(s.rank_width(), s.count())  # 1 (≈3.657j)
```

---

## Building from source (venv)

If you have already built Ganak with CMake (see the top-level `README.md`
for instructions), the extension is in `build/lib/pyganak*.so`.  Test it
without a full `pip install` using a venv:

```bash
python3 -m venv venv
venv/bin/pip install pytest
PYTHONPATH=build/lib venv/bin/pytest python/tests/ -v
```

To rebuild the extension after editing `python/src/pyganak.cpp`:

```bash
# If the build directory already exists:
cmake -DBUILD_PYTHON_EXTENSION=ON build   # or cmake .. from inside build/
cmake --build build --target pyganak -j$(nproc)

# Then re-run tests as above.
```

To do a full `pip install` (builds everything from scratch):

```bash
pip install .
```

Requires: GMP ≥ 5, MPFR ≥ 3, FLINT ≥ 2.  On Ubuntu/Debian:

```bash
sudo apt install libgmp-dev libmpfr-dev libflint-dev
```

On macOS:

```bash
brew install gmp mpfr flint
```

## License

MIT — see `LICENSE.txt` in the repository root.
