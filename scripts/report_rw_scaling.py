#!/usr/bin/env python3
"""Run only the rw_* benchmarks, repeat each timing several times, and emit
both a detailed markdown table and a matplotlib scaling plot (PNG).

Usage:
    python3 scripts/report_rw_scaling.py [--repeats 3] [--out rw_report]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from collections import defaultdict
from statistics import median

GANAK = "/home/gperez/GIT-repos/ganak/build/ganak"
BENCH_DIR = "/home/gperez/GIT-repos/ganak/tests/benchmarks"

_RE_FLOAT = re.compile(
    r"^c [os] exact quadruple float\s+"
    r"(?P<re>[-+]?\d+\.\d+e[-+]?\d+)"
    r"(?:\s*\+\s*(?P<im>[-+]?\d+\.\d+e[-+]?\d+)i)?",
    re.MULTILINE,
)
_RE_RW_LINE = re.compile(
    r"\[rw\] SOP variables: (?P<n>\d+), rank-width: (?P<rw>\d+)"
)


def parse_answer(out: str):
    m = _RE_FLOAT.search(out)
    if not m:
        return None
    re_v = float(m.group("re"))
    im_v = float(m.group("im")) if m.group("im") else 0.0
    return complex(re_v, im_v)


def parse_rw(out: str) -> int | None:
    m = _RE_RW_LINE.search(out)
    return int(m.group("rw")) if m else None


def parse_p_cnf(path: str) -> tuple[int, int]:
    with open(path) as f:
        for line in f:
            if line.startswith("p cnf"):
                parts = line.split()
                return int(parts[2]), int(parts[3])
    return 0, 0


def run(cnf: str, flags: list[str], timeout: float) -> tuple[float, str]:
    t0 = time.perf_counter()
    p = subprocess.run([GANAK] + flags + [cnf],
                       capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    return wall, p.stdout


def family_of(name: str) -> str:
    # rw_path_n040 -> path, rw_grid_3x10 -> grid, rw_rnd3_n050 -> rnd3, ...
    parts = name.split("_")
    return parts[1] if len(parts) >= 2 else "other"


def agreement(a, b, tol_rel=1e-6, tol_abs=1e-9) -> bool:
    if a is None or b is None:
        return False
    if abs(a - b) <= tol_abs:
        return True
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom <= tol_rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", default="rw_report",
                   help="Output basename (writes .md and .png)")
    p.add_argument("--bench", default=BENCH_DIR)
    args = p.parse_args()

    rw_files = sorted(f for f in os.listdir(args.bench)
                      if f.startswith("rw_") and f.endswith(".cnf"))

    rows = []
    print(f"Running {len(rw_files)} rw instances x {args.repeats} repeats...")
    for fname in rw_files:
        path = os.path.join(args.bench, fname)
        n_total, m_total = parse_p_cnf(path)

        # Structured (SOP-DP) – run repeats and take median.
        on_times, on_ans, n_sop, rw = [], None, None, None
        for _ in range(args.repeats):
            t, out = run(path, ["--mode", "6", "--rw", "1", "--rwmaxk", "20"],
                         timeout=args.timeout)
            on_times.append(t)
            ans = parse_answer(out)
            if ans is not None:
                on_ans = ans
            m_rw = _RE_RW_LINE.search(out)
            if m_rw:
                n_sop = int(m_rw.group("n"))
                rw = int(m_rw.group("rw"))

        # DPLL
        off_times, off_ans = [], None
        for _ in range(args.repeats):
            t, out = run(path, ["--mode", "6", "--rw", "0"], timeout=args.timeout)
            off_times.append(t)
            ans = parse_answer(out)
            if ans is not None:
                off_ans = ans

        on_med = median(on_times)
        off_med = median(off_times)
        ok = agreement(on_ans, off_ans)
        speedup = (off_med / on_med) if on_med > 0 else float("inf")
        rows.append({
            "name":     fname[:-4],
            "family":   family_of(fname[:-4]),
            "n_sop":    n_sop if n_sop is not None else 0,
            "n_total":  n_total,
            "m_total":  m_total,
            "rw":       rw if rw is not None else -1,
            "sop_t":    on_med,
            "dpll_t":   off_med,
            "speedup":  speedup,
            "agree":    ok,
        })
        print(f"  {fname[:-4]:<22} rw={rw}  SOP={on_med:.3f}s  "
              f"DPLL={off_med:.3f}s  {speedup:.1f}x  {'OK' if ok else 'MISMATCH'}")

    # ---- Markdown table ---------------------------------------------------
    md_path = args.out + ".md"
    with open(md_path, "w") as f:
        f.write("# Rank-width SOP-DP vs DPLL — paired benchmark scaling\n\n")
        f.write("Each row uses one CNF that carries both the `c rw_sop` / `c rw_edge` "
                "metadata (used by `--rw 1`) and an equivalent weighted CNF encoding "
                "(auxiliary `y_e = x_u ∧ x_v` clauses, `w(x_v=1)=omega_8^{b_v}`, "
                "`w(y_e=1)=-1`) for `--rw 0`. Both paths therefore compute the same "
                "complex amplitude.\n\n")
        f.write(f"Median of {args.repeats} runs per cell. Mode 6 (complex MPFR).\n\n")
        f.write("| family | instance | n (SOP) | total vars | clauses | rank-width "
                "| SOP-DP (s) | DPLL (s) | speedup | agree |\n")
        f.write("|--------|----------|--------:|-----------:|--------:|-----------:"
                "|-----------:|---------:|--------:|:-----:|\n")
        for r in rows:
            f.write(f"| {r['family']} | `{r['name']}` "
                    f"| {r['n_sop']} | {r['n_total']} | {r['m_total']} "
                    f"| {r['rw']} "
                    f"| {r['sop_t']:.3f} | {r['dpll_t']:.3f} "
                    f"| {r['speedup']:.1f}× | {'✓' if r['agree'] else '✗'} |\n")

        # Per-family summary
        f.write("\n## Per-family summary\n\n")
        f.write("| family | instances | median speedup | max speedup | "
                "max SOP-DP (s) | max DPLL (s) |\n")
        f.write("|--------|----------:|---------------:|------------:|"
                "---------------:|-------------:|\n")
        per_fam = defaultdict(list)
        for r in rows:
            per_fam[r["family"]].append(r)
        for fam, items in sorted(per_fam.items()):
            sp = [r["speedup"] for r in items]
            mxs = max(r["sop_t"] for r in items)
            mxd = max(r["dpll_t"] for r in items)
            f.write(f"| {fam} | {len(items)} | {median(sp):.1f}× | {max(sp):.1f}× "
                    f"| {mxs:.3f} | {mxd:.3f} |\n")

    print(f"\nWrote {md_path}")

    # ---- Matplotlib plot --------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    families = sorted(set(r["family"] for r in rows))
    colors = plt.cm.tab10.colors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, fam in enumerate(families):
        items = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["n_sop"])
        xs = [r["n_sop"] for r in items]
        ys_sop = [r["sop_t"] for r in items]
        ys_dpll = [r["dpll_t"] for r in items]
        c = colors[i % len(colors)]
        ax1.plot(xs, ys_sop, "o--", color=c, label=f"{fam}", markersize=5)
        ax1.plot(xs, ys_dpll, "s-",  color=c, markersize=5)

    ax1.set_xlabel("SOP variables n")
    ax1.set_ylabel("Wall time (s)")
    ax1.set_yscale("log")
    ax1.set_title("Wall time vs n  (dashed: SOP-DP, solid: DPLL)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best", fontsize=8, ncol=2)

    for i, fam in enumerate(families):
        items = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["n_sop"])
        xs = [r["n_sop"] for r in items]
        sp = [r["speedup"] for r in items]
        c = colors[i % len(colors)]
        ax2.plot(xs, sp, "o-", color=c, label=fam, markersize=6)

    ax2.set_xlabel("SOP variables n")
    ax2.set_ylabel("DPLL time / SOP-DP time")
    ax2.set_title("Speedup of SOP-DP over DPLL")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8, ncol=2)
    ax2.axhline(1.0, linestyle=":", color="gray", linewidth=1)

    fig.suptitle("Rank-width SOP-DP vs DPLL — paired Z₈ quadratic-SOP benchmarks",
                 y=1.02)
    fig.tight_layout()
    png_path = args.out + ".png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
