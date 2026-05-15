#!/usr/bin/env python3
"""Convert a GRCS quantum circuit file to ganak SOP CNF format.

GRCS format (from https://github.com/sboixo/GRCS, cz_v2 directory):
  Line 1:  number of qubits
  Remaining lines:  <cycle> <gate> <qubit> [<qubit2>]

Supported gates: h, cz, t, t_dag, s, s_dag, z
  (x_1_2 / y_1_2 are non-diagonal and will cause an error)

The SOP encoding follows the Feynman path integral representation:
  f(x) = c + sum_v b_v x_v + (r/2) sum_{(u,v) in E} x_u x_v  mod r
with r=8, and Z = sum_{x in {0,1}^n} omega_r^{f(x)}.

Variables x_{a,j} represent qubit a at Hadamard depth j.
Interior variables (1 <= j < k_a) are free; boundary variables are pinned
to 0 (|0^n> -> |0^n> amplitude).

Usage:
  python3 grcs_to_sop.py circuit.txt [--mode N] [--verify]
  python3 grcs_to_sop.py circuit.txt > circuit.cnf

Options:
  --mode N   Output mode: 1=rational (real WMC), 6=complex MPFR (default: 6)
  --verify   Brute-force verify the WMC for small instances (n <= 20)
"""

import sys
import math
import cmath
import argparse

PHASE_GATES = {'t': 1, 't_dag': 7, 's': 2, 's_dag': 6, 'z': 4}


def parse_grcs(content):
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    n_qubits = int(lines[0])
    gates = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        cycle, gate = int(parts[0]), parts[1].lower()
        qubits = [int(p) for p in parts[2:]]
        gates.append((cycle, gate, qubits))
    return n_qubits, gates


def grcs_to_sop(n_qubits, gates):
    """Convert parsed GRCS gates to a SOPInstance dict.

    Returns (sop_dict, error_string_or_None).
    sop_dict has keys: n, r, c, b, edges
    """
    r = 8
    h_depth = [0] * n_qubits
    var_id = {}
    b_map = {}
    edges = set()
    c = 0
    unsupported = []

    def mkvar(a, j):
        key = (a, j)
        if key not in var_id:
            vid = len(var_id)
            var_id[key] = vid
            b_map[vid] = 0
        return var_id[key]

    for cycle, gate, qubits in gates:
        if gate == 'h':
            h_depth[qubits[0]] += 1
        elif gate == 'cz':
            a, b = qubits[0], qubits[1]
            ja, jb = h_depth[a], h_depth[b]
            va = mkvar(a, ja) if ja > 0 else None
            vb = mkvar(b, jb) if jb > 0 else None
            if va is not None and vb is not None:
                edges.add((min(va, vb), max(va, vb)))
            # Boundary-pinned qubits contribute 0 phase for |0> input/output
        elif gate in PHASE_GATES:
            a = qubits[0]
            ja = h_depth[a]
            if ja > 0:
                va = mkvar(a, ja)
                b_map[va] = (b_map[va] + PHASE_GATES[gate]) % r
        else:
            unsupported.append(gate)

    if unsupported:
        return None, f"Unsupported gate(s): {sorted(set(unsupported))}"

    # Keep only interior variables: depth 1 .. h_depth[a]-1
    free = {vid for (a, j), vid in var_id.items()
            if j != 0 and j != h_depth[a]}

    if not free:
        return {'n': 0, 'r': r, 'c': c, 'b': [], 'edges': []}, None

    otn = {vid: i for i, vid in enumerate(sorted(free))}
    n = len(otn)

    if n > 64:
        return None, f"Too many SOP variables ({n} > 64); bitmask DP requires n <= 64"

    b_list = [0] * n
    for vid, nid in otn.items():
        b_list[nid] = b_map[vid]

    new_edges = sorted({(otn[u], otn[v]) for u, v in edges
                        if u in otn and v in otn})
    return {'n': n, 'r': r, 'c': c, 'b': b_list, 'edges': new_edges}, None


def sop_to_cnf(sop, mode=6, circuit_name="", comment=""):
    """Render a SOPInstance as ganak CNF text (with rw_sop/rw_edge metadata)."""
    n, r, c, b, edges = sop['n'], sop['r'], sop['c'], sop['b'], sop['edges']
    lines = []
    if circuit_name:
        lines.append(f"c Circuit: {circuit_name}")
    if comment:
        lines.append(f"c {comment}")
    lines.append(f"c c RUN: %solver --mode {mode} --rw 1 %s | %OutputCheck %s")
    lines.append(f"p cnf {n} 0")
    lines.append("c t wmc")
    b_str = " ".join(str(bv) for bv in b)
    lines.append(f"c rw_sop {n} {r} {c} {b_str}")
    for u, v in edges:
        lines.append(f"c rw_edge {u} {v}")
    return "\n".join(lines) + "\n"


def brute_force_wmc(sop):
    n, r, c, b, edges = sop['n'], sop['r'], sop['c'], sop['b'], sop['edges']
    total = complex(0)
    for x in range(1 << n):
        exp = c
        for v in range(n):
            if (x >> v) & 1:
                exp += b[v]
        for u, v in edges:
            if ((x >> u) & 1) and ((x >> v) & 1):
                exp += r // 2
        total += cmath.exp(2j * math.pi * exp / r)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("circuit", help="GRCS circuit file (.txt)")
    parser.add_argument("--mode", type=int, default=6,
                        help="Ganak mode: 1=rational/real, 6=complex MPFR (default: 6)")
    parser.add_argument("--verify", action="store_true",
                        help="Brute-force verify WMC (only feasible for n <= 20)")
    parser.add_argument("--output", default="-",
                        help="Output CNF file (default: stdout)")
    args = parser.parse_args()

    with open(args.circuit) as f:
        content = f.read()

    n_qubits, gates = parse_grcs(content)
    sop, err = grcs_to_sop(n_qubits, gates)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    import os
    circuit_name = os.path.basename(args.circuit)
    cnf = sop_to_cnf(sop, mode=args.mode, circuit_name=circuit_name)

    if args.output == "-":
        print(cnf, end="")
    else:
        with open(args.output, "w") as f:
            f.write(cnf)

    if args.verify:
        if sop['n'] > 20:
            print(f"[verify] n={sop['n']} > 20, skipping brute force", file=sys.stderr)
        else:
            z = brute_force_wmc(sop)
            print(f"[verify] WMC = {z.real:.8e} + {z.imag:.8e}i", file=sys.stderr)

    print(f"[info] n={sop['n']}, |E|={len(sop['edges'])}, r={sop['r']}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
