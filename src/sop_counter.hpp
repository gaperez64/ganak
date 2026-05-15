/******************************************
Copyright (C) 2023 Authors of GANAK, see AUTHORS file

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
***********************************************/

#pragma once

#include <complex>
#include <cstdint>
#include <vector>
#include "rw_decomp.hpp"

namespace GanakInt {

// Quadratic sum-of-powers (SOP) instance from a quantum circuit.
// The SOP sum is Z = sum_{x in {0,1}^n} omega_r^{f(x)} where
// f(x) = c + sum_v b[v]*x_v + (r/2) * sum_{(u,v) in E} x_u*x_v  (mod r).
struct SOPInstance {
    uint32_t n = 0;            // number of SOP variables (graph vertices)
    uint32_t r = 8;            // modulus (even; r=8 for H/T/CZ gate set)
    int32_t  c = 0;            // global constant term in f(x)
    std::vector<int32_t> b;    // per-vertex coefficients b[v] in Z_r (size n)
    std::vector<uint64_t> adj; // adjacency bitmasks (size n): adj[i] bit j set iff edge {i,j}
    uint64_t all_vars_mask = 0; // (1<<n)-1
};

// Compute Z = A^(1)_root[empty_boundary] via Fourier-mode rank-width DP
// (Algorithm 1 of "WMC for Quantum Simulation Also Breaks Treewidth Barrier",
// Fourier variant with mode a=1). Returns the SOP sum as a complex number.
std::complex<double> sop_count(const SOPInstance& sop, const RankDecomp& decomp);

} // namespace GanakInt
