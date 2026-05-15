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

#include "sop_counter.hpp"
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <cassert>
#include <numeric>

namespace GanakInt {

// =============================================================================
// Rank decomposition construction
// =============================================================================

// Greedy bipartition of vars_mask into two non-empty halves minimising cut-rank.
static std::pair<uint64_t, uint64_t> greedy_bipartition(
    uint64_t vars_mask, const std::vector<uint64_t>& adj)
{
    // Extract variable indices
    std::vector<int> vars;
    vars.reserve(__builtin_popcountll(vars_mask));
    uint64_t tmp = vars_mask;
    while (tmp) { int v = __builtin_ctzll(tmp); tmp &= tmp-1; vars.push_back(v); }

    int n = (int)vars.size();
    assert(n >= 2);

    // Initial balanced split: lower half left, upper half right
    uint64_t left = 0, right = 0;
    for (int i = 0; i < n; i++) {
        if (i < n/2) left  |= (1ULL << vars[i]);
        else          right |= (1ULL << vars[i]);
    }

    int best = f2_cut_rank(adj, left, right);

    // Local search: try moving each vertex to the other side
    bool improved = true;
    while (improved && best > 0) {
        improved = false;
        for (int v : vars) {
            bool in_left = (left >> v) & 1;
            // Keep both sides non-empty
            if (in_left  && __builtin_popcountll(left)  == 1) continue;
            if (!in_left && __builtin_popcountll(right) == 1) continue;

            uint64_t nl = in_left ? (left  ^ (1ULL<<v)) : (left  | (1ULL<<v));
            uint64_t nr = in_left ? (right | (1ULL<<v)) : (right ^ (1ULL<<v));
            int r = f2_cut_rank(adj, nl, nr);
            if (r < best) {
                left = nl; right = nr; best = r; improved = true;
            }
        }
    }
    return {left, right};
}

// Recursively build the decomposition tree, returning the new node index.
static int32_t build_decomp(uint64_t vars_mask, std::vector<RWNode>& nodes,
                              const std::vector<uint64_t>& adj, int& width)
{
    RWNode node;
    node.vars_inside = vars_mask;

    if (__builtin_popcountll(vars_mask) == 1) {
        node.is_leaf = true;
        node.var = (uint32_t)__builtin_ctzll(vars_mask);
        int32_t idx = (int32_t)nodes.size();
        nodes.push_back(node);
        return idx;
    }

    auto [left_mask, right_mask] = greedy_bipartition(vars_mask, adj);
    assert(left_mask && right_mask);
    assert((left_mask | right_mask) == vars_mask);
    assert((left_mask & right_mask) == 0);

    int cr = f2_cut_rank(adj, left_mask, right_mask);
    if (cr > width) width = cr;

    // Build children before pushing this internal node (indices shift on push_back)
    int32_t lc = build_decomp(left_mask,  nodes, adj, width);
    int32_t rc = build_decomp(right_mask, nodes, adj, width);

    node.is_leaf = false;
    node.left  = lc;
    node.right = rc;
    int32_t idx = (int32_t)nodes.size();
    nodes.push_back(node);
    return idx;
}

RankDecomp compute_rank_decomp(uint32_t n, const std::vector<uint64_t>& adj) {
    assert(n <= 64 && "Rank decomposition requires n <= 64");
    assert((uint32_t)adj.size() == n);

    RankDecomp decomp;
    decomp.n = n;
    decomp.width = 0;
    if (n == 0) { decomp.root = -1; return decomp; }

    uint64_t all = (n == 64) ? ~0ULL : ((1ULL << n) - 1);
    int w = 0;
    decomp.root = build_decomp(all, decomp.nodes, adj, w);
    decomp.width = w;
    return decomp;
}

// =============================================================================
// Fourier-mode rank-width DP  (Algorithm 1, mode a = 1)
// =============================================================================

// DP table entry: accumulated Fourier value A^(1)[sigma] and one representative
// z-assignment (as a bitmask over variables in the current subtree).
struct DPEntry {
    std::complex<double> val;
    uint64_t repr;
    DPEntry() : val(0.0), repr(0) {}
    DPEntry(std::complex<double> v, uint64_t r) : val(v), repr(r) {}
};

using DPTable = std::unordered_map<uint64_t, DPEntry>;

static inline std::complex<double> omega_pow(int32_t exp, uint32_t r) {
    double angle = 2.0 * M_PI * (double)exp / (double)r;
    return {std::cos(angle), std::sin(angle)};
}

// Leaf: two entries for z_v = 0 and z_v = 1.
static DPTable process_leaf(const RWNode& node, const SOPInstance& sop) {
    uint32_t v = node.var;
    DPTable table;

    // z_v = 0: sigma = 0, value = 1, repr = 0
    table[0] = DPEntry(1.0, 0);

    // z_v = 1: sigma = adj[v] excluding v itself, value = omega^{b[v]}
    uint64_t sig1 = sop.adj[v] & ~(1ULL << v);
    std::complex<double> w1 = omega_pow(sop.b[v], sop.r);
    auto it = table.find(sig1);
    if (it == table.end()) {
        table[sig1] = DPEntry(w1, 1ULL << v);
    } else {
        it->second.val += w1;
        // repr already set (may be same sigma as z_v=0 if adj[v]=0)
    }
    return table;
}

// Internal node: combine left and right tables.
// X_L, X_R are bitmasks of variables in the left/right subtrees.
static DPTable process_join(const DPTable& tL, const DPTable& tR,
                             uint64_t X_L, uint64_t X_R,
                             const SOPInstance& sop)
{
    // Y = boundary of parent = V \ (X_L | X_R)
    uint64_t Y = sop.all_vars_mask ^ (X_L | X_R);

    DPTable table;
    for (const auto& [alpha, eL] : tL) {
        for (const auto& [beta, eR] : tR) {
            // chi = z_L^T A[X_L, X_R] z_R mod 2, using representatives
            uint64_t chi_acc = 0;
            uint64_t ls = eL.repr & X_L;
            while (ls) {
                int i = __builtin_ctzll(ls); ls &= ls - 1;
                chi_acc ^= (sop.adj[i] & X_R);
            }
            int chi = __builtin_popcountll(chi_acc & eR.repr) & 1;

            // gamma = Y-projection of alpha XOR Y-projection of beta
            uint64_t gamma = (alpha & Y) ^ (beta & Y);

            std::complex<double> contrib = eL.val * eR.val;
            if (chi) contrib = -contrib;

            auto it = table.find(gamma);
            if (it == table.end()) {
                table[gamma] = DPEntry(contrib, (eL.repr | eR.repr) & (X_L | X_R));
            } else {
                it->second.val += contrib;
            }
        }
    }
    return table;
}

static DPTable run_dp(int32_t idx, const std::vector<RWNode>& nodes,
                       const SOPInstance& sop)
{
    const RWNode& node = nodes[idx];
    if (node.is_leaf) return process_leaf(node, sop);

    DPTable tL = run_dp(node.left,  nodes, sop);
    DPTable tR = run_dp(node.right, nodes, sop);

    uint64_t X_L = nodes[node.left ].vars_inside;
    uint64_t X_R = nodes[node.right].vars_inside;
    return process_join(tL, tR, X_L, X_R, sop);
}

std::complex<double> sop_count(const SOPInstance& sop, const RankDecomp& decomp) {
    assert(sop.n <= 64);
    assert(decomp.root >= 0);

    DPTable table = run_dp(decomp.root, decomp.nodes, sop);

    // WMC = A^(1)_root[empty boundary] = table[0]
    auto it = table.find(0);
    std::complex<double> result = (it != table.end()) ? it->second.val : 0.0;

    // Apply global phase omega^c (a=1)
    result *= omega_pow(sop.c, sop.r);
    return result;
}

} // namespace GanakInt
