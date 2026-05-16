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
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
***********************************************/

#include "tw_counter.hpp"
#include <treedecomp/IFlowCutter.hpp>
#include <treedecomp/graph.hpp>

#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <queue>
#include <vector>

namespace GanakInt {

// Returns the position of val in sorted vec, or -1.
static int sorted_pos(int val, const std::vector<int>& vec) {
    auto it = std::lower_bound(vec.begin(), vec.end(), val);
    return (it != vec.end() && *it == val) ? (int)(it - vec.begin()) : -1;
}

// Junction-tree WMC on an incidence-graph tree decomposition.
//
// Each bag t holds a mix of variable-nodes V_t and clause-nodes C_t. The DP
// table at t is indexed by a mask of size |V_t| + |C_t|:
//   low |V_t| bits = truth assignment of variables in V_t (1 = true)
//   high |C_t| bits = "has this clause been satisfied so far in t's subtree?"
//
// Per-edge ownership: every literal (v in clause c) is a graph edge in the
// incidence graph. The TD property guarantees at least one bag contains both
// v and the clause-node n+c. We pick one owner bag per edge; at that bag we
// OR-update the clause's sat bit using v's current assignment.
//
// Forgetting: when going from a child bag b to its parent t, variables in
// V_b \ V_t are marginalised (sum, weighted by pos_w / neg_w) and clauses in
// C_b \ C_t are required to be satisfied (sat=0 entries are dropped).
//
// At the root, remaining variables are marginalised and remaining clauses
// must be satisfied.
std::optional<std::complex<double>> tw_wmc_count(
    const TWInstance& inst,
    int max_tw,
    int64_t td_steps,
    int td_iters,
    int verb)
{
    const uint32_t n = inst.n;
    const uint32_t m = (uint32_t)inst.clauses.size();
    const auto& clauses = inst.clauses;

    if (n == 0) return std::complex<double>(1.0, 0.0);

    // An empty clause is unsatisfiable.
    for (uint32_t ci = 0; ci < m; ci++)
        if (clauses[ci].empty()) return std::complex<double>(0.0, 0.0);

    // ---- Build incidence graph -------------------------------------------
    // Nodes 0..n-1 are variables; nodes n..n+m-1 are clauses.
    TWD::Graph incidence(n + m);
    for (uint32_t ci = 0; ci < m; ci++)
        for (auto [v, neg] : clauses[ci])
            incidence.addEdge((int)v, (int)(n + ci));

    // ---- Run FlowCutter --------------------------------------------------
    TWD::IFlowCutter fc(incidence.numNodes(), incidence.numEdges(), verb > 2 ? verb : 0);
    fc.importGraph(incidence);
    auto td = fc.constructTD(td_steps, td_iters);

    int tw = td.width();  // already max_bag_size - 1 (standard treewidth)
    if (verb >= 1)
        std::cout << "c o [tw] incidence-graph treewidth: " << tw
                  << " (limit: " << max_tw << ")" << std::endl;
    if (tw < 0 || tw > max_tw) return {};

    const auto& bags = td.Bags();
    const auto& adj  = td.get_adj_list();
    int nb = (int)bags.size();
    if (nb == 0) return std::complex<double>(1.0, 0.0);

    // ---- Split bags into var-nodes and clause-nodes; sort both lists -----
    struct BagInfo {
        std::vector<int> vars;  // sorted var indices (0..n-1)
        std::vector<int> cls;   // sorted clause indices (bag-node minus n)
    };
    std::vector<BagInfo> bi(nb);
    for (int t = 0; t < nb; t++) {
        for (int node : bags[t]) {
            if (node < (int)n) bi[t].vars.push_back(node);
            else               bi[t].cls.push_back(node - (int)n);
        }
        std::sort(bi[t].vars.begin(), bi[t].vars.end());
        std::sort(bi[t].cls.begin(),  bi[t].cls.end());
        int bag_sz = (int)(bi[t].vars.size() + bi[t].cls.size());
        if (bag_sz > 30) {
            if (verb >= 0)
                std::cerr << "c o [tw] bag " << t << " size " << bag_sz
                          << " > 30; would overflow table mask\n";
            return {};
        }
    }

    // ---- Root the tree (BFS from node 0) ---------------------------------
    std::vector<int> par(nb, -1);
    std::vector<std::vector<int>> ch(nb);
    std::vector<int> bfs_order;
    bfs_order.reserve(nb);
    {
        std::vector<bool> seen(nb, false);
        std::queue<int> q;
        q.push(0); seen[0] = true;
        while (!q.empty()) {
            int t = q.front(); q.pop();
            bfs_order.push_back(t);
            for (int nb2 : adj[t]) {
                if (!seen[nb2]) {
                    seen[nb2] = true;
                    par[nb2] = t;
                    ch[t].push_back(nb2);
                    q.push(nb2);
                }
            }
        }
    }
    // Post-order: process children before parent.
    std::vector<int> postorder(bfs_order.rbegin(), bfs_order.rend());

    // ---- Per-edge ownership ----------------------------------------------
    // For each incidence-graph edge (v, n+ci), pick any bag containing both
    // endpoints and own the OR-update there.
    struct EdgeOwner {
        int vi;   // index of var in owner bag's V_t
        int cj;   // index of clause in owner bag's C_t
        bool neg; // polarity of v in clause ci (true = literal is ~v)
    };
    std::vector<std::vector<EdgeOwner>> bag_edges(nb);
    for (uint32_t ci = 0; ci < m; ci++) {
        for (auto [v, neg] : clauses[ci]) {
            int owner = -1, owner_vi = -1, owner_cj = -1;
            for (int t = 0; t < nb; t++) {
                int vi = sorted_pos((int)v, bi[t].vars);
                if (vi < 0) continue;
                int cj = sorted_pos((int)ci, bi[t].cls);
                if (cj < 0) continue;
                owner = t; owner_vi = vi; owner_cj = cj;
                break;
            }
            if (owner < 0) {
                if (verb >= 0)
                    std::cerr << "c o [tw] edge (var=" << v
                              << ", clause=" << ci
                              << ") has no covering bag; TD invalid\n";
                return {};
            }
            bag_edges[owner].push_back({owner_vi, owner_cj, neg});
        }
    }

    // ---- Junction-tree DP -----------------------------------------------
    std::vector<std::vector<std::complex<double>>> tables(nb);

    for (int t : postorder) {
        const int nv = (int)bi[t].vars.size();
        const int nc = (int)bi[t].cls.size();
        const uint64_t tsz = 1ull << (nv + nc);
        const uint64_t var_mask = (nv > 0) ? ((1ull << nv) - 1ull) : 0ull;

        auto& tbl = tables[t];
        tbl.assign(tsz, std::complex<double>(0.0, 0.0));
        // Each clause local to t starts unsatisfied (sat bits = 0).
        for (uint64_t va = 0; va <= var_mask; va++)
            tbl[va] = std::complex<double>(1.0, 0.0);

        // ---- Combine each child cb into tbl ------------------------------
        for (int cb : ch[t]) {
            const int cnv = (int)bi[cb].vars.size();
            const int cnc = (int)bi[cb].cls.size();
            const uint64_t cts = 1ull << (cnv + cnc);

            // Map child positions to parent positions (-1 = forgotten).
            std::vector<int> cv_to_pv(cnv, -1);
            for (int i = 0; i < cnv; i++)
                cv_to_pv[i] = sorted_pos(bi[cb].vars[i], bi[t].vars);
            std::vector<int> cc_to_pc(cnc, -1);
            for (int j = 0; j < cnc; j++)
                cc_to_pc[j] = sorted_pos(bi[cb].cls[j], bi[t].cls);

            // Bits in the parent's var-mask that correspond to shared vars.
            uint64_t parent_shared_var_mask = 0;
            for (int i = 0; i < cnv; i++)
                if (cv_to_pv[i] >= 0)
                    parent_shared_var_mask |= (1ull << cv_to_pv[i]);

            std::vector<std::complex<double>> new_tbl(tsz, std::complex<double>(0.0, 0.0));

            for (uint64_t cs = 0; cs < cts; cs++) {
                const std::complex<double>& base = tables[cb][cs];
                if (base.real() == 0.0 && base.imag() == 0.0) continue;

                // Forgotten clauses (in cb but not in t) must be satisfied.
                bool valid = true;
                for (int j = 0; j < cnc; j++) {
                    if (cc_to_pc[j] < 0 && ((cs >> (cnv + j)) & 1) == 0) {
                        valid = false; break;
                    }
                }
                if (!valid) continue;

                // Apply forgotten-variable weights.
                std::complex<double> contrib = base;
                for (int i = 0; i < cnv; i++) {
                    if (cv_to_pv[i] < 0) {
                        int v = bi[cb].vars[i];
                        contrib *= ((cs >> i) & 1) ? inst.pos_w[v] : inst.neg_w[v];
                    }
                }

                // Shared-var bits expressed in parent positions.
                uint64_t child_sv_in_parent = 0;
                for (int i = 0; i < cnv; i++)
                    if (cv_to_pv[i] >= 0 && ((cs >> i) & 1))
                        child_sv_in_parent |= (1ull << cv_to_pv[i]);

                // Shared-clause sat bits expressed in parent positions.
                uint64_t child_sc_in_parent = 0;
                for (int j = 0; j < cnc; j++)
                    if (cc_to_pc[j] >= 0 && ((cs >> (cnv + j)) & 1))
                        child_sc_in_parent |= (1ull << (nv + cc_to_pc[j]));

                // Join: every parent state whose shared-var bits match
                // child_sv_in_parent combines with this child state.
                // The new parent sat bits are OR'd with the child's.
                for (uint64_t ps_old = 0; ps_old < tsz; ps_old++) {
                    if ((ps_old & parent_shared_var_mask) != child_sv_in_parent) continue;
                    const std::complex<double>& v = tbl[ps_old];
                    if (v.real() == 0.0 && v.imag() == 0.0) continue;
                    uint64_t ps_new = ps_old | child_sc_in_parent;
                    new_tbl[ps_new] += v * contrib;
                }
            }

            tbl = std::move(new_tbl);
            tables[cb].clear();
            tables[cb].shrink_to_fit();
        }

        // ---- Apply edge OR-updates owned by t ----------------------------
        // For each owned literal (v, clause c, polarity neg), every state where
        // c is not yet satisfied AND v's value satisfies the literal gets its
        // sat bit forced to 1; the value moves from old state to new.
        for (const auto& e : bag_edges[t]) {
            uint64_t sat_bit = 1ull << (nv + e.cj);
            uint64_t var_bit = 1ull << e.vi;
            for (uint64_t st = 0; st < tsz; st++) {
                if (st & sat_bit) continue;            // already satisfied
                bool val_true = (st & var_bit) != 0;
                bool literal_sat = e.neg ? !val_true : val_true;
                if (!literal_sat) continue;
                std::complex<double>& src = tbl[st];
                if (src.real() == 0.0 && src.imag() == 0.0) continue;
                tbl[st | sat_bit] += src;
                src = std::complex<double>(0.0, 0.0);
            }
        }
    }

    // ---- Marginalize at the root ----------------------------------------
    const int rnv = (int)bi[0].vars.size();
    const int rnc = (int)bi[0].cls.size();
    const uint64_t rtsz = 1ull << (rnv + rnc);
    std::complex<double> answer(0.0, 0.0);
    for (uint64_t st = 0; st < rtsz; st++) {
        // All remaining clauses must be satisfied.
        bool ok = true;
        for (int j = 0; j < rnc; j++) {
            if (((st >> (rnv + j)) & 1) == 0) { ok = false; break; }
        }
        if (!ok) continue;
        std::complex<double> contrib = tables[0][st];
        if (contrib.real() == 0.0 && contrib.imag() == 0.0) continue;
        for (int i = 0; i < rnv; i++) {
            int v = bi[0].vars[i];
            contrib *= ((st >> i) & 1) ? inst.pos_w[v] : inst.neg_w[v];
        }
        answer += contrib;
    }
    return answer;
}

} // namespace GanakInt
