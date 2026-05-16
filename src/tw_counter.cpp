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

    // ---- Build incidence graph -------------------------------------------
    // Nodes 0..n-1 are variables; nodes n..n+m-1 are clauses.
    // An edge connects variable v to clause-node n+ci whenever v appears in ci.
    TWD::Graph incidence(n + m);
    for (uint32_t ci = 0; ci < m; ci++)
        for (auto [v, neg] : clauses[ci])
            incidence.addEdge((int)v, (int)(n + ci));

    // ---- Run FlowCutter --------------------------------------------------
    TWD::IFlowCutter fc(incidence.numNodes(), incidence.numEdges(), verb > 2 ? verb : 0);
    fc.importGraph(incidence);
    auto td = fc.constructTD(td_steps, td_iters);

    int tw = td.width() - 1;
    if (verb >= 1)
        std::cout << "c o [tw] incidence-graph treewidth: " << tw
                  << " (limit: " << max_tw << ")" << std::endl;
    if (tw < 0 || tw > max_tw) return {};

    const auto& bags = td.Bags();
    const auto& adj  = td.get_adj_list();
    int nb = (int)bags.size();
    if (nb == 0) return std::complex<double>(1.0, 0.0);

    // ---- Split bags into var-nodes and clause-nodes; sort var lists -------
    struct BagInfo {
        std::vector<int> vars;   // sorted var indices (0..n-1)
        std::vector<int> cls;    // clause indices (bag-node minus n)
    };
    std::vector<BagInfo> bi(nb);
    for (int t = 0; t < nb; t++) {
        for (int node : bags[t]) {
            if (node < (int)n) bi[t].vars.push_back(node);
            else               bi[t].cls.push_back(node - (int)n);
        }
        std::sort(bi[t].vars.begin(), bi[t].vars.end());
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
    // Post-order = reverse BFS (leaves before parents)
    std::vector<int> postorder(bfs_order.rbegin(), bfs_order.rend());

    // ---- Assign each clause to its deepest covering bag ------------------
    // "Covering" means the bag contains the clause-node n+ci AND all var-nodes
    // of the clause. The Helly property guarantees such a bag exists.
    std::vector<std::vector<int>> bag_clauses(nb);
    for (uint32_t ci = 0; ci < m; ci++) {
        int owner = -1;
        for (int t : postorder) {
            // Does bag t contain clause-node ci?
            bool has_cl = std::find(bi[t].cls.begin(), bi[t].cls.end(), (int)ci)
                          != bi[t].cls.end();
            if (!has_cl) continue;
            // Does bag t contain all var-nodes of clause ci?
            bool all_vars = true;
            for (auto [v, neg] : clauses[ci])
                if (sorted_pos((int)v, bi[t].vars) < 0) { all_vars = false; break; }
            if (all_vars) { owner = t; break; }
        }
        if (owner < 0) {
            if (verb >= 0)
                std::cerr << "c o [tw] clause " << ci
                          << " has no covering bag; TD invalid\n";
            return {};
        }
        bag_clauses[owner].push_back((int)ci);
    }

    // ---- Junction-tree DP -----------------------------------------------
    // table[t]: size 2^|bi[t].vars|, indexed by assignment bitmask.
    // Bit i of bitmask = value of bi[t].vars[i].
    //
    // Variable weights are applied at the "forget" step: when var v is in
    // bag t but NOT in parent(t), we sum over v's values weighted by pos_w/neg_w.
    // Clause constraints are applied at each clause's owner bag.
    std::vector<std::vector<std::complex<double>>> tables(nb);

    for (int t : postorder) {
        const auto& Vt = bi[t].vars;
        int szt = (int)Vt.size();
        uint32_t tsz = 1u << szt;

        auto& tbl = tables[t];
        tbl.assign(tsz, std::complex<double>(1.0, 0.0));

        // Apply clause constraints owned by this bag.
        for (int ci : bag_clauses[t]) {
            std::vector<std::pair<int, bool>> local_cl;
            local_cl.reserve(clauses[ci].size());
            for (auto [v, neg] : clauses[ci])
                local_cl.emplace_back(sorted_pos((int)v, Vt), neg);

            for (uint32_t sigma = 0; sigma < tsz; sigma++) {
                bool sat = false;
                for (auto [li, neg] : local_cl) {
                    int val = (sigma >> li) & 1;
                    if (neg ? (val == 0) : (val == 1)) { sat = true; break; }
                }
                if (!sat) tbl[sigma] = 0.0;
            }
        }

        // Merge each child's (already weighted-marginalized) table.
        for (int c : ch[t]) {
            const auto& Vc = bi[c].vars;
            int szc = (int)Vc.size();

            // For each position in Vc: its position in Vt, or -1 (= forgotten).
            std::vector<int> c2p(szc, -1);
            for (int ci2 = 0; ci2 < szc; ci2++)
                c2p[ci2] = sorted_pos(Vc[ci2], Vt);

            std::vector<int> forget_pos;  // positions in Vc of vars not in Vt
            for (int ci2 = 0; ci2 < szc; ci2++)
                if (c2p[ci2] < 0) forget_pos.push_back(ci2);

            uint32_t shared_mask = 0;
            for (int ci2 = 0; ci2 < szc; ci2++)
                if (c2p[ci2] >= 0) shared_mask |= (1u << c2p[ci2]);

            // Compute marginal: sum child table over forgotten vars, weighted.
            std::vector<std::complex<double>> marginal(tsz, 0.0);
            const auto& ctbl = tables[c];
            for (uint32_t sc = 0; sc < (1u << szc); sc++) {
                // Multiply in weights for forgotten vars.
                std::complex<double> contrib = ctbl[sc];
                for (int fi : forget_pos) {
                    int v = Vc[fi];
                    int val = (sc >> fi) & 1;
                    contrib *= val ? inst.pos_w[v] : inst.neg_w[v];
                }
                // Map to parent-table shared bits.
                uint32_t st_bits = 0;
                for (int ci2 = 0; ci2 < szc; ci2++)
                    if (c2p[ci2] >= 0 && ((sc >> ci2) & 1))
                        st_bits |= (1u << c2p[ci2]);
                marginal[st_bits] += contrib;
            }

            for (uint32_t st = 0; st < tsz; st++)
                tbl[st] *= marginal[st & shared_mask];

            tables[c].clear();
            tables[c].shrink_to_fit();
        }
    }

    // ---- Sum root table, applying weights for all root vars (all forgotten) --
    std::complex<double> answer(0.0, 0.0);
    const auto& Vr = bi[0].vars;
    for (uint32_t sigma = 0; sigma < (1u << Vr.size()); sigma++) {
        std::complex<double> w = tables[0][sigma];
        for (int i = 0; i < (int)Vr.size(); i++)
            w *= ((sigma >> i) & 1) ? inst.pos_w[Vr[i]] : inst.neg_w[Vr[i]];
        answer += w;
    }
    return answer;
}

} // namespace GanakInt
