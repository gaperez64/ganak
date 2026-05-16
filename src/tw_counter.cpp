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
#include <functional>
#include <iostream>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace GanakInt {

// Returns the index of var in bag, or -1 if not present.
static int bag_idx(uint32_t var, const std::vector<int>& bag) {
    for (int i = 0; i < (int)bag.size(); i++)
        if (bag[i] == (int)var) return i;
    return -1;
}

std::optional<std::complex<double>> tw_wmc_count(
    const TWInstance& inst,
    int max_tw,
    int64_t td_steps,
    int td_iters,
    int verb)
{
    const uint32_t n = inst.n;
    const auto& clauses = inst.clauses;

    // ---- Build primal graph -----------------------------------------------
    TWD::Graph primal(n);
    for (const auto& cl : clauses) {
        for (uint32_t i = 0; i < cl.size(); i++)
            for (uint32_t j = i + 1; j < cl.size(); j++)
                primal.addEdge(cl[i].first, cl[j].first);
    }

    // ---- Run FlowCutter ---------------------------------------------------
    if (primal.numEdges() == 0 && n == 0)
        return std::complex<double>(1.0, 0.0);

    TWD::IFlowCutter fc(primal.numNodes(), primal.numEdges(), verb > 2 ? verb : 0);
    fc.importGraph(primal);
    auto td = fc.constructTD(td_steps, td_iters);

    int tw = td.width() - 1;
    if (verb >= 1)
        std::cout << "c o [tw] treewidth: " << tw << " (limit: " << max_tw << ")" << std::endl;
    if (tw < 0 || tw > max_tw) return {};

    const auto& bags = td.Bags();           // vector<vector<int>>, 0-based vars
    const auto& adj  = td.get_adj_list();   // vector<vector<int>>, bag adjacency
    int nb = (int)bags.size();
    if (nb == 0) return std::complex<double>(1.0, 0.0);

    // ---- Root the tree at node 0 (BFS) ------------------------------------
    std::vector<int> parent(nb, -1);
    std::vector<std::vector<int>> children(nb);
    std::vector<int> bfs_order;
    bfs_order.reserve(nb);
    std::vector<bool> visited(nb, false);
    std::queue<int> bfs_q;
    bfs_q.push(0); visited[0] = true;
    while (!bfs_q.empty()) {
        int t = bfs_q.front(); bfs_q.pop();
        bfs_order.push_back(t);
        for (int nb2 : adj[t]) {
            if (!visited[nb2]) {
                visited[nb2] = true;
                parent[nb2] = t;
                children[t].push_back(nb2);
                bfs_q.push(nb2);
            }
        }
    }

    // ---- Assign each variable to its shallowest bag (first in BFS order) --
    std::vector<int> var_owner(n, -1);
    for (int t : bfs_order) {
        for (int v : bags[t])
            if (var_owner[v] == -1) var_owner[v] = t;
    }

    // ---- Assign each clause to its shallowest covering bag ----------------
    // A bag covers a clause if it contains all clause vars.
    // BFS processes shallowest bags first, so take the first covering bag.
    // Pre-build: for each var, which bags contain it.
    std::vector<std::vector<int>> var_in_bags(n);
    for (int t = 0; t < nb; t++)
        for (int v : bags[t]) var_in_bags[v].push_back(t);

    // For each bag, which clauses it owns.
    std::vector<std::vector<int>> bag_clauses(nb);
    for (int ci = 0; ci < (int)clauses.size(); ci++) {
        const auto& cl = clauses[ci];
        if (cl.empty()) continue;  // tautology / empty clause handled below

        // Find shallowest bag containing all clause vars: BFS from root.
        // Use BFS order array directly.
        int owner = -1;
        for (int t : bfs_order) {
            bool covers = true;
            for (auto [v, neg] : cl) {
                if (bag_idx(v, bags[t]) < 0) { covers = false; break; }
            }
            if (covers) { owner = t; break; }
        }
        if (owner < 0) {
            // TD doesn't cover this clause; shouldn't happen for valid TD.
            if (verb >= 0)
                std::cerr << "c o [tw] ERROR: clause " << ci
                          << " not covered by any bag!" << std::endl;
            return {};
        }
        bag_clauses[owner].push_back(ci);
    }

    // ---- DP in post-order (leaves before parents) -------------------------
    // Post-order = reverse of BFS order reversed = reverse BFS order.
    std::vector<int> postorder(bfs_order.rbegin(), bfs_order.rend());

    // table[t]: vector of size 2^|bags[t]|, indexed by assignment bitmask.
    // Bit i of the bitmask = value (0 or 1) of bags[t][i].
    std::vector<std::vector<std::complex<double>>> tables(nb);

    for (int t : postorder) {
        const auto& Vt = bags[t];
        int szt = (int)Vt.size();
        uint32_t tsz = 1u << szt;

        auto& tbl = tables[t];
        tbl.assign(tsz, std::complex<double>(1.0, 0.0));

        // Multiply in variable weights for vars owned by this bag.
        for (int li = 0; li < szt; li++) {
            int v = Vt[li];
            if (var_owner[v] != t) continue;
            for (uint32_t sigma = 0; sigma < tsz; sigma++) {
                if ((sigma >> li) & 1) tbl[sigma] *= inst.pos_w[v];
                else                   tbl[sigma] *= inst.neg_w[v];
            }
        }

        // Apply clause constraint factors for clauses owned by this bag.
        for (int ci : bag_clauses[t]) {
            // Pre-compute local bit index for each literal.
            const auto& cl = clauses[ci];
            std::vector<std::pair<int, bool>> local_cl;
            local_cl.reserve(cl.size());
            for (auto [v, neg] : cl) {
                int li = bag_idx(v, Vt);
                assert(li >= 0);
                local_cl.emplace_back(li, neg);
            }
            for (uint32_t sigma = 0; sigma < tsz; sigma++) {
                bool sat = false;
                for (auto [li, neg] : local_cl) {
                    int val = (sigma >> li) & 1;
                    // neg=true (¬v): satisfied when val=0; neg=false (v): when val=1
                    if (neg ? (val == 0) : (val == 1)) { sat = true; break; }
                }
                if (!sat) tbl[sigma] = std::complex<double>(0.0, 0.0);
            }
        }

        // Merge each child table into this table.
        for (int c : children[t]) {
            const auto& Vc = bags[c];
            int szc = (int)Vc.size();

            // Build: for each position in Vc, the corresponding position in Vt (-1 if not shared).
            std::vector<int> child_to_parent(szc, -1);
            for (int ci2 = 0; ci2 < szc; ci2++)
                child_to_parent[ci2] = bag_idx(Vc[ci2], Vt);

            // Build shared_mask_in_t: bitmask of positions in Vt that are in Vc.
            uint32_t shared_mask = 0;
            for (int ci2 = 0; ci2 < szc; ci2++)
                if (child_to_parent[ci2] >= 0)
                    shared_mask |= (1u << child_to_parent[ci2]);

            // Compute marginal[sigma_t_bits] = Σ_{sigma_c matching} child_table[sigma_c]
            // sigma_t_bits is the subset of parent-table bits covered by shared vars.
            // We store marginal in a vector of size tsz (indexed by sigma_t & shared_mask).
            std::vector<std::complex<double>> marginal(tsz, std::complex<double>(0.0, 0.0));
            const auto& ctbl = tables[c];
            for (uint32_t sc = 0; sc < (1u << szc); sc++) {
                // Map child assignment sc to the parent-bit pattern.
                uint32_t st_bits = 0;
                for (int ci2 = 0; ci2 < szc; ci2++) {
                    int ti = child_to_parent[ci2];
                    if (ti >= 0 && ((sc >> ci2) & 1))
                        st_bits |= (1u << ti);
                }
                marginal[st_bits] += ctbl[sc];
            }

            // Multiply parent table: each sigma_t * marginal[sigma_t & shared_mask].
            for (uint32_t st = 0; st < tsz; st++)
                tbl[st] *= marginal[st & shared_mask];

            // Free child memory.
            tables[c].clear();
            tables[c].shrink_to_fit();
        }
    }

    // ---- Sum over all root assignments ------------------------------------
    std::complex<double> answer(0.0, 0.0);
    for (auto& v : tables[0]) answer += v;
    return answer;
}

} // namespace GanakInt
