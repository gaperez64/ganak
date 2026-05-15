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

#include <vector>
#include <cstdint>
#include <cassert>

namespace GanakInt {

struct RWNode {
    bool is_leaf = false;
    uint32_t var = 0;       // leaf: 0-based variable index
    int32_t  left  = -1;    // index of left child in nodes array
    int32_t  right = -1;    // index of right child in nodes array
    uint64_t vars_inside = 0; // bitmask of variables in this subtree
};

struct RankDecomp {
    std::vector<RWNode> nodes;
    int32_t root  = -1;
    int32_t width = 0;
    uint32_t n    = 0;
};

// F_2 rank of the adj[left_mask][right_mask] submatrix via Gaussian elimination.
inline int f2_cut_rank(const std::vector<uint64_t>& adj,
                        uint64_t left_mask, uint64_t right_mask) {
    std::vector<uint64_t> rows;
    rows.reserve(__builtin_popcountll(left_mask));
    uint64_t tmp = left_mask;
    while (tmp) {
        int v = __builtin_ctzll(tmp); tmp &= tmp - 1;
        uint64_t row = adj[v] & right_mask;
        if (row) rows.push_back(row);
    }
    int rank = 0;
    for (int col = 63; col >= 0; --col) {
        if (!((right_mask >> col) & 1)) continue;
        uint64_t col_bit = (1ULL << col);
        int pivot = -1;
        for (int i = rank; i < (int)rows.size(); ++i)
            if (rows[i] & col_bit) { pivot = i; break; }
        if (pivot < 0) continue;
        std::swap(rows[rank], rows[pivot]);
        for (int i = 0; i < (int)rows.size(); ++i)
            if (i != rank && (rows[i] & col_bit)) rows[i] ^= rows[rank];
        rank++;
    }
    return rank;
}

// Build a rank decomposition of an n-vertex graph (n <= 64) given as adjacency
// bitmasks adj[i] where bit j is set iff edge {i,j} exists. Returns the
// decomposition tree; width field holds the rank-width.
RankDecomp compute_rank_decomp(uint32_t n, const std::vector<uint64_t>& adj);

} // namespace GanakInt
