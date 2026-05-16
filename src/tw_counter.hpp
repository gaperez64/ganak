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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL ANY
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
***********************************************/

#pragma once

#include <complex>
#include <optional>
#include <vector>
#include <cstdint>

namespace GanakInt {

struct TWInstance {
    uint32_t n = 0;
    // clauses[i] = list of {var (0-based), is_negated}
    std::vector<std::vector<std::pair<uint32_t, bool>>> clauses;
    // weights: w(v=1) and w(v=0) for each variable v in [0, n)
    std::vector<std::complex<double>> pos_w;
    std::vector<std::complex<double>> neg_w;
};

// Junction-tree WMC via tree decomposition of the primal graph.
// Returns {} if the computed treewidth exceeds max_tw or FlowCutter fails.
// td_steps / td_iters are passed to IFlowCutter::constructTD.
std::optional<std::complex<double>> tw_wmc_count(
    const TWInstance& inst,
    int max_tw,
    int64_t td_steps,
    int td_iters,
    int verb = 0);

} // namespace GanakInt
