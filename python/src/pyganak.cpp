/******************************************
Copyright (C) 2023-2025 Authors of GANAK, see AUTHORS file

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

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <ganak.hpp>
#include <arjun/arjun.h>
#include <cryptominisat5/solvertypesmini.h>
#include <mpfr.h>

// FPT dispatch paths (incidence-graph junction-tree DP and quadratic SOP DP).
// These live under src/ and are reached via the include path passed to the
// pyganak target in src/CMakeLists.txt.
#include "tw_counter.hpp"
#include "sop_counter.hpp"
#include "rw_decomp.hpp"

#include <cmath>
#include <complex>
#include <vector>
#include <set>
#include <string>
#include <memory>
#include <stdexcept>

#ifndef PYGANAK_VERSION
#define PYGANAK_VERSION "1.1.0"
#endif

// -------------------------------------------------------------------------
// Internal state stored per Counter object
// -------------------------------------------------------------------------
struct CounterState {
    int verbose = 0;
    uint64_t seed = 0;
    uint32_t max_var = 0;  // highest 1-indexed variable seen so far
    std::vector<std::vector<CMSat::Lit>> clauses;
    std::vector<uint32_t> sampling_set;   // 0-indexed (CMSat convention)
    bool sampling_set_given = false;
    bool use_tw = false;
    int  tw_max = 20;
    bool counted = false;  // count() may only be called once
};

typedef struct {
    PyObject_HEAD
    CounterState* state;
} Counter;

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

// Convert a Python integer literal value (signed, 1-indexed) to CMSat::Lit.
// Returns false and sets a Python exception on error.
static bool lit_from_py_long(PyObject* item, CMSat::Lit& out_lit, uint32_t& max_var) {
    long val = PyLong_AsLong(item);
    if (val == -1L && PyErr_Occurred()) return false;
    if (val == 0) {
        PyErr_SetString(PyExc_ValueError, "Clause contains 0; literals must be nonzero integers");
        return false;
    }
    uint32_t var = (uint32_t)((val < 0) ? -val : val) - 1;  // convert to 0-indexed
    bool sign = (val < 0);                                    // true = negative literal
    out_lit = CMSat::Lit(var, sign);
    if (var + 1 > max_var) max_var = var + 1;
    return true;
}

// Parse an iterable of signed integers into a vector of CMSat::Lit.
static bool parse_clause(PyObject* clause_obj,
                          std::vector<CMSat::Lit>& lits,
                          uint32_t& max_var) {
    PyObject* iter = PyObject_GetIter(clause_obj);
    if (!iter) return false;

    PyObject* item;
    while ((item = PyIter_Next(iter)) != nullptr) {
        CMSat::Lit lit;
        bool ok = lit_from_py_long(item, lit, max_var);
        Py_DECREF(item);
        if (!ok) { Py_DECREF(iter); return false; }
        lits.push_back(lit);
    }
    Py_DECREF(iter);
    return !PyErr_Occurred();
}

// -------------------------------------------------------------------------
// Counter.__new__ / __init__ / __dealloc__
// -------------------------------------------------------------------------

static PyObject* Counter_new(PyTypeObject* type, PyObject* /*args*/, PyObject* /*kwds*/) {
    Counter* self = (Counter*)type->tp_alloc(type, 0);
    if (self) {
        self->state = new (std::nothrow) CounterState();
        if (!self->state) {
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
    }
    return (PyObject*)self;
}

static int Counter_init(Counter* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"verbose", "seed", "use_tw", "tw_max", nullptr};
    int verbose = 0;
    unsigned long long seed = 0;
    int use_tw = 0;
    int tw_max = 20;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iKpi",
                                      const_cast<char**>(kwlist),
                                      &verbose, &seed, &use_tw, &tw_max))
        return -1;
    if (tw_max < 0) {
        PyErr_SetString(PyExc_ValueError, "tw_max must be non-negative");
        return -1;
    }
    self->state->verbose = verbose;
    self->state->seed = (uint64_t)seed;
    self->state->use_tw = (use_tw != 0);
    self->state->tw_max = tw_max;
    return 0;
}

static void Counter_dealloc(Counter* self) {
    delete self->state;
    self->state = nullptr;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// -------------------------------------------------------------------------
// Counter.add_clause(clause)
// -------------------------------------------------------------------------

static PyObject* Counter_add_clause(Counter* self, PyObject* args) {
    PyObject* clause_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &clause_obj)) return nullptr;

    std::vector<CMSat::Lit> lits;
    if (!parse_clause(clause_obj, lits, self->state->max_var)) return nullptr;

    self->state->clauses.push_back(std::move(lits));
    Py_RETURN_NONE;
}

// -------------------------------------------------------------------------
// Counter.add_clauses(clauses)   -- iterable of clauses
// -------------------------------------------------------------------------

static PyObject* Counter_add_clauses(Counter* self, PyObject* args) {
    PyObject* clauses_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &clauses_obj)) return nullptr;

    PyObject* outer_iter = PyObject_GetIter(clauses_obj);
    if (!outer_iter) return nullptr;

    PyObject* clause_obj;
    while ((clause_obj = PyIter_Next(outer_iter)) != nullptr) {
        std::vector<CMSat::Lit> lits;
        bool ok = parse_clause(clause_obj, lits, self->state->max_var);
        Py_DECREF(clause_obj);
        if (!ok) { Py_DECREF(outer_iter); return nullptr; }
        self->state->clauses.push_back(std::move(lits));
    }
    Py_DECREF(outer_iter);
    if (PyErr_Occurred()) return nullptr;
    Py_RETURN_NONE;
}

// -------------------------------------------------------------------------
// Counter.set_sampling_set(vars)
// vars: iterable of 1-indexed variable numbers
// -------------------------------------------------------------------------

static PyObject* Counter_set_sampling_set(Counter* self, PyObject* args) {
    PyObject* vars_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &vars_obj)) return nullptr;

    PyObject* iter = PyObject_GetIter(vars_obj);
    if (!iter) return nullptr;

    std::vector<uint32_t> sampl;
    PyObject* item;
    while ((item = PyIter_Next(iter)) != nullptr) {
        long v = PyLong_AsLong(item);
        Py_DECREF(item);
        if (v == -1L && PyErr_Occurred()) { Py_DECREF(iter); return nullptr; }
        if (v <= 0) {
            PyErr_SetString(PyExc_ValueError, "Sampling set variables must be positive integers (1-indexed)");
            Py_DECREF(iter);
            return nullptr;
        }
        uint32_t var0 = (uint32_t)v - 1;   // 0-indexed
        sampl.push_back(var0);
        if (var0 + 1 > self->state->max_var) self->state->max_var = var0 + 1;
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) return nullptr;

    self->state->sampling_set = std::move(sampl);
    self->state->sampling_set_given = true;
    Py_RETURN_NONE;
}

// -------------------------------------------------------------------------
// Treewidth DP dispatch (shared by Counter and WeightedCounter).
//
// Returns std::nullopt if the DP cannot or should not be used (treewidth
// exceeds cap, projection set is a strict subset, or n == 0). Callers must
// fall back to the standard Arjun + Ganak pipeline in that case.
// -------------------------------------------------------------------------
static std::optional<std::complex<double>> try_tw_dp(
        uint32_t max_var,
        const std::vector<std::vector<CMSat::Lit>>& clauses,
        const std::vector<uint32_t>& sampl_set, bool ss_given,
        int tw_max, int verbose,
        const std::vector<std::complex<double>>* pos_w = nullptr,
        const std::vector<std::complex<double>>* neg_w = nullptr)
{
    if (max_var == 0) return {};

    // The DP sums over the full Boolean cube; a strict projection set would
    // give the wrong answer, so we refuse it.
    if (ss_given && sampl_set.size() != max_var) return {};

    GanakInt::TWInstance inst;
    inst.n = max_var;
    inst.pos_w.assign(inst.n, std::complex<double>(1.0, 0.0));
    inst.neg_w.assign(inst.n, std::complex<double>(1.0, 0.0));
    if (pos_w) inst.pos_w = *pos_w;
    if (neg_w) inst.neg_w = *neg_w;

    inst.clauses.reserve(clauses.size());
    for (const auto& cl : clauses) {
        std::vector<std::pair<uint32_t, bool>> tw_cl;
        tw_cl.reserve(cl.size());
        for (const auto& l : cl) tw_cl.emplace_back(l.var(), l.sign());
        inst.clauses.push_back(std::move(tw_cl));
    }

    // td_steps / td_iters: match the CounterConfiguration defaults so the
    // FlowCutter behaviour is identical to the CLI.
    return GanakInt::tw_wmc_count(inst, tw_max, /*td_steps=*/100000,
                                  /*td_iters=*/900, /*verb=*/verbose);
}

// -------------------------------------------------------------------------
// Counter.count()
// Returns a Python int (exact model count).
// Arjun preprocessing is run automatically.
// -------------------------------------------------------------------------

// Worker: runs entirely outside the GIL.  Returns the count or throws.
static std::unique_ptr<CMSat::Field> do_count(
        int verbose, uint64_t seed, uint32_t max_var,
        const std::vector<std::vector<CMSat::Lit>>& clauses,
        const std::vector<uint32_t>& sampl_set, bool ss_given)
{
    std::unique_ptr<CMSat::FieldGen> fg = std::make_unique<ArjunNS::FGenMpz>();

    // Trivial case: no variables → 1 satisfying assignment (empty assignment).
    if (max_var == 0) return fg->one();

    ArjunNS::SimplifiedCNF cnf(fg);
    cnf.new_vars(max_var);
    for (const auto& cl : clauses) cnf.add_clause(cl);

    if (ss_given) {
        cnf.set_sampl_vars(sampl_set);
    } else {
        std::vector<uint32_t> all_vars;
        all_vars.reserve(max_var);
        for (uint32_t i = 0; i < max_var; i++) all_vars.push_back(i);
        cnf.set_sampl_vars(all_vars);
    }

    ArjunNS::Arjun arjun;
    arjun.set_verb(0);
    ArjunNS::Arjun::ElimToFileConf etof_conf;
    ArjunNS::SimpConf simp_conf;
    arjun.standalone_minimize_indep(cnf, /*all_indep=*/false);
    arjun.standalone_elim_to_file(cnf, etof_conf, simp_conf);

    GanakInt::CounterConfiguration conf;
    conf.verb = verbose;
    conf.seed = seed;
    Ganak counter(conf, fg);
    setup_ganak(cnf, counter);

    const auto& mw = cnf.get_multiplier_weight();
    auto cnt = mw->dup();
    if (!mw->is_zero()) {
        *cnt *= *counter.count();
    }
    return cnt;
}

static PyObject* Counter_count(Counter* self, PyObject*) {
    if (self->state->counted) {
        PyErr_SetString(PyExc_RuntimeError, "count() may only be called once per Counter instance");
        return nullptr;
    }
    self->state->counted = true;

    // Snapshot all state with GIL held, then release GIL for the slow work.
    const int      verbose  = self->state->verbose;
    const uint64_t seed     = self->state->seed;
    const uint32_t max_var  = self->state->max_var;
    const bool     ss_given = self->state->sampling_set_given;
    const bool     use_tw   = self->state->use_tw;
    const int      tw_max   = self->state->tw_max;
    auto clauses_copy   = self->state->clauses;
    auto sampl_set_copy = self->state->sampling_set;

    std::unique_ptr<CMSat::Field> result_field;
    std::optional<std::complex<double>> tw_result;
    std::string error_msg;
    bool success = false;

    Py_BEGIN_ALLOW_THREADS
    try {
        if (use_tw) {
            tw_result = try_tw_dp(max_var, clauses_copy, sampl_set_copy, ss_given,
                                  tw_max, verbose);
        }
        if (!tw_result) {
            result_field = do_count(verbose, seed, max_var,
                                    clauses_copy, sampl_set_copy, ss_given);
        }
        success = true;
    } catch (const std::exception& e) {
        error_msg = e.what();
    } catch (...) {
        error_msg = "Unknown C++ exception in count()";
    }
    Py_END_ALLOW_THREADS

    if (!success) {
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return nullptr;
    }

    // TW DP path: returns complex<double>; round real part to int. Above 2^53
    // the rounding loses precision — document this and let the caller decide
    // whether to use_tw for huge counts.
    if (tw_result) {
        double re = tw_result->real();
        if (std::isnan(re) || std::isinf(re)) {
            PyErr_SetString(PyExc_RuntimeError,
                "tw DP produced non-finite count");
            return nullptr;
        }
        double rounded = std::round(re);
        return PyLong_FromDouble(rounded);
    }

    auto& mpz_result = static_cast<ArjunNS::FMpz&>(*result_field);
    std::string s = mpz_result.val.get_str(10);
    return PyLong_FromString(s.c_str(), nullptr, 10);
}

// -------------------------------------------------------------------------
// Counter.new_vars(n)  -- declare n additional variables
// (Useful to force the variable count without adding clauses.)
// -------------------------------------------------------------------------

static PyObject* Counter_new_vars(Counter* self, PyObject* args) {
    unsigned int n = 0;
    if (!PyArg_ParseTuple(args, "I", &n)) return nullptr;
    uint32_t new_max = self->state->max_var + n;
    if (new_max < self->state->max_var) {
        PyErr_SetString(PyExc_OverflowError, "new_vars: variable count overflow");
        return nullptr;
    }
    self->state->max_var = new_max;
    Py_RETURN_NONE;
}

// -------------------------------------------------------------------------
// Counter.nof_vars()  -- number of variables declared so far
// -------------------------------------------------------------------------

static PyObject* Counter_nof_vars(Counter* self, PyObject*) {
    return PyLong_FromUnsignedLong((unsigned long)self->state->max_var);
}

// -------------------------------------------------------------------------
// Counter.nof_clauses()
// -------------------------------------------------------------------------

static PyObject* Counter_nof_clauses(Counter* self, PyObject*) {
    return PyLong_FromSize_t(self->state->clauses.size());
}

// -------------------------------------------------------------------------
// Method table
// -------------------------------------------------------------------------

static PyMethodDef counter_methods[] = {
    {"add_clause",       (PyCFunction)Counter_add_clause,       METH_VARARGS,
     "add_clause(clause)\n\n"
     "Add a clause. *clause* is an iterable of nonzero signed integers;\n"
     "positive integers represent positive literals, negative integers\n"
     "represent negative literals.  Variable indices are 1-based.\n\n"
     "Example::\n\n"
     "    c.add_clause([1, -2, 3])   # x1 OR NOT x2 OR x3\n"},

    {"add_clauses",      (PyCFunction)Counter_add_clauses,      METH_VARARGS,
     "add_clauses(clauses)\n\n"
     "Add multiple clauses at once. *clauses* is an iterable of clauses,\n"
     "where each clause is an iterable of nonzero signed integers.\n"},

    {"set_sampling_set", (PyCFunction)Counter_set_sampling_set, METH_VARARGS,
     "set_sampling_set(vars)\n\n"
     "Set the *sampling set* (also called the independent support).\n"
     "*vars* is an iterable of positive integers (1-based variable indices).\n"
     "Only variables in this set are counted; variables not in the set are\n"
     "treated as existentially quantified.  If this method is never called,\n"
     "all variables are included in the sampling set (unweighted projected\n"
     "model counting becomes plain model counting).\n\n"
     "If any variable index in *vars* exceeds the current variable count,\n"
     "the variable universe is extended automatically to accommodate it.\n"},

    {"count",            (PyCFunction)Counter_count,            METH_NOARGS,
     "count() -> int\n\n"
     "Run Arjun preprocessing followed by the Ganak model counter and\n"
     "return the exact model count as a Python integer.\n\n"
     "If the Counter was constructed with ``use_tw=True``, an FPT junction-tree\n"
     "DP on the incidence-graph tree decomposition is attempted first; it runs\n"
     "whenever the treewidth is at most ``tw_max`` and the projection set\n"
     "covers all variables.  The DP returns a double-precision real value, so\n"
     "counts above 2**53 lose exact-integer precision via this path.  When\n"
     "treewidth exceeds the cap, falls back to the standard DPLL pipeline.\n\n"
     "count() may only be called once per Counter instance.  Calling it\n"
     "a second time raises RuntimeError.  To count a modified formula,\n"
     "create a new Counter and add the desired clauses to it.\n"},

    {"new_vars",         (PyCFunction)Counter_new_vars,         METH_VARARGS,
     "new_vars(n)\n\n"
     "Declare *n* new variables.  Equivalent to extending the variable\n"
     "universe without adding any clauses.  Useful when you want to count\n"
     "over a known number of variables without adding tautological clauses.\n"},

    {"nof_vars",         (PyCFunction)Counter_nof_vars,         METH_NOARGS,
     "nof_vars() -> int\n\nReturn the number of variables seen so far.\n"},

    {"nof_clauses",      (PyCFunction)Counter_nof_clauses,      METH_NOARGS,
     "nof_clauses() -> int\n\nReturn the number of clauses added so far.\n"},

    {nullptr, nullptr, 0, nullptr}
};

// -------------------------------------------------------------------------
// Type object
// -------------------------------------------------------------------------

static PyTypeObject counter_type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "pyganak.Counter",                           /* tp_name */
    sizeof(Counter),                             /* tp_basicsize */
    0,                                           /* tp_itemsize */
    (destructor)Counter_dealloc,                 /* tp_dealloc */
    0,                                           /* tp_vectorcall_offset */
    nullptr,                                     /* tp_getattr */
    nullptr,                                     /* tp_setattr */
    nullptr,                                     /* tp_as_async */
    nullptr,                                     /* tp_repr */
    nullptr,                                     /* tp_as_number */
    nullptr,                                     /* tp_as_sequence */
    nullptr,                                     /* tp_as_mapping */
    nullptr,                                     /* tp_hash */
    nullptr,                                     /* tp_call */
    nullptr,                                     /* tp_str */
    nullptr,                                     /* tp_getattro */
    nullptr,                                     /* tp_setattro */
    nullptr,                                     /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                          /* tp_flags */
    "Counter(verbose=0, seed=0, use_tw=False, tw_max=20)\n\n"
    "Exact model counter backed by the Ganak solver.\n\n"
    "If ``use_tw=True`` is passed, count() first attempts the FPT\n"
    "junction-tree DP on the incidence-graph tree decomposition; it runs\n"
    "whenever the treewidth is at most ``tw_max`` and there is no strict\n"
    "projection set, otherwise it falls back to the standard DPLL pipeline.\n"
    "The DP is double-precision real, so it is only safe for #SAT below 2**53.\n\n"
    "Usage example::\n\n"
    "    from pyganak import Counter\n"
    "    c = Counter()\n"
    "    c.add_clause([1, 2])    # x1 OR x2\n"
    "    c.add_clause([-1, 2])   # NOT x1 OR x2\n"
    "    print(c.count())        # prints 2\n",
                                                 /* tp_doc */
    nullptr,                                     /* tp_traverse */
    nullptr,                                     /* tp_clear */
    nullptr,                                     /* tp_richcompare */
    0,                                           /* tp_weaklistoffset */
    nullptr,                                     /* tp_iter */
    nullptr,                                     /* tp_iternext */
    counter_methods,                             /* tp_methods */
    nullptr,                                     /* tp_members */
    nullptr,                                     /* tp_getset */
    nullptr,                                     /* tp_base */
    nullptr,                                     /* tp_dict */
    nullptr,                                     /* tp_descr_get */
    nullptr,                                     /* tp_descr_set */
    0,                                           /* tp_dictoffset */
    (initproc)Counter_init,                      /* tp_init */
    nullptr,                                     /* tp_alloc */
    Counter_new,                                 /* tp_new */
};

// =========================================================================
// WeightedCounter — like Counter but with per-literal MPFR weights.
// Accepts weights as Python floats (doubles); uses MPFR internally.
// count() returns a Python float (the MPFR result converted to double).
// =========================================================================

struct WCounterState {
    int verbose = 0;
    uint64_t seed = 0;
    uint32_t max_var = 0;
    mpfr_prec_t prec = 128;   // default 128-bit precision
    std::vector<std::vector<CMSat::Lit>> clauses;
    std::vector<uint32_t> sampling_set;
    bool sampling_set_given = false;
    std::vector<std::pair<CMSat::Lit, double>> weights;
    bool use_tw = false;
    int  tw_max = 20;
    bool counted = false;
};

typedef struct {
    PyObject_HEAD
    WCounterState* state;
} WCounter;

// -------------------------------------------------------------------------
// WeightedCounter helpers (reuse parse_clause / lit_from_py_long above)
// -------------------------------------------------------------------------

static std::unique_ptr<CMSat::Field> do_count_weighted(
    int verbose, uint64_t seed, uint32_t max_var, mpfr_prec_t prec,
    const std::vector<std::vector<CMSat::Lit>>& clauses,
    const std::vector<uint32_t>& sampl_set, bool ss_given,
    const std::vector<std::pair<CMSat::Lit, double>>& weights)
{
    std::unique_ptr<CMSat::FieldGen> fg = std::make_unique<ArjunNS::FGenMpfr>(prec);

    if (max_var == 0) return fg->one();

    ArjunNS::SimplifiedCNF cnf(fg);
    cnf.new_vars(max_var);
    for (const auto& cl : clauses) cnf.add_clause(cl);

    if (!weights.empty()) {
        cnf.set_weighted(true);
        for (const auto& [lit, w] : weights)
            cnf.set_lit_weight(lit, ArjunNS::FMpfr(w, prec));
    }

    if (ss_given) {
        cnf.set_sampl_vars(sampl_set);
    } else {
        std::vector<uint32_t> all_vars;
        all_vars.reserve(max_var);
        for (uint32_t i = 0; i < max_var; i++) all_vars.push_back(i);
        cnf.set_sampl_vars(all_vars);
    }

    ArjunNS::Arjun arjun;
    arjun.set_verb(0);
    ArjunNS::Arjun::ElimToFileConf etof_conf;
    ArjunNS::SimpConf simp_conf;
    arjun.standalone_minimize_indep(cnf, /*all_indep=*/false);
    arjun.standalone_elim_to_file(cnf, etof_conf, simp_conf);

    GanakInt::CounterConfiguration conf;
    conf.verb = verbose;
    conf.seed = seed;
    Ganak counter(conf, fg);
    setup_ganak(cnf, counter);

    const auto& mw = cnf.get_multiplier_weight();
    auto cnt = mw->dup();
    if (!mw->is_zero()) *cnt *= *counter.count();
    return cnt;
}

// -------------------------------------------------------------------------
// WeightedCounter.__new__ / __init__ / __dealloc__
// -------------------------------------------------------------------------

static PyObject* WCounter_new(PyTypeObject* type, PyObject* /*args*/, PyObject* /*kwds*/) {
    WCounter* self = (WCounter*)type->tp_alloc(type, 0);
    if (self) {
        self->state = new (std::nothrow) WCounterState();
        if (!self->state) { Py_DECREF(self); return PyErr_NoMemory(); }
    }
    return (PyObject*)self;
}

static int WCounter_init(WCounter* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"verbose", "seed", "prec",
                                   "use_tw", "tw_max", nullptr};
    int verbose = 0;
    unsigned long long seed = 0;
    unsigned long prec = 128;
    int use_tw = 0;
    int tw_max = 20;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iKkpi",
                                      const_cast<char**>(kwlist),
                                      &verbose, &seed, &prec, &use_tw, &tw_max))
        return -1;
    if (prec < 2) {
        PyErr_SetString(PyExc_ValueError, "prec must be at least 2 bits");
        return -1;
    }
    if (tw_max < 0) {
        PyErr_SetString(PyExc_ValueError, "tw_max must be non-negative");
        return -1;
    }
    self->state->verbose = verbose;
    self->state->seed = (uint64_t)seed;
    self->state->prec = (mpfr_prec_t)prec;
    self->state->use_tw = (use_tw != 0);
    self->state->tw_max = tw_max;
    return 0;
}

static void WCounter_dealloc(WCounter* self) {
    delete self->state;
    self->state = nullptr;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// -------------------------------------------------------------------------
// WeightedCounter.add_clause / add_clauses / set_sampling_set / new_vars
// -------------------------------------------------------------------------

static PyObject* WCounter_add_clause(WCounter* self, PyObject* args) {
    PyObject* clause_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &clause_obj)) return nullptr;
    std::vector<CMSat::Lit> lits;
    if (!parse_clause(clause_obj, lits, self->state->max_var)) return nullptr;
    self->state->clauses.push_back(std::move(lits));
    Py_RETURN_NONE;
}

static PyObject* WCounter_add_clauses(WCounter* self, PyObject* args) {
    PyObject* clauses_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &clauses_obj)) return nullptr;
    PyObject* outer_iter = PyObject_GetIter(clauses_obj);
    if (!outer_iter) return nullptr;
    PyObject* clause_obj;
    while ((clause_obj = PyIter_Next(outer_iter)) != nullptr) {
        std::vector<CMSat::Lit> lits;
        bool ok = parse_clause(clause_obj, lits, self->state->max_var);
        Py_DECREF(clause_obj);
        if (!ok) { Py_DECREF(outer_iter); return nullptr; }
        self->state->clauses.push_back(std::move(lits));
    }
    Py_DECREF(outer_iter);
    if (PyErr_Occurred()) return nullptr;
    Py_RETURN_NONE;
}

static PyObject* WCounter_set_sampling_set(WCounter* self, PyObject* args) {
    PyObject* vars_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &vars_obj)) return nullptr;
    PyObject* iter = PyObject_GetIter(vars_obj);
    if (!iter) return nullptr;
    std::vector<uint32_t> sampl;
    PyObject* item;
    while ((item = PyIter_Next(iter)) != nullptr) {
        long v = PyLong_AsLong(item);
        Py_DECREF(item);
        if (v == -1L && PyErr_Occurred()) { Py_DECREF(iter); return nullptr; }
        if (v <= 0) {
            PyErr_SetString(PyExc_ValueError, "Sampling set variables must be positive integers (1-indexed)");
            Py_DECREF(iter); return nullptr;
        }
        uint32_t var0 = (uint32_t)v - 1;
        sampl.push_back(var0);
        if (var0 + 1 > self->state->max_var) self->state->max_var = var0 + 1;
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) return nullptr;
    self->state->sampling_set = std::move(sampl);
    self->state->sampling_set_given = true;
    Py_RETURN_NONE;
}

// -------------------------------------------------------------------------
// WeightedCounter.set_lit_weight(lit, weight)
//   lit    -- signed 1-indexed integer (positive = positive literal)
//   weight -- Python float (double); stored and later passed to FMpfr
// -------------------------------------------------------------------------

static PyObject* WCounter_set_lit_weight(WCounter* self, PyObject* args) {
    int lit_val = 0;
    double weight = 0.0;
    if (!PyArg_ParseTuple(args, "id", &lit_val, &weight)) return nullptr;
    if (lit_val == 0) {
        PyErr_SetString(PyExc_ValueError, "Literal must be nonzero");
        return nullptr;
    }
    uint32_t var = (uint32_t)((lit_val < 0) ? -lit_val : lit_val) - 1;
    bool sign = (lit_val < 0);
    CMSat::Lit lit(var, sign);
    if (var + 1 > self->state->max_var) self->state->max_var = var + 1;
    self->state->weights.emplace_back(lit, weight);
    Py_RETURN_NONE;
}

// -------------------------------------------------------------------------
// WeightedCounter.new_vars / nof_vars / nof_clauses
// -------------------------------------------------------------------------

static PyObject* WCounter_new_vars(WCounter* self, PyObject* args) {
    unsigned int n = 0;
    if (!PyArg_ParseTuple(args, "I", &n)) return nullptr;
    uint32_t new_max = self->state->max_var + n;
    if (new_max < self->state->max_var) {
        PyErr_SetString(PyExc_OverflowError, "new_vars: variable count overflow");
        return nullptr;
    }
    self->state->max_var = new_max;
    Py_RETURN_NONE;
}

static PyObject* WCounter_nof_vars(WCounter* self, PyObject*) {
    return PyLong_FromUnsignedLong((unsigned long)self->state->max_var);
}

static PyObject* WCounter_nof_clauses(WCounter* self, PyObject*) {
    return PyLong_FromSize_t(self->state->clauses.size());
}

// -------------------------------------------------------------------------
// WeightedCounter.count() -> float
// -------------------------------------------------------------------------

static PyObject* WCounter_count(WCounter* self, PyObject*) {
    if (self->state->counted) {
        PyErr_SetString(PyExc_RuntimeError, "count() may only be called once per WeightedCounter instance");
        return nullptr;
    }
    self->state->counted = true;

    const int       verbose  = self->state->verbose;
    const uint64_t  seed     = self->state->seed;
    const uint32_t  max_var  = self->state->max_var;
    const mpfr_prec_t prec   = self->state->prec;
    const bool      ss_given = self->state->sampling_set_given;
    const bool      use_tw   = self->state->use_tw;
    const int       tw_max   = self->state->tw_max;
    auto clauses_copy   = self->state->clauses;
    auto sampl_set_copy = self->state->sampling_set;
    auto weights_copy   = self->state->weights;

    // Materialise per-variable pos/neg weights as complex<double> for the
    // (real-only) tw DP. Anything the user didn't set stays at 1.0.
    std::vector<std::complex<double>> pos_w(max_var, std::complex<double>(1.0, 0.0));
    std::vector<std::complex<double>> neg_w(max_var, std::complex<double>(1.0, 0.0));
    for (const auto& [lit, w] : weights_copy) {
        uint32_t v = lit.var();
        if (v >= max_var) continue;
        if (lit.sign()) neg_w[v] = std::complex<double>(w, 0.0);
        else            pos_w[v] = std::complex<double>(w, 0.0);
    }

    std::unique_ptr<CMSat::Field> result_field;
    std::optional<std::complex<double>> tw_result;
    std::string error_msg;
    bool success = false;

    Py_BEGIN_ALLOW_THREADS
    try {
        if (use_tw) {
            tw_result = try_tw_dp(max_var, clauses_copy, sampl_set_copy, ss_given,
                                  tw_max, verbose, &pos_w, &neg_w);
        }
        if (!tw_result) {
            result_field = do_count_weighted(verbose, seed, max_var, prec,
                                             clauses_copy, sampl_set_copy, ss_given,
                                             weights_copy);
        }
        success = true;
    } catch (const std::exception& e) {
        error_msg = e.what();
    } catch (...) {
        error_msg = "Unknown C++ exception in count()";
    }
    Py_END_ALLOW_THREADS

    if (!success) {
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return nullptr;
    }

    if (tw_result) {
        // Weighted counts are real-valued; ignore any tiny imag noise.
        return PyFloat_FromDouble(tw_result->real());
    }

    auto& mpfr_result = static_cast<ArjunNS::FMpfr&>(*result_field);
    double d = mpfr_get_d(mpfr_result.val, MPFR_RNDN);
    return PyFloat_FromDouble(d);
}

// -------------------------------------------------------------------------
// WeightedCounter method table
// -------------------------------------------------------------------------

static PyMethodDef wcounter_methods[] = {
    {"add_clause",       (PyCFunction)WCounter_add_clause,       METH_VARARGS,
     "add_clause(clause)\n\nAdd a clause (iterable of nonzero signed integers).\n"},
    {"add_clauses",      (PyCFunction)WCounter_add_clauses,      METH_VARARGS,
     "add_clauses(clauses)\n\nAdd multiple clauses at once.\n"},
    {"set_sampling_set", (PyCFunction)WCounter_set_sampling_set, METH_VARARGS,
     "set_sampling_set(vars)\n\nSet the projection set (1-indexed variable numbers).\n"},
    {"set_lit_weight",   (PyCFunction)WCounter_set_lit_weight,   METH_VARARGS,
     "set_lit_weight(lit, weight)\n\n"
     "Set the weight of a literal.\n\n"
     "*lit* is a nonzero signed integer (1-indexed, positive = positive literal).\n"
     "*weight* is a Python float (double) — stored internally as an MPFR value\n"
     "at the precision specified in the constructor (default 128 bits).\n\n"
     "Call this for both a literal and its negation to get well-defined weighted\n"
     "counting; unspecified literals default to weight 1.\n\n"
     "Example::\n\n"
     "    c.set_lit_weight( 1, 0.3)   # weight of x1  = 0.3\n"
     "    c.set_lit_weight(-1, 0.7)   # weight of ¬x1 = 0.7\n"},
    {"count",            (PyCFunction)WCounter_count,            METH_NOARGS,
     "count() -> float\n\n"
     "Run Arjun preprocessing followed by Ganak weighted model counting.\n"
     "Returns the exact weighted count as a Python float.\n"
     "The internal computation uses MPFR at the precision set in the constructor.\n"
     "count() may only be called once per instance.\n"},
    {"new_vars",         (PyCFunction)WCounter_new_vars,         METH_VARARGS,
     "new_vars(n)\n\nDeclare n new variables.\n"},
    {"nof_vars",         (PyCFunction)WCounter_nof_vars,         METH_NOARGS,
     "nof_vars() -> int\n\nReturn the number of variables seen so far.\n"},
    {"nof_clauses",      (PyCFunction)WCounter_nof_clauses,      METH_NOARGS,
     "nof_clauses() -> int\n\nReturn the number of clauses added so far.\n"},
    {nullptr, nullptr, 0, nullptr}
};

// -------------------------------------------------------------------------
// WeightedCounter type object
// -------------------------------------------------------------------------

static PyTypeObject wcounter_type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "pyganak.WeightedCounter",                    /* tp_name */
    sizeof(WCounter),                             /* tp_basicsize */
    0,                                            /* tp_itemsize */
    (destructor)WCounter_dealloc,                 /* tp_dealloc */
    0,                                            /* tp_vectorcall_offset */
    nullptr,                                      /* tp_getattr */
    nullptr,                                      /* tp_setattr */
    nullptr,                                      /* tp_as_async */
    nullptr,                                      /* tp_repr */
    nullptr,                                      /* tp_as_number */
    nullptr,                                      /* tp_as_sequence */
    nullptr,                                      /* tp_as_mapping */
    nullptr,                                      /* tp_hash */
    nullptr,                                      /* tp_call */
    nullptr,                                      /* tp_str */
    nullptr,                                      /* tp_getattro */
    nullptr,                                      /* tp_setattro */
    nullptr,                                      /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                           /* tp_flags */
    "WeightedCounter(verbose=0, seed=0, prec=128, use_tw=False, tw_max=20)\n\n"
    "Weighted exact model counter backed by Ganak (MPFR field).\n\n"
    "Weights are supplied as Python floats (doubles) via set_lit_weight().\n"
    "Internally the computation uses MPFR at *prec* bits of precision\n"
    "(default 128).  count() returns a Python float.\n\n"
    "If ``use_tw=True`` is passed, count() first attempts the FPT\n"
    "junction-tree DP on the incidence-graph tree decomposition (double\n"
    "precision; ignores *prec*); it runs whenever the treewidth is at most\n"
    "``tw_max`` and there is no strict projection set.  Falls back to MPFR\n"
    "DPLL otherwise.\n\n"
    "Example::\n\n"
    "    from pyganak import WeightedCounter\n"
    "    c = WeightedCounter()\n"
    "    c.add_clause([1, 2])      # x1 OR x2\n"
    "    c.set_lit_weight( 1, 0.3)\n"
    "    c.set_lit_weight(-1, 0.7)\n"
    "    c.set_lit_weight( 2, 0.4)\n"
    "    c.set_lit_weight(-2, 0.6)\n"
    "    print(c.count())\n",                     /* tp_doc */
    nullptr,                                      /* tp_traverse */
    nullptr,                                      /* tp_clear */
    nullptr,                                      /* tp_richcompare */
    0,                                            /* tp_weaklistoffset */
    nullptr,                                      /* tp_iter */
    nullptr,                                      /* tp_iternext */
    wcounter_methods,                             /* tp_methods */
    nullptr,                                      /* tp_members */
    nullptr,                                      /* tp_getset */
    nullptr,                                      /* tp_base */
    nullptr,                                      /* tp_dict */
    nullptr,                                      /* tp_descr_get */
    nullptr,                                      /* tp_descr_set */
    0,                                            /* tp_dictoffset */
    (initproc)WCounter_init,                      /* tp_init */
    nullptr,                                      /* tp_alloc */
    WCounter_new,                                 /* tp_new */
};

// =========================================================================
// SOPCounter — rank-width DP for quadratic sum-of-powers (Z = Σ ω_r^{f(x)}).
//
// f(x) = c + Σ_v b_v x_v + (r/2) Σ_{(u,v) ∈ E} x_u x_v   (mod r)
//
// The input is structurally distinct from CNF: a graph (interaction graph
// of the SOP) with per-vertex coefficients, not a clause set. We expose it
// as its own class.
// =========================================================================

struct SOPCounterState {
    uint32_t n = 0;
    uint32_t r = 8;
    int32_t  c = 0;
    int  rw_max = 12;
    int  verbose = 0;
    std::vector<int32_t> b;     // per-vertex coefficients, size n
    std::vector<uint64_t> adj;  // adjacency bitmasks, size n
    bool counted = false;
};

typedef struct {
    PyObject_HEAD
    SOPCounterState* state;
} SOPCounterObj;

static PyObject* SOPCounter_new(PyTypeObject* type, PyObject*, PyObject*) {
    SOPCounterObj* self = (SOPCounterObj*)type->tp_alloc(type, 0);
    if (self) {
        self->state = new (std::nothrow) SOPCounterState();
        if (!self->state) { Py_DECREF(self); return PyErr_NoMemory(); }
    }
    return (PyObject*)self;
}

static int SOPCounter_init(SOPCounterObj* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"n", "r", "c", "rw_max", "verbose", nullptr};
    unsigned int n = 0;
    unsigned int r = 8;
    int c = 0;
    int rw_max = 12;
    int verbose = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "I|Iiii",
                                      const_cast<char**>(kwlist),
                                      &n, &r, &c, &rw_max, &verbose))
        return -1;
    if (n == 0 || n > 64) {
        PyErr_SetString(PyExc_ValueError, "n must be in [1, 64]");
        return -1;
    }
    if (r == 0 || (r % 2) != 0) {
        PyErr_SetString(PyExc_ValueError, "r must be a positive even integer");
        return -1;
    }
    self->state->n = n;
    self->state->r = r;
    self->state->c = ((c % (int)r) + (int)r) % (int)r;
    self->state->rw_max = rw_max;
    self->state->verbose = verbose;
    self->state->b.assign(n, 0);
    self->state->adj.assign(n, 0);
    return 0;
}

static void SOPCounter_dealloc(SOPCounterObj* self) {
    delete self->state;
    self->state = nullptr;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* SOPCounter_set_b(SOPCounterObj* self, PyObject* args) {
    PyObject* seq = nullptr;
    if (!PyArg_ParseTuple(args, "O", &seq)) return nullptr;
    PyObject* iter = PyObject_GetIter(seq);
    if (!iter) return nullptr;

    std::vector<int32_t> tmp;
    tmp.reserve(self->state->n);
    PyObject* item;
    while ((item = PyIter_Next(iter)) != nullptr) {
        long v = PyLong_AsLong(item);
        Py_DECREF(item);
        if (v == -1L && PyErr_Occurred()) { Py_DECREF(iter); return nullptr; }
        tmp.push_back((int32_t)v);
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) return nullptr;
    if (tmp.size() != self->state->n) {
        PyErr_Format(PyExc_ValueError,
                     "set_b: expected %u coefficients, got %zu",
                     self->state->n, tmp.size());
        return nullptr;
    }
    const int r = (int)self->state->r;
    for (size_t i = 0; i < tmp.size(); i++)
        self->state->b[i] = ((tmp[i] % r) + r) % r;
    Py_RETURN_NONE;
}

static PyObject* SOPCounter_set_b_v(SOPCounterObj* self, PyObject* args) {
    unsigned int v = 0;
    int b_v = 0;
    if (!PyArg_ParseTuple(args, "Ii", &v, &b_v)) return nullptr;
    if (v >= self->state->n) {
        PyErr_Format(PyExc_IndexError, "vertex %u out of range [0, %u)",
                     v, self->state->n);
        return nullptr;
    }
    const int r = (int)self->state->r;
    self->state->b[v] = ((b_v % r) + r) % r;
    Py_RETURN_NONE;
}

static PyObject* SOPCounter_add_edge(SOPCounterObj* self, PyObject* args) {
    unsigned int u = 0, v = 0;
    if (!PyArg_ParseTuple(args, "II", &u, &v)) return nullptr;
    if (u >= self->state->n || v >= self->state->n) {
        PyErr_Format(PyExc_IndexError,
                     "edge endpoints (%u, %u) out of range [0, %u)",
                     u, v, self->state->n);
        return nullptr;
    }
    if (u == v) {
        PyErr_SetString(PyExc_ValueError, "self-loops are not allowed");
        return nullptr;
    }
    self->state->adj[u] |= (1ULL << v);
    self->state->adj[v] |= (1ULL << u);
    Py_RETURN_NONE;
}

static PyObject* SOPCounter_rank_width(SOPCounterObj* self, PyObject*) {
    GanakInt::RankDecomp d = GanakInt::compute_rank_decomp(
        self->state->n, self->state->adj);
    return PyLong_FromLong((long)d.width);
}

static PyObject* SOPCounter_count(SOPCounterObj* self, PyObject*) {
    if (self->state->counted) {
        PyErr_SetString(PyExc_RuntimeError,
            "count() may only be called once per SOPCounter instance");
        return nullptr;
    }
    self->state->counted = true;

    const uint32_t n      = self->state->n;
    const uint32_t r      = self->state->r;
    const int32_t  c      = self->state->c;
    const int      rw_max = self->state->rw_max;
    auto b_copy   = self->state->b;
    auto adj_copy = self->state->adj;

    std::complex<double> result(0.0, 0.0);
    int rank_width = -1;
    std::string error_msg;
    bool success = false;
    bool exceeded_cap = false;

    Py_BEGIN_ALLOW_THREADS
    try {
        GanakInt::SOPInstance sop;
        sop.n = n;
        sop.r = r;
        sop.c = c;
        sop.b = b_copy;
        sop.adj = adj_copy;
        sop.all_vars_mask = (n == 64) ? ~0ULL : ((1ULL << n) - 1);
        GanakInt::RankDecomp decomp = GanakInt::compute_rank_decomp(n, sop.adj);
        rank_width = decomp.width;
        if (decomp.width <= rw_max) {
            result = GanakInt::sop_count(sop, decomp);
        } else {
            exceeded_cap = true;
        }
        success = true;
    } catch (const std::exception& e) {
        error_msg = e.what();
    } catch (...) {
        error_msg = "Unknown C++ exception in count()";
    }
    Py_END_ALLOW_THREADS

    if (!success) {
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return nullptr;
    }
    if (exceeded_cap) {
        PyErr_Format(PyExc_RuntimeError,
            "rank-width %d exceeds rw_max=%d; raise rw_max or use a different "
            "decomposition", rank_width, rw_max);
        return nullptr;
    }
    return PyComplex_FromDoubles(result.real(), result.imag());
}

static PyMethodDef sopcounter_methods[] = {
    {"set_b",       (PyCFunction)SOPCounter_set_b,       METH_VARARGS,
     "set_b(coefs)\n\nSet all per-vertex coefficients at once. *coefs* must be\n"
     "an iterable of n integers (taken mod r).\n"},
    {"set_b_v",     (PyCFunction)SOPCounter_set_b_v,     METH_VARARGS,
     "set_b_v(v, b_v)\n\nSet the coefficient for a single vertex v (0-indexed,\n"
     "b_v taken mod r).\n"},
    {"add_edge",    (PyCFunction)SOPCounter_add_edge,    METH_VARARGS,
     "add_edge(u, v)\n\nAdd an undirected edge {u, v} to the interaction\n"
     "graph; both endpoints must be in [0, n) and u != v.\n"},
    {"rank_width",  (PyCFunction)SOPCounter_rank_width,  METH_NOARGS,
     "rank_width() -> int\n\nReturn the rank-width of the current interaction\n"
     "graph (computed by the same greedy bipartition used by count()).\n"},
    {"count",       (PyCFunction)SOPCounter_count,       METH_NOARGS,
     "count() -> complex\n\nRun the Fourier-mode rank-width DP and return the\n"
     "complex amplitude Z = sum_x omega_r^{f(x)}. Raises RuntimeError if the\n"
     "interaction graph's rank-width exceeds rw_max. Only callable once.\n"},
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject sopcounter_type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "pyganak.SOPCounter",                         /* tp_name */
    sizeof(SOPCounterObj),                        /* tp_basicsize */
    0,                                            /* tp_itemsize */
    (destructor)SOPCounter_dealloc,               /* tp_dealloc */
    0, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    Py_TPFLAGS_DEFAULT,                           /* tp_flags */
    "SOPCounter(n, r=8, c=0, rw_max=12, verbose=0)\n\n"
    "Rank-width FPT DP for quadratic sum-of-powers (Z_r-valued WMC).\n\n"
    "Computes Z = sum_{x in {0,1}^n} omega_r^{f(x)} where\n"
    "  f(x) = c + sum_v b_v * x_v + (r/2) * sum_{(u,v) in E} x_u * x_v   (mod r)\n"
    "via Algorithm 1 (Fourier-mode, a=1) of \"WMC for Quantum Simulation\n"
    "Also Breaks the Treewidth Barrier\". Time O(n * r * 4^k) where k is the\n"
    "rank-width of the interaction graph. The internal adjacency uses a\n"
    "64-bit mask per vertex, so n <= 64.\n\n"
    "Example (qc_2qubit, n=2, r=8, edge (0,1), all b_v = 0; Z = 2)::\n\n"
    "    from pyganak import SOPCounter\n"
    "    s = SOPCounter(n=2, r=8)\n"
    "    s.add_edge(0, 1)\n"
    "    print(s.count())   # (2+0j)\n",
    nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    sopcounter_methods,                           /* tp_methods */
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, 0,
    (initproc)SOPCounter_init,                    /* tp_init */
    nullptr,
    SOPCounter_new,                               /* tp_new */
};

// -------------------------------------------------------------------------
// Module definition
// -------------------------------------------------------------------------

static PyModuleDef pyganak_module = {
    PyModuleDef_HEAD_INIT,
    "pyganak",
    "Python bindings for Ganak, an exact model counter.\n\n"
    "Ganak supports exact projected model counting with Arjun preprocessing.\n"
    "See Counter for the main API.\n",
    -1,
    nullptr,  /* module-level methods */
};

PyMODINIT_FUNC PyInit_pyganak(void) {
    if (PyType_Ready(&counter_type) < 0) return nullptr;
    if (PyType_Ready(&wcounter_type) < 0) return nullptr;
    if (PyType_Ready(&sopcounter_type) < 0) return nullptr;

    PyObject* mod = PyModule_Create(&pyganak_module);
    if (!mod) return nullptr;

    Py_INCREF(&counter_type);
    if (PyModule_AddObject(mod, "Counter", (PyObject*)&counter_type) < 0) {
        Py_DECREF(&counter_type);
        Py_DECREF(mod);
        return nullptr;
    }

    Py_INCREF(&wcounter_type);
    if (PyModule_AddObject(mod, "WeightedCounter", (PyObject*)&wcounter_type) < 0) {
        Py_DECREF(&wcounter_type);
        Py_DECREF(mod);
        return nullptr;
    }

    Py_INCREF(&sopcounter_type);
    if (PyModule_AddObject(mod, "SOPCounter", (PyObject*)&sopcounter_type) < 0) {
        Py_DECREF(&sopcounter_type);
        Py_DECREF(mod);
        return nullptr;
    }

    if (PyModule_AddStringConstant(mod, "__version__", PYGANAK_VERSION) < 0) {
        Py_DECREF(mod);
        return nullptr;
    }
    if (PyModule_AddStringConstant(mod, "VERSION", PYGANAK_VERSION) < 0) {
        Py_DECREF(mod);
        return nullptr;
    }

    return mod;
}
