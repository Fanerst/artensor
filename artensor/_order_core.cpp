#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr double LOG2_10 = 3.32192809488736234787;

struct RNG {
    uint64_t s0;
    uint64_t s1;

    explicit RNG(uint64_t seed)
        : s0((232342352345ULL + seed) | 144115188075855872ULL),
          s1((435243623436ULL + seed) | 9007199254740992ULL) {
        for (int i = 0; i < 5; ++i) next_u64();
    }

    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t next_u64() {
        s1 ^= s0;
        s0 = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        s1 = rotl(s1, 37);
        return s0 + s1;
    }

    double uniform() {
        return static_cast<double>(next_u64() >> 11) *
               (1.0 / 9007199254740992.0);
    }
};

double log2sum2(double a, double b) {
    if (std::isinf(a) && a < 0) return b;
    if (std::isinf(b) && b < 0) return a;
    const double hi = std::max(a, b);
    const double lo = std::min(a, b);
    return hi + std::log2(1.0 + std::exp2(lo - hi));
}

double log2sum3(double a, double b, double c) {
    return log2sum2(log2sum2(a, b), c);
}

double log10sumexp2(const std::vector<double>& values) {
    if (values.empty()) return -std::numeric_limits<double>::infinity();
    const double maximum = *std::max_element(values.begin(), values.end());
    if (std::isinf(maximum)) return maximum / LOG2_10;
    double accum = 0.0;
    for (double value : values) accum += std::exp2(value - maximum);
    return maximum / LOG2_10 + std::log10(accum);
}

double log10sumexp2(double a, double b) {
    return log10sumexp2(std::vector<double>{a, b});
}

struct Network {
    int n = 0;
    int m = 0;
    std::vector<std::vector<int>> tensor_bonds;
    std::vector<std::vector<int>> bond_tensors;
    std::vector<double> logdims;
    std::vector<char> open;
    std::vector<char> final_tensor;
    double log2_max_bitstrings = 0.0;
};

PyObject* get_item(PyObject* mapping, PyObject* key) {
    PyObject* value = PyObject_GetItem(mapping, key);
    if (!value) throw std::runtime_error("failed to read tensor-network mapping");
    return value;
}

Network parse_network(PyObject* tn) {
    Network net;
    PyObject* tensor_bonds = PyObject_GetAttrString(tn, "tensor_bonds");
    PyObject* bond_dims = PyObject_GetAttrString(tn, "bond_dims");
    PyObject* final_qubits = PyObject_GetAttrString(tn, "final_qubits");
    PyObject* log2_max = PyObject_GetAttrString(tn, "log2_max_bitstring");
    PyObject* output_bonds = PyObject_GetAttrString(tn, "output_bonds");
    if (!tensor_bonds || !bond_dims || !final_qubits || !log2_max || !output_bonds) {
        Py_XDECREF(tensor_bonds);
        Py_XDECREF(bond_dims);
        Py_XDECREF(final_qubits);
        Py_XDECREF(log2_max);
        Py_XDECREF(output_bonds);
        throw std::runtime_error("invalid AbstractTensorNetwork object");
    }

    try {
        net.n = static_cast<int>(PyMapping_Size(tensor_bonds));
        net.m = static_cast<int>(PyMapping_Size(bond_dims));
        if (net.n < 1) throw std::runtime_error("tensor network is empty");
        if (net.m < 0) throw std::runtime_error("invalid bond_dims mapping");
        net.log2_max_bitstrings = PyFloat_AsDouble(log2_max);
        if (PyErr_Occurred()) throw std::runtime_error("invalid max_bitstrings");

        PyObject* bond_to_id = PyDict_New();
        if (!bond_to_id) throw std::runtime_error("failed to allocate bond map");
        net.logdims.reserve(net.m);
        net.open.reserve(net.m);

        PyObject* keys = PyMapping_Keys(bond_dims);
        if (!keys) {
            Py_DECREF(bond_to_id);
            throw std::runtime_error("failed to enumerate bond_dims");
        }
        PyObject* keys_fast = PySequence_Fast(keys, "bond_dims keys must be a sequence");
        Py_DECREF(keys);
        if (!keys_fast) {
            Py_DECREF(bond_to_id);
            throw std::runtime_error("failed to enumerate bond_dims");
        }
        const Py_ssize_t key_count = PySequence_Fast_GET_SIZE(keys_fast);
        for (Py_ssize_t index = 0; index < key_count; ++index) {
            PyObject* key = PySequence_Fast_GET_ITEM(keys_fast, index);
            PyObject* dim_obj = get_item(bond_dims, key);
            const double dim = PyFloat_AsDouble(dim_obj);
            Py_DECREF(dim_obj);
            if (PyErr_Occurred() || !(dim > 0.0)) {
                Py_DECREF(keys_fast);
                Py_DECREF(bond_to_id);
                throw std::runtime_error("bond dimensions must be positive numbers");
            }
            PyObject* id_obj = PyLong_FromSsize_t(index);
            if (!id_obj || PyDict_SetItem(bond_to_id, key, id_obj) < 0) {
                Py_XDECREF(id_obj);
                Py_DECREF(keys_fast);
                Py_DECREF(bond_to_id);
                throw std::runtime_error("failed to construct bond map");
            }
            Py_DECREF(id_obj);
            net.logdims.push_back(std::log2(dim));
            const int is_output = PySequence_Contains(output_bonds, key);
            if (is_output < 0) {
                Py_DECREF(keys_fast);
                Py_DECREF(bond_to_id);
                throw std::runtime_error("failed to read output_bonds");
            }
            net.open.push_back(is_output != 0);
        }

        net.tensor_bonds.resize(net.n);
        net.bond_tensors.resize(net.m);
        for (int tensor = 0; tensor < net.n; ++tensor) {
            PyObject* tensor_key = PyLong_FromLong(tensor);
            if (!tensor_key) {
                Py_DECREF(keys_fast);
                Py_DECREF(bond_to_id);
                throw std::runtime_error("failed to allocate tensor ID");
            }
            PyObject* bonds_obj = PyObject_GetItem(tensor_bonds, tensor_key);
            Py_DECREF(tensor_key);
            if (!bonds_obj) {
                PyErr_Clear();
                Py_DECREF(keys_fast);
                Py_DECREF(bond_to_id);
                throw std::runtime_error(
                    "tensor IDs must be consecutive integers starting at zero"
                );
            }
            PyObject* bonds_fast = PySequence_Fast(
                bonds_obj, "tensor bond lists must be sequences"
            );
            Py_DECREF(bonds_obj);
            if (!bonds_fast) {
                Py_DECREF(keys_fast);
                Py_DECREF(bond_to_id);
                throw std::runtime_error("tensor bond lists must be sequences");
            }
            const Py_ssize_t count = PySequence_Fast_GET_SIZE(bonds_fast);
            auto& target = net.tensor_bonds[tensor];
            target.reserve(count);
            for (Py_ssize_t j = 0; j < count; ++j) {
                PyObject* label = PySequence_Fast_GET_ITEM(bonds_fast, j);
                PyObject* id_obj = PyDict_GetItemWithError(bond_to_id, label);
                if (!id_obj) {
                    Py_DECREF(bonds_fast);
                    Py_DECREF(keys_fast);
                    Py_DECREF(bond_to_id);
                    throw std::runtime_error("tensor uses a bond missing from bond_dims");
                }
                const int bond = static_cast<int>(PyLong_AsLong(id_obj));
                target.push_back(bond);
                net.bond_tensors[bond].push_back(tensor);
            }
            std::sort(target.begin(), target.end());
            target.erase(std::unique(target.begin(), target.end()), target.end());
            Py_DECREF(bonds_fast);
        }
        Py_DECREF(keys_fast);
        Py_DECREF(bond_to_id);

        for (int bond = 0; bond < net.m; ++bond) {
            auto& incident = net.bond_tensors[bond];
            std::sort(incident.begin(), incident.end());
            incident.erase(std::unique(incident.begin(), incident.end()), incident.end());
            net.open[bond] = net.open[bond] || incident.size() <= 1;
        }

        net.final_tensor.resize(net.n, false);
        PyObject* final_fast = PySequence_Fast(
            final_qubits, "final_qubits must be a sequence"
        );
        if (!final_fast) throw std::runtime_error("final_qubits must be a sequence");
        const Py_ssize_t final_count = PySequence_Fast_GET_SIZE(final_fast);
        for (Py_ssize_t i = 0; i < final_count; ++i) {
            const long tensor = PyLong_AsLong(PySequence_Fast_GET_ITEM(final_fast, i));
            if (PyErr_Occurred() || tensor < 0 || tensor >= net.n) {
                Py_DECREF(final_fast);
                throw std::runtime_error("final_qubits contains an invalid tensor ID");
            }
            net.final_tensor[tensor] = true;
        }
        Py_DECREF(final_fast);
    } catch (...) {
        Py_DECREF(tensor_bonds);
        Py_DECREF(bond_dims);
        Py_DECREF(final_qubits);
        Py_DECREF(log2_max);
        Py_DECREF(output_bonds);
        throw;
    }

    Py_DECREF(tensor_bonds);
    Py_DECREF(bond_dims);
    Py_DECREF(final_qubits);
    Py_DECREF(log2_max);
    Py_DECREF(output_bonds);
    return net;
}

struct PairCandidate {
    double cost;
    uint64_t tie;
    int i;
    int j;
    uint64_t generation_i;
    uint64_t generation_j;
};

struct CandidateGreater {
    bool operator()(const PairCandidate& lhs, const PairCandidate& rhs) const {
        if (lhs.cost != rhs.cost) return lhs.cost > rhs.cost;
        return lhs.tie > rhs.tie;
    }
};

struct PairInfo {
    double all_log = 0.0;
    double output_log = 0.0;
    double input1_log = 0.0;
    double input2_log = 0.0;
    double factor = 0.0;
    std::vector<int> result_bonds;
    std::vector<int> free1;
    std::vector<int> free2;
};

struct GreedyRun {
    std::vector<std::pair<int, int>> order;
    double tc = -std::numeric_limits<double>::infinity();
    double sc = 0.0;
};

struct VectorHash {
    std::size_t operator()(const std::vector<int>& values) const noexcept {
        std::size_t seed = values.size();
        for (int value : values) {
            seed ^= static_cast<std::size_t>(value) + 0x9e3779b9U +
                    (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};

class GreedySearch {
public:
    GreedySearch(
        const Network& network,
        uint64_t seed,
        int strategy,
        int cost_id = -1,
        bool thermal_chooser = false
    )
        : net(network), rng(seed), strategy(strategy), cost_id(cost_id),
          thermal(thermal_chooser),
          active(net.n, true), generation(net.n, 0), final_count(net.n, 0),
          legs(net.n), neighbors(net.n), bond_nodes(net.m) {
        for (int tensor = 0; tensor < net.n; ++tensor) {
            final_count[tensor] = net.final_tensor[tensor] ? 1 : 0;
            legs[tensor].insert(
                net.tensor_bonds[tensor].begin(), net.tensor_bonds[tensor].end()
            );
            for (int bond : net.tensor_bonds[tensor]) bond_nodes[bond].insert(tensor);
        }
        remaining = net.n;
        n_total = static_cast<double>(net.n);
        global_alpha = rng.uniform();
    }

    GreedyRun run() {
        std::vector<double> step_tcs;
        double peak_sc = 0.0;
        for (int tensor = 0; tensor < net.n; ++tensor) {
            peak_sc = std::max(peak_sc, node_size_log(tensor));
        }

        // Match opt_einsum/OME's first greedy phase: identical index sets are
        // Hadamard-multiplied before constructing the (potentially very large)
        // candidate graph.
        std::unordered_map<std::vector<int>, int, VectorHash> identical;
        for (int tensor = 0; tensor < net.n; ++tensor) {
            const auto found = identical.find(net.tensor_bonds[tensor]);
            if (found == identical.end()) {
                identical.emplace(net.tensor_bonds[tensor], tensor);
            } else {
                auto [tc, sc] = contract(found->second, tensor);
                step_tcs.push_back(tc);
                peak_sc = std::max(peak_sc, sc);
            }
        }

        for (int bond = 0; bond < net.m; ++bond) {
            if (bond_nodes[bond].size() < 2) continue;
            std::vector<int> incident(bond_nodes[bond].begin(), bond_nodes[bond].end());
            for (std::size_t i = 0; i < incident.size(); ++i) {
                for (std::size_t j = i + 1; j < incident.size(); ++j) {
                    neighbors[incident[i]].insert(incident[j]);
                    neighbors[incident[j]].insert(incident[i]);
                }
            }
        }
        for (int tensor = 0; tensor < net.n; ++tensor) {
            if (!active[tensor]) continue;
            for (int other : neighbors[tensor]) {
                if (tensor < other) push_candidate(tensor, other);
            }
        }

        while (!queue.empty()) {
            std::optional<PairCandidate> selected = pop_candidate();
            if (!selected) continue;
            PairCandidate candidate = *selected;
            auto [tc, sc] = contract(candidate.i, candidate.j);
            step_tcs.push_back(tc);
            peak_sc = std::max(peak_sc, sc);
        }

        std::vector<int> live;
        for (int i = 0; i < net.n; ++i) if (active[i]) live.push_back(i);
        if (!live.empty()) {
            const int source = live.front();
            for (std::size_t k = 1; k < live.size(); ++k) {
                auto [tc, sc] = contract(source, live[k]);
                step_tcs.push_back(tc);
                peak_sc = std::max(peak_sc, sc);
            }
        }

        GreedyRun result;
        result.order = std::move(order);
        result.tc = log10sumexp2(step_tcs);
        result.sc = peak_sc;
        return result;
    }

private:
    const Network& net;
    RNG rng;
    int strategy;
    int cost_id;
    bool thermal;
    std::vector<char> active;
    std::vector<uint64_t> generation;
    std::vector<int> final_count;
    std::vector<std::unordered_set<int>> legs;
    std::vector<std::unordered_set<int>> neighbors;
    std::vector<std::unordered_set<int>> bond_nodes;
    std::priority_queue<PairCandidate, std::vector<PairCandidate>, CandidateGreater> queue;
    std::vector<std::pair<int, int>> order;
    int remaining = 0;
    double n_total = 0.0;
    double global_alpha = 0.0;

    double sum_dims(const std::unordered_set<int>& bonds) const {
        double result = 0.0;
        for (int bond : bonds) result += net.logdims[bond];
        return result;
    }

    double node_size_log(int node) const {
        return sum_dims(legs[node]) +
               std::min(net.log2_max_bitstrings, static_cast<double>(final_count[node]));
    }

    PairInfo analyze(int i, int j) const {
        PairInfo info;
        info.input1_log = node_size_log(i);
        info.input2_log = node_size_log(j);
        info.factor = std::min(
            net.log2_max_bitstrings,
            static_cast<double>(final_count[i] + final_count[j])
        );
        info.result_bonds.reserve(legs[i].size() + legs[j].size());
        for (int bond : legs[i]) {
            info.all_log += net.logdims[bond];
            if (legs[j].find(bond) == legs[j].end()) {
                info.free1.push_back(bond);
                info.result_bonds.push_back(bond);
            } else if (!net.open[bond] && bond_nodes[bond].size() <= 2) {
            } else {
                info.result_bonds.push_back(bond);
            }
        }
        for (int bond : legs[j]) {
            if (legs[i].find(bond) == legs[i].end()) {
                info.all_log += net.logdims[bond];
                info.free2.push_back(bond);
                info.result_bonds.push_back(bond);
            }
        }
        info.output_log = info.factor;
        for (int bond : info.result_bonds) info.output_log += net.logdims[bond];
        return info;
    }

    static double safe_exp2(double exponent) {
        return std::exp2(std::min(1020.0, exponent));
    }

    double portfolio_cost(const PairInfo& info) {
        const double size12 = safe_exp2(info.output_log);
        const double size1 = safe_exp2(info.input1_log);
        const double size2 = safe_exp2(info.input2_log);
        const double cooling = std::max(
            1e-12, 1.0 - (n_total - static_cast<double>(remaining)) / n_total
        );
        const double ratio =
            (std::log2(size2 + 2.0) + std::log2(size1 + 2.0)) /
            std::max(1e-12, std::log2(size12 + 2.0)) / cooling;
        double adjustment = 0.0;
        for (int bond : info.free1) {
            if (!net.open[bond]) adjustment -= bond_nodes[bond].size() * 0.15;
        }
        for (int bond : info.free2) {
            if (!net.open[bond]) adjustment -= bond_nodes[bond].size() * 0.15;
        }

        double cost = size12;
        switch (cost_id) {
            case 0: {  // balanced Boltzmann
                cost = size12 - (size1 + size2) + rng.uniform() - rng.uniform() * ratio;
                cost += adjustment;
                const double alpha = 0.8 + 0.2 * rng.uniform();
                const double boltzmann_cost =
                    size12 + alpha * (size1 + size2) + std::max(size1, size2) -
                    rng.uniform() * ratio;
                const double temperature = (size12 != 0.0 ? size12 + rng.uniform() : 0.0) + 1e-12;
                const double weight = std::exp(-boltzmann_cost / temperature);
                cost = 0.2 * cost / (size12 + 1e-12) + 0.8 * weight;
                break;
            }
            case 1: {  // Boltzmann
                const double alpha = 0.8 + 0.2 * rng.uniform();
                cost = size12 + alpha * (size1 + size2) + std::max(size1, size2) -
                       rng.uniform() * ratio;
                const double temperature = (size12 != 0.0 ? size12 + rng.uniform() : 0.0) + 1e-12;
                cost = std::exp(-cost / temperature);
                break;
            }
            case 2: {  // maximally skewed
                cost = size12 - rng.uniform() * ratio;
                const double alpha = (n_total - static_cast<double>(remaining)) / n_total;
                cost += alpha * size12 * rng.uniform();
                cost /= (1.0 - alpha) * std::max(size1, size2) *
                            std::abs(size1 - size2) +
                        rng.uniform() + size1 * size2 + 1e-12;
                break;
            }
            case 3: {  // anti-balanced
                cost = size12 - (size1 + size2);
                cost -= std::abs(size1 - size2) * 0.15;
                cost -= std::max(size1, size2) * 0.15;
                cost += rng.uniform() - rng.uniform() * ratio;
                cost -= adjustment;
                cost /= size12 + rng.uniform() + 1e-12;
                break;
            }
            case 4: {  // skew-balanced
                cost = size12 - (size1 + size2);
                cost += std::abs(size1 - size2) * 0.15;
                cost += rng.uniform() - rng.uniform() * ratio;
                cost += adjustment;
                cost /= size12 + rng.uniform() + 1e-12;
                break;
            }
            case 5:  // logarithmic
                cost = std::log2(size12 + 2.0) /
                       std::log2((size1 + size2) * 0.65 + 2.0 + rng.uniform());
                break;
            case 6:  // memory-removed jitter
                cost = size12 - global_alpha * 50.0 * (size1 + size2) -
                       rng.uniform() * ratio + (rng.uniform() - 0.5) * 0.1;
                break;
            case 7: {  // batch-balanced
                cost = size12 - (size1 + size2) + rng.uniform() -
                       rng.uniform() * 0.01 * ratio;
                double degree_adjustment = 0.0;
                for (int bond : info.free1) {
                    if (net.open[bond]) continue;
                    const double degree = std::log2(std::max<std::size_t>(1, bond_nodes[bond].size()));
                    degree_adjustment -= degree * degree * 0.2;
                }
                for (int bond : info.free2) {
                    if (net.open[bond]) continue;
                    const double degree = std::log2(std::max<std::size_t>(1, bond_nodes[bond].size()));
                    degree_adjustment -= degree * degree * 0.2;
                }
                for (int bond : info.result_bonds) {
                    if (net.open[bond]) continue;
                    const double degree = std::log2(
                        std::max<std::size_t>(1, bond_nodes[bond].size())
                    );
                    degree_adjustment += degree * degree * 0.2;
                }
                cost += degree_adjustment;
                const double bc = 10.0 * std::max(size1, size2) - rng.uniform() * ratio;
                const double temperature = (size12 != 0.0 ? size12 + rng.uniform() : 0.0) + 1.0 + 1e-12;
                cost = 0.2 * cost / (size12 + 1e-12) +
                       0.8 * std::exp(-bc / temperature);
                break;
            }
            default:
                break;
        }
        if (!std::isfinite(cost)) return std::numeric_limits<double>::max() / 4.0;
        return cost;
    }

    double pair_cost(int i, int j) {
        PairInfo info = analyze(i, j);
        if (cost_id >= 0) return portfolio_cost(info);
        if (strategy == 0) return info.output_log;
        return info.output_log - info.input1_log - info.input2_log;
    }

    void push_candidate(int i, int j) {
        if (i == j || !active[i] || !active[j]) return;
        if (i > j) std::swap(i, j);
        queue.push(PairCandidate{
            pair_cost(i, j), rng.next_u64(), i, j, generation[i], generation[j]
        });
    }

    bool valid(const PairCandidate& candidate) const {
        return active[candidate.i] && active[candidate.j] &&
               generation[candidate.i] == candidate.generation_i &&
               generation[candidate.j] == candidate.generation_j &&
               neighbors[candidate.i].find(candidate.j) != neighbors[candidate.i].end();
    }

    std::optional<PairCandidate> pop_candidate() {
        if (!thermal) {
            while (!queue.empty()) {
                PairCandidate candidate = queue.top();
                queue.pop();
                if (valid(candidate)) return candidate;
            }
            return std::nullopt;
        }

        const int branch_count = static_cast<int>(rng.next_u64() % 32U) + 1;
        const double relative_temperature = std::pow(
            static_cast<double>(remaining) / n_total,
            4.0 + static_cast<double>(rng.next_u64() % 8U)
        );
        std::vector<PairCandidate> choices;
        choices.reserve(branch_count);
        while (!queue.empty() && static_cast<int>(choices.size()) < branch_count) {
            PairCandidate candidate = queue.top();
            queue.pop();
            if (valid(candidate)) choices.push_back(candidate);
        }
        if (choices.empty()) return std::nullopt;
        if (choices.size() == 1) return choices.front();

        const double minimum = choices.front().cost;
        const double temperature = relative_temperature *
                                   std::max(1.0, std::abs(minimum));
        std::vector<double> cumulative;
        cumulative.reserve(choices.size());
        double total = 0.0;
        for (const PairCandidate& candidate : choices) {
            const double weight = temperature == 0.0
                ? (candidate.cost == minimum ? 1.0 : 0.0)
                : std::exp(-(candidate.cost - minimum) / temperature);
            total += weight;
            cumulative.push_back(total);
        }
        const double draw = rng.uniform() * total;
        std::size_t chosen = static_cast<std::size_t>(
            std::lower_bound(cumulative.begin(), cumulative.end(), draw) -
            cumulative.begin()
        );
        if (chosen >= choices.size()) chosen = choices.size() - 1;
        PairCandidate result = choices[chosen];
        for (std::size_t index = 0; index < choices.size(); ++index) {
            if (index != chosen) queue.push(choices[index]);
        }
        return result;
    }

    std::pair<double, double> contract(int i, int j) {
        if (i > j) std::swap(i, j);
        PairInfo info = analyze(i, j);
        const double tc = info.all_log + info.factor;
        const double sc = info.output_log;
        order.emplace_back(i, j);

        std::unordered_set<int> affected = neighbors[i];
        affected.insert(neighbors[j].begin(), neighbors[j].end());
        affected.erase(i);
        affected.erase(j);
        ++generation[i];
        ++generation[j];
        active[j] = false;
        --remaining;
        final_count[i] += final_count[j];
        final_count[j] = 0;

        std::unordered_set<int> union_bonds = legs[i];
        union_bonds.insert(legs[j].begin(), legs[j].end());
        legs[i].clear();
        for (int bond : union_bonds) {
            bond_nodes[bond].erase(j);
            bond_nodes[bond].insert(i);
            if (!net.open[bond] && bond_nodes[bond].size() == 1) {
                bond_nodes[bond].clear();
            } else {
                legs[i].insert(bond);
            }
        }
        legs[j].clear();

        neighbors[i].clear();
        neighbors[j].clear();
        for (int other : affected) {
            if (!active[other]) continue;
            neighbors[other].erase(j);
            neighbors[other].erase(i);
            bool connected = false;
            for (int bond : legs[i]) {
                if (bond_nodes[bond].find(other) != bond_nodes[bond].end()) {
                    connected = true;
                    break;
                }
            }
            if (connected) {
                neighbors[i].insert(other);
                neighbors[other].insert(i);
                push_candidate(i, other);
            }
        }
        return {tc, sc};
    }
};

bool better_run(const GreedyRun& current, const GreedyRun& best, bool minimize_flops) {
    if (best.order.empty()) return true;
    if (minimize_flops) {
        return current.tc < best.tc || (current.tc == best.tc && current.sc < best.sc);
    }
    return current.sc < best.sc || (current.sc == best.sc && current.tc < best.tc);
}

struct TreeNode {
    int left = -1;
    int right = -1;
    std::vector<uint64_t> tensors;
    std::vector<int> legs;
    int final_count = 0;
    double factor = 0.0;
    double tc = 0.0;
    double sc = 0.0;
    double mc = 0.0;

    bool leaf() const { return left < 0; }
};

struct TreeMetrics {
    double tc;
    double sc;
    double mc;
};

class AnnealTree {
public:
    AnnealTree(
        const Network& network,
        const std::vector<std::pair<int, int>>& initial_order,
        uint64_t seed,
        double target,
        double memory_alpha,
        double space_weight
    ) : net(network), rng(seed), sc_target(target), alpha(memory_alpha),
        sc_weight(space_weight), words((net.n + 63) / 64), nodes(2 * net.n - 1) {
        for (int tensor = 0; tensor < net.n; ++tensor) {
            TreeNode& node = nodes[tensor];
            node.tensors.assign(words, 0);
            node.tensors[tensor / 64] |= uint64_t(1) << (tensor % 64);
            node.legs = net.tensor_bonds[tensor];
            node.final_count = net.final_tensor[tensor] ? 1 : 0;
            node.factor = std::min(
                net.log2_max_bitstrings, static_cast<double>(node.final_count)
            );
            node.sc = node.factor;
            for (int bond : node.legs) node.sc += net.logdims[bond];
        }

        std::vector<int> representatives(net.n);
        std::vector<char> active(net.n, true);
        for (int i = 0; i < net.n; ++i) representatives[i] = i;
        int next = net.n;
        for (const auto& pair : initial_order) {
            const int i = pair.first;
            const int j = pair.second;
            if (i < 0 || i >= net.n || j < 0 || j >= net.n || i == j ||
                !active[i] || !active[j]) {
                throw std::runtime_error("invalid Artensor pairwise order");
            }
            nodes[next] = combine(representatives[i], representatives[j]);
            representatives[i] = next;
            active[j] = false;
            ++next;
        }
        if (next != 2 * net.n - 1) {
            throw std::runtime_error("pairwise order must contain n-1 contractions");
        }
        root = -1;
        for (int i = 0; i < net.n; ++i) {
            if (active[i]) {
                if (root >= 0) throw std::runtime_error("pairwise order is disconnected");
                root = representatives[i];
            }
        }
        best_left.resize(nodes.size());
        best_right.resize(nodes.size());
        snapshot();
        best_metrics = metrics();
        best_score = score(best_metrics);
    }

    TreeMetrics optimize(const std::vector<double>& betas, int niters) {
        for (double beta : betas) {
            for (int iter = 0; iter < niters; ++iter) {
                sweep(root, beta);
                TreeMetrics current = metrics();
                const double current_score = score(current);
                if (current_score < best_score) {
                    best_score = current_score;
                    best_metrics = current;
                    snapshot();
                }
            }
        }
        restore();
        recompute_postorder(root);
        best_metrics = metrics();
        return best_metrics;
    }

    std::vector<std::pair<int, int>> order() const {
        std::vector<std::pair<int, int>> result;
        result.reserve(net.n - 1);
        order_postorder(root, result);
        return result;
    }

private:
    const Network& net;
    RNG rng;
    double sc_target;
    double alpha;
    double sc_weight;
    int words;
    std::vector<TreeNode> nodes;
    int root = -1;
    std::vector<int> best_left;
    std::vector<int> best_right;
    TreeMetrics best_metrics{};
    double best_score = std::numeric_limits<double>::infinity();

    bool contains_all(const std::vector<uint64_t>& mask, int bond) const {
        for (int tensor : net.bond_tensors[bond]) {
            if ((mask[tensor / 64] & (uint64_t(1) << (tensor % 64))) == 0) {
                return false;
            }
        }
        return true;
    }

    TreeNode combine_states(const TreeNode& lhs, const TreeNode& rhs, int left, int right) const {
        TreeNode result;
        result.left = left;
        result.right = right;
        result.tensors.resize(words);
        for (int w = 0; w < words; ++w) {
            result.tensors[w] = lhs.tensors[w] | rhs.tensors[w];
        }
        result.legs.reserve(lhs.legs.size() + rhs.legs.size());
        double all_log = 0.0;
        std::size_t lhs_index = 0;
        std::size_t rhs_index = 0;
        while (lhs_index < lhs.legs.size() || rhs_index < rhs.legs.size()) {
            int bond;
            bool common = false;
            if (rhs_index == rhs.legs.size() ||
                (lhs_index < lhs.legs.size() && lhs.legs[lhs_index] < rhs.legs[rhs_index])) {
                bond = lhs.legs[lhs_index++];
            } else if (lhs_index == lhs.legs.size() ||
                       rhs.legs[rhs_index] < lhs.legs[lhs_index]) {
                bond = rhs.legs[rhs_index++];
            } else {
                bond = lhs.legs[lhs_index++];
                ++rhs_index;
                common = true;
            }
            all_log += net.logdims[bond];
            const bool eliminate = common && !net.open[bond] &&
                                   contains_all(result.tensors, bond);
            if (!eliminate) result.legs.push_back(bond);
        }
        result.final_count = lhs.final_count + rhs.final_count;
        const double combined_factor = lhs.factor + rhs.factor;
        result.factor = std::min(net.log2_max_bitstrings, combined_factor);
        result.tc = all_log + result.factor;
        result.sc = result.factor;
        for (int bond : result.legs) result.sc += net.logdims[bond];
        if (combined_factor > net.log2_max_bitstrings) {
            result.mc = log2sum3(
                lhs.sc - lhs.factor + result.factor,
                rhs.sc - rhs.factor + result.factor,
                result.sc
            );
        } else {
            result.mc = log2sum3(lhs.sc, rhs.sc, result.sc);
        }
        return result;
    }

    TreeNode combine(int left, int right) const {
        return combine_states(nodes[left], nodes[right], left, right);
    }

    TreeMetrics local_metrics(
        const TreeNode& branch,
        const TreeNode& top,
        const TreeNode& a,
        const TreeNode& b,
        const TreeNode& c
    ) const {
        return TreeMetrics{
            log10sumexp2(branch.tc, top.tc),
            std::max({branch.sc, top.sc, a.sc, b.sc, c.sc}),
            log10sumexp2(branch.mc, top.mc),
        };
    }

    double score(const TreeMetrics& value) const {
        double memory_time = value.tc;
        if (alpha > 0.0) {
            const double memory_term = std::log10(alpha) + value.mc;
            const double maximum = std::max(value.tc, memory_term);
            memory_time = maximum + std::log10(
                std::pow(10.0, value.tc - maximum) +
                std::pow(10.0, memory_term - maximum)
            );
        }
        return memory_time + sc_weight * std::log10(2.0) *
               std::max(0.0, value.sc - sc_target);
    }

    void sweep(int node_id, double beta) {
        if (nodes[node_id].leaf()) return;
        int branch_id = -1;
        bool branch_on_left = false;
        const bool can_rotate_left = !nodes[nodes[node_id].left].leaf();
        const bool can_rotate_right = !nodes[nodes[node_id].right].leaf();
        uint64_t proposal = rng.next_u64();
        if (can_rotate_left && can_rotate_right) {
            branch_on_left = proposal % 4 < 2;
            branch_id = branch_on_left ? nodes[node_id].left : nodes[node_id].right;
        } else if (can_rotate_left) {
            branch_id = nodes[node_id].left;
            branch_on_left = true;
        } else if (can_rotate_right) {
            branch_id = nodes[node_id].right;
        }

        if (branch_id >= 0) {
            const TreeNode old_top = nodes[node_id];
            const TreeNode old_branch = nodes[branch_id];
            int a, b, c;
            if (branch_on_left) {
                a = old_branch.left;
                b = old_branch.right;
                c = old_top.right;
            } else {
                a = old_top.left;
                b = old_branch.left;
                c = old_branch.right;
            }
            const TreeMetrics before = local_metrics(
                old_branch, old_top, nodes[a], nodes[b], nodes[c]
            );
            const bool second = (proposal & 1U) != 0;
            int x, y, z;
            if (branch_on_left) {
                if (!second) {
                    x = a;
                    y = c;
                    z = b;
                } else {
                    x = b;
                    y = c;
                    z = a;
                }
            } else {
                if (!second) {
                    x = a;
                    y = c;
                    z = b;
                } else {
                    x = b;
                    y = a;
                    z = c;
                }
            }
            TreeNode proposed_branch = combine_states(nodes[x], nodes[y], x, y);
            TreeNode proposed_top = combine_states(
                proposed_branch, nodes[z], branch_id, z
            );
            const TreeMetrics after = local_metrics(
                proposed_branch, proposed_top, nodes[a], nodes[b], nodes[c]
            );
            const double delta = score(after) - score(before);
            if (delta <= 0.0 || rng.uniform() < std::exp(-beta * delta)) {
                nodes[branch_id] = std::move(proposed_branch);
                nodes[node_id] = std::move(proposed_top);
            }
        }
        sweep(nodes[node_id].left, beta);
        sweep(nodes[node_id].right, beta);
    }

    TreeMetrics metrics() const {
        std::vector<double> tcs;
        std::vector<double> mcs;
        double peak = 0.0;
        tcs.reserve(net.n - 1);
        mcs.reserve(net.n - 1);
        for (const TreeNode& node : nodes) {
            peak = std::max(peak, node.sc);
            if (!node.leaf()) {
                tcs.push_back(node.tc);
                mcs.push_back(node.mc);
            }
        }
        return TreeMetrics{log10sumexp2(tcs), peak, log10sumexp2(mcs)};
    }

    void snapshot() {
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            best_left[i] = nodes[i].left;
            best_right[i] = nodes[i].right;
        }
    }

    void restore() {
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            nodes[i].left = best_left[i];
            nodes[i].right = best_right[i];
        }
    }

    void recompute_postorder(int node_id) {
        if (nodes[node_id].leaf()) return;
        const int left = nodes[node_id].left;
        const int right = nodes[node_id].right;
        recompute_postorder(left);
        recompute_postorder(right);
        nodes[node_id] = combine(left, right);
    }

    int order_postorder(
        int node_id, std::vector<std::pair<int, int>>& result
    ) const {
        if (nodes[node_id].leaf()) return node_id;
        const int left_rep = order_postorder(nodes[node_id].left, result);
        const int right_rep = order_postorder(nodes[node_id].right, result);
        result.emplace_back(left_rep, right_rep);
        return left_rep;
    }
};

std::vector<std::pair<int, int>> parse_order(PyObject* order_obj, int n) {
    PyObject* fast = PySequence_Fast(order_obj, "order must be a sequence");
    if (!fast) throw std::runtime_error("order must be a sequence");
    std::vector<std::pair<int, int>> order;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(fast);
    order.reserve(count);
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* pair_fast = PySequence_Fast(
            PySequence_Fast_GET_ITEM(fast, i), "order entries must be pairs"
        );
        if (!pair_fast || PySequence_Fast_GET_SIZE(pair_fast) != 2) {
            Py_XDECREF(pair_fast);
            Py_DECREF(fast);
            throw std::runtime_error("order entries must be pairs");
        }
        const long left = PyLong_AsLong(PySequence_Fast_GET_ITEM(pair_fast, 0));
        const long right = PyLong_AsLong(PySequence_Fast_GET_ITEM(pair_fast, 1));
        Py_DECREF(pair_fast);
        if (PyErr_Occurred() || left < 0 || left >= n || right < 0 || right >= n) {
            Py_DECREF(fast);
            throw std::runtime_error("order contains an invalid tensor ID");
        }
        order.emplace_back(static_cast<int>(left), static_cast<int>(right));
    }
    Py_DECREF(fast);
    return order;
}

std::vector<double> parse_doubles(PyObject* values_obj) {
    PyObject* fast = PySequence_Fast(values_obj, "betas must be a sequence");
    if (!fast) throw std::runtime_error("betas must be a sequence");
    std::vector<double> values;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(fast);
    values.reserve(count);
    for (Py_ssize_t i = 0; i < count; ++i) {
        const double value = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, i));
        if (PyErr_Occurred()) {
            Py_DECREF(fast);
            throw std::runtime_error("betas must contain numbers");
        }
        values.push_back(value);
    }
    Py_DECREF(fast);
    return values;
}

PyObject* order_to_python(const std::vector<std::pair<int, int>>& order) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(order.size()));
    if (!list) return nullptr;
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(order.size()); ++i) {
        PyObject* pair = Py_BuildValue("(ii)", order[i].first, order[i].second);
        if (!pair) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, i, pair);
    }
    return list;
}

PyObject* greedy_result_to_python(const GreedyRun& result) {
    PyObject* tuple = PyTuple_New(3);
    PyObject* order = order_to_python(result.order);
    if (!tuple || !order) {
        Py_XDECREF(tuple);
        Py_XDECREF(order);
        return nullptr;
    }
    PyTuple_SET_ITEM(tuple, 0, order);
    PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(result.tc));
    PyTuple_SET_ITEM(tuple, 2, PyFloat_FromDouble(result.sc));
    return tuple;
}

PyObject* py_greedy_order(PyObject*, PyObject* args) {
    PyObject* tn;
    const char* strategy;
    int seed;
    if (!PyArg_ParseTuple(args, "Osi", &tn, &strategy, &seed)) return nullptr;
    try {
        Network net = parse_network(tn);
        const std::string name(strategy);
        const int strategy_id = name == "min_dim" ? 0 : name == "max_reduce" ? 1 : -1;
        if (strategy_id < 0) throw std::runtime_error("unknown greedy strategy");
        GreedySearch search(net, static_cast<uint64_t>(seed), strategy_id);
        return greedy_result_to_python(search.run());
    } catch (const std::exception& error) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_multicost_greedy_order(PyObject*, PyObject* args) {
    PyObject* tn;
    int seed;
    const char* minimize;
    int max_repeats;
    double max_time;
    int requested_cost_id;
    if (!PyArg_ParseTuple(
            args, "Oisidi", &tn, &seed, &minimize, &max_repeats,
            &max_time, &requested_cost_id
        )) return nullptr;
    try {
        Network net = parse_network(tn);
        const bool minimize_flops = std::string(minimize) == "flops";
        if (!minimize_flops && std::string(minimize) != "size") {
            throw std::runtime_error("minimize must be 'size' or 'flops'");
        }
        const int portfolio_size = requested_cost_id >= 0 ? 1 : 8;
        const int calls_per_cost = std::max(
            8, static_cast<int>(0.25 * max_repeats / portfolio_size)
        );
        std::vector<int> calls(8, 0);
        int best_cost = requested_cost_id >= 0 ? requested_cost_id : 0;
        int completed = 0;
        GreedyRun best;
        const auto started = std::chrono::steady_clock::now();
        RNG seeds(static_cast<uint64_t>(seed));
        for (int repeat = 0; repeat < max_repeats; ++repeat) {
            if (max_time > 0.0 && completed > 0) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started
                ).count();
                if (elapsed >= max_time) break;
            }
            int cost = requested_cost_id;
            if (cost < 0) {
                cost = repeat % 8;
                if (calls[cost] >= calls_per_cost) cost = best_cost;
            }
            const bool use_thermal =
                (cost == 1 || cost == 3 || cost == 4 || cost == 6) &&
                calls[cost] >= 4 * calls_per_cost;
            GreedySearch search(net, seeds.next_u64(), 0, cost, use_thermal);
            GreedyRun current = search.run();
            ++calls[cost];
            ++completed;
            if (better_run(current, best, minimize_flops)) {
                best = std::move(current);
                best_cost = cost;
            }
        }

        PyObject* tuple = PyTuple_New(5);
        PyObject* order = order_to_python(best.order);
        if (!tuple || !order) {
            Py_XDECREF(tuple);
            Py_XDECREF(order);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, 0, order);
        PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(best.tc));
        PyTuple_SET_ITEM(tuple, 2, PyFloat_FromDouble(best.sc));
        PyTuple_SET_ITEM(tuple, 3, PyLong_FromLong(best_cost));
        PyTuple_SET_ITEM(tuple, 4, PyLong_FromLong(completed));
        return tuple;
    } catch (const std::exception& error) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_anneal_order(PyObject*, PyObject* args) {
    PyObject* tn;
    PyObject* order_obj;
    PyObject* betas_obj;
    int niters;
    int seed;
    double sc_target;
    double alpha;
    double sc_weight;
    if (!PyArg_ParseTuple(
            args, "OOOiiddd", &tn, &order_obj, &betas_obj, &niters, &seed,
            &sc_target, &alpha, &sc_weight
        )) return nullptr;
    try {
        Network net = parse_network(tn);
        std::vector<std::pair<int, int>> initial = parse_order(order_obj, net.n);
        std::vector<double> betas = parse_doubles(betas_obj);
        AnnealTree tree(
            net, initial, static_cast<uint64_t>(seed), sc_target, alpha, sc_weight
        );
        TreeMetrics metrics = tree.optimize(betas, niters);
        PyObject* tuple = PyTuple_New(4);
        PyObject* order = order_to_python(tree.order());
        if (!tuple || !order) {
            Py_XDECREF(tuple);
            Py_XDECREF(order);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, 0, order);
        PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(metrics.tc));
        PyTuple_SET_ITEM(tuple, 2, PyFloat_FromDouble(metrics.sc));
        PyTuple_SET_ITEM(tuple, 3, PyFloat_FromDouble(metrics.mc));
        return tuple;
    } catch (const std::exception& error) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"greedy_order", py_greedy_order, METH_VARARGS, "Find a native greedy order."},
    {"multicost_greedy_order", py_multicost_greedy_order, METH_VARARGS,
     "Find a multi-cost portfolio greedy order."},
    {"anneal_order", py_anneal_order, METH_VARARGS,
     "Optimize a pairwise order with native tree simulated annealing."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_order_core",
    "Native greedy and TreeSA kernels for Artensor.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__order_core(void) {
    return PyModule_Create(&module);
}
