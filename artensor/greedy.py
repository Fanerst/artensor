"""Greedy contraction-order optimizers.

The original implementation selected the next contraction by scanning every
candidate on every step.  The implementation below keeps a lazy priority heap
and only recomputes candidates adjacent to the tensor produced by a merge.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import inf
import time

import numpy as np

from .utils import log2_accum_dims, log10sumexp2


MULTI_COST_FUNCTIONS = (
    "balanced_boltzmann",
    "boltzmann",
    "max_skew",
    "anti_balanced",
    "skew_balanced",
    "log",
    "memory_removed_jitter",
    "batch_balanced",
)


def _compiled_core():
    try:
        from . import _order_core
    except ImportError:
        return None
    return _order_core


class GreedyOrderFinder:
    """Find a pairwise contraction order with an incremental greedy search.

    Parameters
    ----------
    tensor_network:
        The :class:`~artensor.AbstractTensorNetwork` to optimize.
    use_compiled:
        Use Artensor's optional native search core when it is available.
    """

    def __init__(self, tensor_network, use_compiled=True) -> None:
        self.tn = tensor_network
        self.use_compiled = use_compiled

    def _pair_value(self, i, j):
        contracted_tensors = self.contain_tensors[i] | self.contain_tensors[j]
        all_bonds = self.contain_bonds[i] | self.contain_bonds[j]
        common_bonds = self.contain_bonds[i] & self.contain_bonds[j]
        contract_bonds = {
            bond
            for bond in common_bonds
            if bond not in self.tn.output_bonds
            and self.tn.bond_tensors[bond].issubset(contracted_tensors)
        }
        result_bonds = all_bonds - contract_bonds
        factor = min(
            self.tn.log2_max_bitstring,
            self.final_counts[i] + self.final_counts[j],
        )
        output_size = log2_accum_dims(self.tn.bond_dims, result_bonds) + factor
        if self.strategy == "min_dim":
            return output_size
        if self.strategy == "max_reduce":
            return output_size - self.input_sizes[i] - self.input_sizes[j]
        raise ValueError(f"unknown greedy strategy {self.strategy!r}")

    def _invalidate(self, pair):
        self.versions[pair] = self.versions.get(pair, 0) + 1
        self.pair_info.pop(pair, None)

    def _push_pair(self, i, j):
        if i == j or not self.active[i] or not self.active[j]:
            return
        pair = (min(i, j), max(i, j))
        version = self.versions.get(pair, 0) + 1
        self.versions[pair] = version
        value = self._pair_value(*pair)
        self.pair_info[pair] = value
        self.serial += 1
        heapq.heappush(
            self.heap,
            (value, self.serial, pair[0], pair[1], version),
        )

    def _valid_entry(self, entry):
        _, _, i, j, version = entry
        pair = (i, j)
        return (
            self.active[i]
            and self.active[j]
            and self.versions.get(pair) == version
            and pair in self.pair_info
        )

    def _pop_valid(self):
        while self.heap:
            entry = heapq.heappop(self.heap)
            if self._valid_entry(entry):
                return entry
        return None

    def _pair_select(self, rng):
        first = self._pop_valid()
        if first is None:
            return None
        min_value = first[0]
        tied = [first]
        while self.heap and self.heap[0][0] == min_value:
            entry = heapq.heappop(self.heap)
            if self._valid_entry(entry):
                tied.append(entry)
        selected = int(rng.randint(len(tied)))
        for index, entry in enumerate(tied):
            if index != selected:
                heapq.heappush(self.heap, entry)
        _, _, i, j, _ = tied[selected]
        return i, j

    def _contract(self, pair):
        i, j = pair
        neighbors_i = self.tensor_neighbors[i]
        neighbors_j = self.tensor_neighbors[j]
        affected = (neighbors_i | neighbors_j) - {i, j}

        for neighbor in neighbors_i:
            self._invalidate((min(i, neighbor), max(i, neighbor)))
        for neighbor in neighbors_j:
            self._invalidate((min(j, neighbor), max(j, neighbor)))

        contracted_tensors = self.contain_tensors[i] | self.contain_tensors[j]
        all_bonds = self.contain_bonds[i] | self.contain_bonds[j]
        common_bonds = self.contain_bonds[i] & self.contain_bonds[j]
        contract_bonds = {
            bond
            for bond in common_bonds
            if bond not in self.tn.output_bonds
            and self.tn.bond_tensors[bond].issubset(contracted_tensors)
        }
        result_bonds = all_bonds - contract_bonds

        combined_final_count = self.final_counts[i] + self.final_counts[j]
        factor = min(self.tn.log2_max_bitstring, combined_final_count)
        sc = log2_accum_dims(self.tn.bond_dims, result_bonds) + factor
        tc = log2_accum_dims(self.tn.bond_dims, all_bonds)
        tc += factor

        self.contain_tensors[i] = contracted_tensors
        self.contain_bonds[i] = result_bonds
        self.final_counts[i] = combined_final_count
        self.input_sizes[i] = sc
        self.active[j] = False
        self.contain_tensors[j] = set()
        self.contain_bonds[j] = set()

        new_neighbors = {neighbor for neighbor in affected if self.active[neighbor]}
        self.tensor_neighbors[i] = new_neighbors
        self.tensor_neighbors[j] = set()
        for neighbor in new_neighbors:
            self.tensor_neighbors[neighbor].discard(j)
            self.tensor_neighbors[neighbor].add(i)
            self._push_pair(i, neighbor)
        return tc, sc

    def _initialize(self, strategy):
        self.strategy = strategy
        tensor_ids = list(self.tn.tensor_bonds)
        if tensor_ids != list(range(len(tensor_ids))):
            raise ValueError("Artensor tensor IDs must be consecutive integers starting at zero")
        n = len(tensor_ids)
        self.active = [True] * n
        self.contain_tensors = [{i} for i in range(n)]
        self.contain_bonds = [set(self.tn.tensor_bonds[i]) for i in range(n)]
        final_qubits = set(self.tn.final_qubits)
        self.final_counts = [int(i in final_qubits) for i in range(n)]
        self.input_sizes = [
            log2_accum_dims(self.tn.bond_dims, self.contain_bonds[i])
            + min(self.tn.log2_max_bitstring, self.final_counts[i])
            for i in range(n)
        ]
        self.tensor_neighbors = [set() for _ in range(n)]
        for i in range(n):
            for bond in self.contain_bonds[i]:
                self.tensor_neighbors[i].update(self.tn.bond_tensors[bond])
            self.tensor_neighbors[i].discard(i)

        self.heap = []
        self.versions = {}
        self.pair_info = {}
        self.serial = 0
        for i in range(n):
            for j in self.tensor_neighbors[i]:
                if i < j:
                    self._push_pair(i, j)

    def _python_greedy(self, strategy, seed):
        self._initialize(strategy)
        rng = np.random.RandomState(seed)
        order = []
        tcs = []
        scs = list(self.input_sizes)

        while True:
            pair = self._pair_select(rng)
            if pair is None:
                break
            tc, sc = self._contract(pair)
            order.append(pair)
            tcs.append(tc)
            scs.append(sc)

        active = [i for i, is_active in enumerate(self.active) if is_active]
        if active:
            source = active[0]
            for node in active[1:]:
                tc, sc = self._contract((source, node))
                order.append((source, node))
                tcs.append(tc)
                scs.append(sc)

        tc = log10sumexp2(tcs) if tcs else -inf
        sc = max(scs, default=0.0)
        return order, tc, sc

    def __call__(self, strategy="min_dim", seed=0):
        if strategy not in {"min_dim", "max_reduce"}:
            raise ValueError(f"unknown greedy strategy {strategy!r}")
        core = _compiled_core() if self.use_compiled else None
        if core is not None:
            return core.greedy_order(self.tn, strategy, int(seed))
        return self._python_greedy(strategy, seed)


@dataclass(frozen=True)
class MultiCostGreedyResult:
    order: list[tuple[int, int]]
    tc: float
    sc: float
    cost_function_id: int
    cost_function_name: str
    repeats: int
    elapsed: float


class MultiCostGreedyOrderFinder:
    """Portfolio greedy optimizer from Orgler and Blacher (2024).

    The native core implements the eight cost functions from the reference
    implementation.  Repeated paths initially explore the full portfolio and
    then favor the cost function that has produced the best objective.
    """

    def __init__(self, tensor_network, use_compiled=True):
        self.tn = tensor_network
        self.use_compiled = use_compiled

    def __call__(
        self,
        *,
        seed=0,
        minimize="size",
        max_repeats=128,
        max_time=0.0,
        cost_function_id=None,
    ):
        if minimize not in {"size", "flops"}:
            raise ValueError("minimize must be 'size' or 'flops'")
        if max_repeats < 1:
            raise ValueError("max_repeats must be positive")
        if cost_function_id is not None and not 0 <= cost_function_id < 8:
            raise ValueError("cost_function_id must be between 0 and 7")
        core = _compiled_core() if self.use_compiled else None
        if core is None:
            raise RuntimeError(
                "MultiCostGreedyOrderFinder requires Artensor's native extension; "
                "reinstall Artensor so its C++ extension can be built"
            )
        started = time.perf_counter()
        order, tc, sc, selected, completed = core.multicost_greedy_order(
            self.tn,
            int(seed),
            minimize,
            int(max_repeats),
            float(max_time),
            -1 if cost_function_id is None else int(cost_function_id),
        )
        return MultiCostGreedyResult(
            order=order,
            tc=tc,
            sc=sc,
            cost_function_id=selected,
            cost_function_name=MULTI_COST_FUNCTIONS[selected],
            repeats=completed,
            elapsed=time.perf_counter() - started,
        )


__all__ = [
    "GreedyOrderFinder",
    "MultiCostGreedyOrderFinder",
    "MultiCostGreedyResult",
    "MULTI_COST_FUNCTIONS",
]
