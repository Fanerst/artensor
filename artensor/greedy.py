from .utils import final_qubits_num, log2_accum_cached, log10sumexp2
from math import ceil
import heapq
import numpy as np


class GreedyOrderFinder:
    def __init__(self, tensor_network) -> None:
        """
        Class of greedy order finder
        Parameters:
        -----------
        tensor_network: AbstractTensorNetwork class
            the underlying tensor network
        -----------
        """
        self.tn = tensor_network

    def _construct_pair_info(self):
        """
        Construct the pair contraction info
        """
        self.pair_info = {}.fromkeys(self.potential_contraction_pair)
        for pair in self.pair_info.keys():
            self.pair_info[pair] = self._update_pair_info(pair)

    def _update_pair_info(self, pair):
        """
        Update the pair contraction info
        """
        i, j = pair
        contracted_tensors = self.contain_tensors[i] | self.contain_tensors[j]
        all_bonds = self.contain_bonds[i] | self.contain_bonds[j]
        common_bonds = self.contain_bonds[i] & self.contain_bonds[j]
        contract_bonds = set(
            bond for bond in common_bonds
            if bond not in self.tn.open_bonds and self.tn.bond_tensors[bond].issubset(contracted_tensors)
        )
        result_bonds = all_bonds - contract_bonds
        factor = min(self.tn.log2_max_bitstring, final_qubits_num(self.tn.num_fq, contracted_tensors))
        sc = log2_accum_cached(self.tn.log2_bond_dims, result_bonds)
        sc += factor
        if 'min_dim' in self.strategy:
            value = sc
        elif 'max_reduce' in self.strategy:
            value = sc - (
                log2_accum_cached(self.tn.log2_bond_dims, self.contain_bonds[i]) +
                log2_accum_cached(self.tn.log2_bond_dims, self.contain_bonds[j])
            )
        else:
            value = 1.0
        return value

    def contract(self, pair):
        """
        Contract a pair and calculate the complexity
        """
        i, j = pair
        pairs_add = []
        for neigh in self.tensor_neighbors[j]:
            pair_eliminate = (min(j, neigh), max(j, neigh))
            self.pair_info.pop(pair_eliminate)
            if neigh != i and neigh not in self.tensor_neighbors[i]:
                pairs_add.append((min(i, neigh), max(i, neigh)))
        pairs_add += [(min(i, m), max(i, m)) for m in self.tensor_neighbors[i] if m != j]
        pairs_add = set(pairs_add)

        contracted_tensors = self.contain_tensors[i] | self.contain_tensors[j]
        all_bonds = self.contain_bonds[i] | self.contain_bonds[j]
        common_bonds = self.contain_bonds[i] & self.contain_bonds[j]
        contract_bonds = set(
            bond for bond in common_bonds
            if bond not in self.tn.open_bonds and self.tn.bond_tensors[bond].issubset(contracted_tensors)
        )
        result_bonds = all_bonds - contract_bonds

        l_num_fq = final_qubits_num(self.tn.num_fq, self.contain_tensors[i])
        r_num_fq = final_qubits_num(self.tn.num_fq, self.contain_tensors[j])
        num_fq = l_num_fq + r_num_fq
        factor = min(self.tn.log2_max_bitstring, num_fq)
        if (
            l_num_fq < self.tn.log2_max_bitstring and
            r_num_fq < self.tn.log2_max_bitstring and
            num_fq > ceil(self.tn.log2_max_bitstring)
        ):
            factor += num_fq - ceil(self.tn.log2_max_bitstring)
        sc = log2_accum_cached(self.tn.log2_bond_dims, result_bonds)
        all_bonds_cost = log2_accum_cached(self.tn.log2_bond_dims, all_bonds)
        tc = all_bonds_cost if contract_bonds else all_bonds_cost - 1
        sc += factor
        tc += factor
        self.contain_tensors[i] = contracted_tensors
        self.contain_bonds[i] = result_bonds
        self.tensor_neighbors[i] = self.tensor_neighbors[i] | self.tensor_neighbors[j]
        self.tensor_neighbors[i].discard(i)
        self.tensor_neighbors[i].discard(j)

        for tensor_id in self.tensor_neighbors[j]:
            if tensor_id != i:
                self.tensor_neighbors[tensor_id].discard(j)
                self.tensor_neighbors[tensor_id].add(i)

        for pair_update in pairs_add:
            self.pair_info[pair_update] = self._update_pair_info(pair_update)

        return tc, sc

    def _pair_select(self, rng):
        """
        select a pair for contraction
        """
        min_value = min(self.pair_info.values())
        min_pairs = [pair for pair in self.pair_info.keys() if self.pair_info[pair] == min_value]
        pair = min_pairs[rng.choice(range(len(min_pairs)))]
        return pair

    def greedy_order(self, seed):
        """
        Return a greedy order according to specific greedy strategy
        """
        tcs, scs, order = [], [
            np.log2(np.prod([self.tn.bond_dims[bond] for bond in self.tn.tensor_bonds[i]]))
            for i in range(len(self.tn.tensor_bonds))
        ], []
        rng = np.random.RandomState(seed)
        uncontract = True
        while uncontract:
            if len(self.pair_info) > 0:
                pair = self._pair_select(rng)
                tc_step, sc_step = self.contract(pair)
                order.append(pair)
                tcs.append(tc_step)
                scs.append(sc_step)
            else:
                involved_nodes = set()
                for pair in order:
                    involved_nodes.add(pair[1])
                uninvolved_nodes = set(list(range(len(self.tn.tensor_bonds)))) - involved_nodes
                source_node = order[-1][0]
                for node in uninvolved_nodes:
                    if node == source_node:
                        continue
                    pair = (source_node, node)
                    tc_step, sc_step = self.contract(pair)
                    order.append(pair)
                    tcs.append(tc_step)
                    scs.append(sc_step)
                uncontract = False

        tc = log10sumexp2(tcs)
        sc = max(scs)

        return order, tc, sc

    def _pair_score_fast(self, i, j):
        contracted_tensor_mask = self.component_tensor_masks[i] | self.component_tensor_masks[j]
        all_bonds_mask = self.component_bond_masks[i] | self.component_bond_masks[j]
        common_bonds_mask = self.component_bond_masks[i] & self.component_bond_masks[j]
        contract_bonds_mask = 0
        remaining_common = common_bonds_mask
        while remaining_common:
            lowest_bit = remaining_common & -remaining_common
            bond = self.tn._bond_ids[lowest_bit.bit_length() - 1]
            if bond not in self.tn.open_bonds and self.tn.bond_tensor_masks[bond] & contracted_tensor_mask == self.tn.bond_tensor_masks[bond]:
                contract_bonds_mask |= lowest_bit
            remaining_common ^= lowest_bit
        result_bonds_mask = all_bonds_mask & ~contract_bonds_mask

        left_fq = self.component_num_fq[i]
        right_fq = self.component_num_fq[j]
        num_fq = left_fq + right_fq
        factor = min(self.tn.log2_max_bitstring, num_fq)
        if (
            left_fq < self.tn.log2_max_bitstring and
            right_fq < self.tn.log2_max_bitstring and
            num_fq > self.log2_max_bitstring_ceil
        ):
            factor += num_fq - self.log2_max_bitstring_ceil

        sc = self.tn.sum_log2_dims_mask(result_bonds_mask) + factor
        all_bonds_cost = self.tn.sum_log2_dims_mask(all_bonds_mask)
        tc = (all_bonds_cost if contract_bonds_mask else all_bonds_cost - 1) + factor
        if self.strategy == 'min_dim':
            value = sc
        elif self.strategy == 'max_reduce':
            value = sc - (
                self.tn.sum_log2_dims_mask(self.component_bond_masks[i]) +
                self.tn.sum_log2_dims_mask(self.component_bond_masks[j])
            )
        else:
            value = 1.0
        return value, tc, sc, contracted_tensor_mask, result_bonds_mask, num_fq

    def _push_pair(self, i, j):
        if i == j or not self.active[i] or not self.active[j]:
            return
        a, b = (i, j) if i < j else (j, i)
        value, tc, _, _, _, _ = self._pair_score_fast(a, b)
        version = self.pair_versions.get((a, b), 0) + 1
        self.pair_versions[(a, b)] = version
        heapq.heappush(self.pair_heap, (value, tc, a, b, version))

    def _pop_best_pair(self):
        while self.pair_heap:
            _, _, i, j, version = heapq.heappop(self.pair_heap)
            if not (self.active[i] and self.active[j]):
                continue
            if self.pair_versions.get((i, j)) != version:
                continue
            return i, j
        return None

    def _contract_fast(self, i, j):
        _, tc, sc, contracted_tensor_mask, result_bonds_mask, num_fq = self._pair_score_fast(i, j)
        old_i_neighbors = self.tensor_neighbors[i].copy()
        old_j_neighbors = self.tensor_neighbors[j].copy()
        new_neighbors = (old_i_neighbors | old_j_neighbors) - {i, j}

        self.component_tensor_masks[i] = contracted_tensor_mask
        self.component_bond_masks[i] = result_bonds_mask
        self.component_num_fq[i] = num_fq
        self.tensor_neighbors[i] = new_neighbors
        self.active[j] = False
        self.tensor_neighbors[j].clear()
        self.component_tensor_masks[j] = 0
        self.component_bond_masks[j] = 0
        self.component_num_fq[j] = 0

        self.pair_versions.pop((i, j), None)
        for neigh in old_j_neighbors:
            if neigh == i:
                continue
            self.tensor_neighbors[neigh].discard(j)
            self.pair_versions.pop((min(j, neigh), max(j, neigh)), None)
        for neigh in old_i_neighbors:
            if neigh != j:
                self.pair_versions.pop((min(i, neigh), max(i, neigh)), None)
        for neigh in new_neighbors:
            self.tensor_neighbors[neigh].discard(j)
            self.tensor_neighbors[neigh].add(i)
            self._push_pair(i, neigh)
        return tc, sc

    def greedy_order_fast(self, seed):
        n = len(self.tn.tensor_bonds)
        self.log2_max_bitstring_ceil = ceil(self.tn.log2_max_bitstring)
        self.active = [True] * n
        self.component_tensor_masks = [self.tn.tensor_bitmasks[i] for i in range(n)]
        self.component_bond_masks = [self.tn.tensor_bond_masks[i] for i in range(n)]
        self.component_num_fq = list(self.tn.num_fq)
        self.tensor_neighbors = []
        for i in range(n):
            neigh = set()
            for bond in self.tn.tensor_bonds[i]:
                neigh |= self.tn.bond_tensors[bond]
            neigh.discard(i)
            self.tensor_neighbors.append(neigh)

        self.pair_heap = []
        self.pair_versions = {}
        for i in range(n):
            for j in self.tensor_neighbors[i]:
                if j > i:
                    self._push_pair(i, j)

        order = []
        tcs = []
        scs = [self.tn.sum_log2_dims_mask(self.tn.tensor_bond_masks[i]) for i in range(n)]
        active_count = n
        while active_count > 1:
            pair = self._pop_best_pair()
            if pair is None:
                active_nodes = [idx for idx, alive in enumerate(self.active) if alive]
                i, j = active_nodes[0], active_nodes[1]
            else:
                i, j = pair
            tc_step, sc_step = self._contract_fast(i, j)
            order.append((i, j))
            tcs.append(tc_step)
            scs.append(sc_step)
            active_count -= 1

        tc = log10sumexp2(tcs)
        sc = max(scs)
        return order, tc, sc

    def __call__(self, strategy='min_dim', seed=0):
        """
        Call the class
        """
        self.strategy = strategy
        if strategy == 'min_dim':
            return self.greedy_order_fast(seed)

        self.contain_tensors = [set([i]) for i in range(len(self.tn.tensor_bonds))]
        self.contain_bonds = [set(self.tn.tensor_bonds[i]) for i in range(len(self.tn.tensor_bonds))]
        self.tensor_neighbors = []
        for i in range(len(self.contain_tensors)):
            self.tensor_neighbors.append(set())
            for bond in self.contain_bonds[i]:
                self.tensor_neighbors[i] = self.tensor_neighbors[i] | self.tn.bond_tensors[bond]
            self.tensor_neighbors[i].discard(i)
        self.potential_contraction_pair = [
            (min(i, j), max(i, j))
            for i in range(len(self.contain_tensors))
            for j in self.tensor_neighbors[i]
        ]
        self._construct_pair_info()
        return self.greedy_order(seed)
