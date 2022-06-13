from math import log2, ceil
import numpy as np
from .utils import(
    log2_accum_dims,
    final_qubits_num,
    log10sumexp2
)

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
        pass

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
        contract_bonds = set([bond for bond in common_bonds if self.tn.bond_tensors[bond].issubset(contracted_tensors)])
        result_bonds = all_bonds - contract_bonds
        factor = min(self.tn.log2_max_bitstring, final_qubits_num(self.tn.num_fq, contracted_tensors))
        sc = log2_accum_dims(self.tn.bond_dims, result_bonds)
        sc += factor
        if 'min_dim' in self.strategy:
            value = sc
        elif 'max_reduce' in self.strategy:
            value = sc - (log2_accum_dims(self.tn.bond_dims, self.contain_bonds[i]) + log2_accum_dims(self.tn.bond_dims, self.contain_bonds[j]))
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
        contract_bonds = set([bond for bond in common_bonds if self.tn.bond_tensors[bond].issubset(contracted_tensors)])
        result_bonds = all_bonds - contract_bonds
    
        l_num_fq = final_qubits_num(self.tn.num_fq, self.contain_tensors[i])
        r_num_fq = final_qubits_num(self.tn.num_fq, self.contain_tensors[j])
        num_fq = l_num_fq + r_num_fq
        factor = min(self.tn.log2_max_bitstring, num_fq)
        if l_num_fq < self.tn.log2_max_bitstring and r_num_fq < self.tn.log2_max_bitstring and num_fq > ceil(self.tn.log2_max_bitstring):
            factor += num_fq - ceil(self.tn.log2_max_bitstring)
        sc = log2_accum_dims(self.tn.bond_dims, result_bonds)
        tc = log2_accum_dims(self.tn.bond_dims, all_bonds) if contract_bonds else log2_accum_dims(self.tn.bond_dims, all_bonds) - 1
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
        tcs, scs, order = [], [np.log2(np.prod([self.tn.bond_dims[bond] for bond in self.tn.tensor_bonds[i]])) for i in range(len(self.tn.tensor_bonds))], []
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

    def __call__(self, strategy='min_dim', seed=0):
        """
        Call the class
        """
        self.strategy = strategy
        self.contain_tensors = [set([i]) for i in range(len(self.tn.tensor_bonds))] 
        self.contain_bonds = [set(self.tn.tensor_bonds[i]) for i in range(len(self.tn.tensor_bonds))] 
        self.tensor_neighbors = []
        for i in range(len(self.contain_tensors)):
            self.tensor_neighbors.append(set())
            for bond in self.contain_bonds[i]:
                self.tensor_neighbors[i] = self.tensor_neighbors[i] | self.tn.bond_tensors[bond]
            self.tensor_neighbors[i].discard(i)
        self.potential_contraction_pair = [(min(i, j), max(i, j)) for i in range(len(self.contain_tensors)) for j in self.tensor_neighbors[i]]
        self._construct_pair_info()
        order, tc, sc = self.greedy_order(seed)
        return order, tc, sc
