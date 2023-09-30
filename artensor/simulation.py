from .order_finder import find_order
from .tensor_network import NumericalTensorNetwork
from .contraction import (
    contraction_scheme,
    contraction_scheme_sparse,
    tensor_contraction,
    tensor_contraction_sparse
)
from .circuit import TensorNetworkCircuit
import numpy as np
from copy import deepcopy
import torch

def check_bitstrings(bitstrings):
    if len(bitstrings):
        max_bitstrings = len(np.unique(bitstrings))
        # if max_bitstrings != len(bitstrings):
        #     print('repeated bitstrings detected.')
        pattern = 'sparse'
    else:
        max_bitstrings = 1
        pattern = 'normal'
    return pattern, max_bitstrings

def get_bond_tensors(tensor_bonds):
    bonds = list(set(sum(tensor_bonds.values(), start=[])))
    bond_tensors = {bond: set() for bond in bonds}
    for i in tensor_bonds.keys():
        for j in tensor_bonds[i]:
            bond_tensors[j].add(i)
    return bond_tensors

class TensorNetworkSimulation:
    def __init__(
            self, tensors, tensor_bonds, bond_dims, final_qubits, 
            bitstrings, pattern, max_bitstrings
        ) -> None:
        self.tensors = tensors
        self.tensor_bonds = tensor_bonds
        self.bond_dims = bond_dims
        self.final_qubits = final_qubits
        self.bitstrings = bitstrings
        self.pattern = pattern
        self.max_bitstrings =  max_bitstrings
        pass

    def prepare_contraction(
            self, sc_target=30, trials=6, iters=20, 
            betas=np.linspace(0.1, 10, 100), slicing_repeat=4, start_seed=0, 
            alpha=32.0):
        bond_tensors = get_bond_tensors(self.tensor_bonds)
        betas = np.linspace(3.0, 21.0, 61)
        order_slicing, slicing_bonds, ctree = find_order(
            self.tensor_bonds, self.bond_dims, self.final_qubits, 0, 
            self.max_bitstrings, sc_target=sc_target, trials=trials, iters=iters, 
            betas=betas, start_seed=start_seed, slicing_repeat=slicing_repeat, 
            alpha=alpha
        )

        self.slicing_indices = {}.fromkeys(slicing_bonds)

        for bond in slicing_bonds:
            tensor_ids = bond_tensors[bond]
            inds = [self.tensor_bonds[tid].index(bond) for tid in tensor_ids]
            self.slicing_indices[bond] = [(tid, ind) for tid, ind in zip(tensor_ids, inds)]

        if self.pattern == 'normal':
            self.scheme, self.output_bonds = contraction_scheme(ctree)
            self.tensor_contraction_func = tensor_contraction
        else:
            self.scheme, self.output_bonds, self.bitstrings_sorted = contraction_scheme_sparse(
                ctree, self.bitstrings, sc_target=sc_target
            )
            self.tensor_contraction_func = tensor_contraction_sparse
            assert len(self.bitstrings_sorted) == self.max_bitstrings
        if len(self.output_bonds) > 0:
            bond_inds = []
            for x in range(len(self.output_bonds)):
                assert len(bond_tensors[self.output_bonds[x]]) == 1
                tensor_id = bond_tensors[self.output_bonds[x]].pop()
                assert tensor_id in self.final_qubits
                bond_inds.append(list(self.final_qubits).index(tensor_id))
            self.permute_dims = tuple(np.argsort(bond_inds))
            if self.pattern == 'sparse':
                self.permute_dims = [0] + [dim+1 for dim in self.permute_dims]
    
    def contraction(self, tensors=None, dtype=torch.complex64, device='cpu'):
        if tensors is None:
            tensors =  [
                self.tensors[i].to(dtype).to(device) 
                for i in range(len(self.tensors))
            ]
        else:
            tensors = [
                tensors[i].to(dtype).to(device) for i in range(len(tensors))
            ]

        collect_tensor_shape = [self.max_bitstrings] + [2] * len(self.output_bonds) \
        if self.pattern == 'sparse' else [2] * len(self.output_bonds)
        collect_tensor = torch.zeros(
            collect_tensor_shape, dtype=dtype, device=device
        )
        slicing_bonds = list(self.slicing_indices.keys())
        for s in range(2**len(slicing_bonds)):
            configs = list(map(int, np.binary_repr(s, len(slicing_bonds))))
            sliced_tensors = tensors.copy()
            for x in range(len(slicing_bonds)):
                bond = slicing_bonds[x]
                for tid, ind in self.slicing_indices[bond]:
                    sliced_tensors[tid] = sliced_tensors[tid].select(ind, configs[x]).clone()
            collect_tensor += self.tensor_contraction_func(sliced_tensors, self.scheme)
        if len(self.output_bonds) > 0:
            collect_tensor = collect_tensor.permute(self.permute_dims)
        return collect_tensor

    @classmethod
    def from_circuit_file(cls, circuit_filename, bitstrings=[]):
        pattern, max_bitstrings = check_bitstrings(bitstrings)
        circ = TensorNetworkCircuit(circuit_filename)
        tensors, tensor_bonds, bond_dims, final_qubits = circ.to_numerical_tn()
        numerical_tn = NumericalTensorNetwork(tensors, tensor_bonds, bond_dims, final_qubits)
        tensor_bonds_reorder, final_qubit_inds = numerical_tn._simplify(pattern)
        tensors = {
            i: numerical_tn.tensors[j] 
            for i, j in enumerate(numerical_tn.tensors.keys())
        }
        sim = cls(
            tensors, tensor_bonds_reorder, bond_dims, final_qubit_inds, 
            bitstrings, pattern, max_bitstrings)
        return sim
    
    @classmethod
    def from_tn_circuit(cls, circ:TensorNetworkCircuit, bitstrings=[]):
        pattern, max_bitstrings = check_bitstrings(bitstrings)
        tensors, tensor_bonds, bond_dims, final_qubits = circ.to_numerical_tn()
        numerical_tn = NumericalTensorNetwork(tensors, tensor_bonds, bond_dims, final_qubits)
        tensor_bonds_reorder, final_qubit_inds = numerical_tn._simplify(pattern)
        tensors = {
            i: numerical_tn.tensors[j] 
            for i, j in enumerate(numerical_tn.tensors.keys())
        }
        sim = cls(
            tensors, tensor_bonds_reorder, bond_dims, final_qubit_inds, 
            bitstrings, pattern, max_bitstrings)
        return sim


def tensor_network_contraction(
        tensors, tensor_bonds, bond_dims, final_qubits, bitstrings=[], sc_target=31, 
        trial_num=8, alpha=0.0, dtype=torch.complex64, device='cpu'
    ):
    pattern, max_bitstrings = check_bitstrings(bitstrings)
    numerical_tn = NumericalTensorNetwork(tensors, tensor_bonds, bond_dims, final_qubits)
    tensor_bonds_reorder, final_qubit_inds = numerical_tn._simplify(pattern)
    bond_tensors_reorder = get_bond_tensors(tensor_bonds_reorder)

    betas = np.linspace(3.0, 21.0, 61)
    order_slicing, slicing_bonds, ctree = find_order(
        deepcopy(tensor_bonds_reorder), deepcopy(numerical_tn.bond_dims), 
        final_qubit_inds, 0, max_bitstrings, sc_target=sc_target, 
        trials=trial_num, iters=50, betas=betas, start_seed=0, alpha=alpha
    )

    slicing_indices = {}.fromkeys(slicing_bonds)
    tensors = {
        i: numerical_tn.tensors[j].to(dtype).to(device) 
        for i, j in enumerate(numerical_tn.tensors.keys())
    }

    for bond in slicing_bonds:
        tensor_ids = bond_tensors_reorder[bond]
        inds = [tensor_bonds_reorder[tid].index(bond) for tid in tensor_ids]
        slicing_indices[bond] = [(tid, ind) for tid, ind in zip(tensor_ids, inds)]

    if pattern == 'normal':
        scheme, output_bonds = contraction_scheme(ctree)
        tensor_contraction_func = tensor_contraction
    else:
        scheme, output_bonds, bitstrings = contraction_scheme_sparse(
            ctree, bitstrings, sc_target=sc_target
        )
        tensor_contraction_func = tensor_contraction_sparse
        assert len(bitstrings) == max_bitstrings
    if len(output_bonds) > 0:
        bond_inds = []
        for x in range(len(output_bonds)):
            assert len(bond_tensors_reorder[output_bonds[x]]) == 1
            tensor_id = bond_tensors_reorder[output_bonds[x]].pop()
            assert tensor_id in final_qubit_inds
            bond_inds.append(list(final_qubit_inds).index(tensor_id))
        permute_dims = tuple(np.argsort(bond_inds))
        if pattern == 'sparse':
            permute_dims = [0] + [dim+1 for dim in permute_dims]

    collect_tensor_shape = [max_bitstrings] + [2] * len(output_bonds) \
    if pattern == 'sparse' else [2] * len(output_bonds)
    collect_tensor = torch.zeros(
        collect_tensor_shape, dtype=dtype, device=device
    )
    for s in range(2**len(slicing_bonds)):
        configs = list(map(int, np.binary_repr(s, len(slicing_bonds))))
        sliced_tensors = tensors.copy()
        for x in range(len(slicing_bonds)):
            bond = slicing_bonds[x]
            for tid, ind in slicing_indices[bond]:
                sliced_tensors[tid] = sliced_tensors[tid].select(ind, configs[x]).clone()
        collect_tensor += tensor_contraction_func(sliced_tensors, scheme)
    if len(output_bonds) > 0:
        collect_tensor = collect_tensor.permute(permute_dims)
    return collect_tensor, bitstrings


def quantum_circuit_simulation(
        circuit_filename, bitstrings=[], sc_target=31, trial_num=8, 
        alpha=0.0, dtype=torch.complex64, device='cpu',
    ):
    circ = TensorNetworkCircuit(circuit_filename)
    tensors, tensor_bonds, bond_dims, final_qubits = circ.to_numerical_tn()
    return tensor_network_contraction(
        tensors, tensor_bonds, bond_dims, final_qubits, bitstrings, sc_target,
        trial_num, alpha, dtype, device,
    )