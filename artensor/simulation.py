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


def tensor_network_contraction(
        tensors, tensor_bonds, bond_dims, final_qubits, bitstrings=[], sc_target=31, 
        trial_num=8, alpha=0.0, dtype=torch.complex64, device='cpu',
    ):
    if len(bitstrings):
        max_bitstrings = len(np.unique(bitstrings))
        if max_bitstrings != len(bitstrings):
            print('repeated bitstrings detected.')
        pattern = 'sparse'
    else:
        max_bitstrings = 1
        pattern = 'normal'
    numerical_tn = NumericalTensorNetwork(tensors, tensor_bonds, bond_dims, final_qubits)
    tensor_bonds_reorder, final_qubit_inds = numerical_tn._simplify(pattern)
    bonds_reorder = list(set(sum(tensor_bonds_reorder.values(), start=[])))
    bond_tensors_reorder = {bond: set() for bond in bonds_reorder}
    for i in tensor_bonds_reorder.keys():
        for j in tensor_bonds_reorder[i]:
            bond_tensors_reorder[j].add(i)

    betas = np.linspace(3.0, 21.0, 61)
    order_slicing, slicing_bonds, ctree_new = find_order(
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
        scheme, output_bonds = contraction_scheme(ctree_new)
        tensor_contraction_func = tensor_contraction
    else:
        scheme, output_bonds, bitstrings = contraction_scheme_sparse(
            ctree_new, bitstrings, sc_target=sc_target
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