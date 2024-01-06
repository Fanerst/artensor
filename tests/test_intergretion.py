# %%
import sys
import numpy as np
sys.path.append('../')
from artensor import (
    tensor_network_contraction,
    quantum_circuit_simulation,
    TensorNetworkSimulation
)
if __name__ == '__main__':
    # correctness_table = {
    #     '100010001000000011011000101000':   4.13647322e-05 + 1j*(-3.39767357e-05),
    #     '110011110101010100110101010111':  -1.18974149e-05 + 1j*( 3.91406047e-06),
    #     '001001110001000100010101100101':  -1.06187972e-05 + 1j*( 9.29561065e-06),
    #     '110010111010111001110011010111':  -1.27609019e-05 + 1j*( 1.50818751e-05),
    #     '000100100110011100111011001101':   1.48501595e-05 + 1j*(-3.10959294e-05),
    # }
    correctness_table = {
        '100001000001' :     0.0198028199 + 1j * (  0.0106442748),
        '000101111011' :    0.00497586094 + 1j * ( -0.0245072283),
        '011000101100' :   -0.00853562169 + 1j * (-0.00701293815),
        '111001100001' :    -0.0100137182 + 1j * (  0.0147468708),
        '001110110000' :    0.00681955926 + 1j * (  0.0106616206),
    }

    circuit_filename = 'circuit_n12_m14_s0_e0_pEFGH.qsim'
    bitstrings = list(correctness_table.keys())
    sim = TensorNetworkSimulation.from_circuit_file(circuit_filename, bitstrings)
    sim.prepare_contraction()
    result = sim.contraction()
    # result, bitstrings = quantum_circuit_simulation(circuit_filename, bitstrings, 30, device='cuda:4')
    for bitstring, amp in zip(sim.bitstrings_sorted, result):
        assert np.allclose(amp.item(), correctness_table[bitstring])

# bitstrings_new = [
#     '111111111111',
#     '000000000000',
#     '000101111011',
#     '111001100001',
#     '001110110000',
#     '011000101100',
# ]
# sim.update_scheme(30, bitstrings_new)
# result = sim.contraction()

# sim = TensorNetworkSimulation.from_circuit_file(circuit_filename)
# sim.prepare_contraction()
# result = sim.contraction()
# result, bitstrings = quantum_circuit_simulation(circuit_filename, [], 30, device='cuda:4')
# amps = result.reshape(-1).cpu().numpy()
# for bitstring, amp in correctness_table.items():
#     assert np.allclose(amps[int(bitstring, 2)], amp)

# circuit_filename = 'circuit_n12_m14_s0_e0_pEFGH.qsim'
# circ = TensorNetworkCircuit(circuit_filename, device='cpu')
# tensors, tensor_bonds, bond_dims, final_qubits = circ.to_numerical_tn()
# atn = NumericalTensorNetwork(tensors, tensor_bonds, bond_dims, final_qubits)

# ## sprase case
# tensor_bonds_reorder, final_qubit_inds = atn._simplify('sparse')
# # reorder_keys = {i:j for i, j in enumerate(atn.tensor_bonds.keys())}
# # final_qubit_inds = [0] * n
# # tensor_bonds_reordered = {}
# # for i, j in reorder_keys.items():
# #     if j in final_qubits:
# #         assert len(atn.tensor_bonds[j]) == 2
# #         bond1, bond2 = atn.tensor_bonds[j]
# #         assert bond1.split('-')[1] == bond2.split('-')[1]
# #         final_qubit_inds[int(bond1.split('-')[1])] = i
# #         if int(bond1.split('-')[0]) > int(bond2.split('-')[0]):
# #             new_bonds = [bond2]
# #         else:
# #             new_bonds = [bond1]
# #             atn.tensors[j] = atn.tensors[j].T
# #     else:
# #         new_bonds = atn.tensor_bonds[j]
# #     tensor_bonds_reordered[i] = new_bonds

# bitstrings = list(correctness_table.keys())
# order_slicing, slicing_bonds, ctree_new = find_order(
#     tensor_bonds_reorder, atn.bond_dims, final_qubit_inds, 0, len(bitstrings),
#     sc_target=30, trials=5, 
#     iters=10, slicing_repeat=1, betas=np.linspace(3.0, 21.0, 61)
# )
# scheme, _, bitstrings_sorted = contraction_scheme_sparse(ctree_new, bitstrings, sc_target=30)
# tensors_reordered = {i: atn.tensors[j] for i, j in enumerate(atn.tensors.keys())}
# result = tensor_contraction_sparse(tensors_reordered, scheme)
# for bitstring, amp in zip(bitstrings_sorted, result):
#     assert np.allclose(amp.item(), correctness_table[bitstring])

# ## open case
# circuit_filename = 'circuit_n12_m14_s0_e0_pEFGH.qsim'
# circ = TensorNetworkCircuit(circuit_filename, device='cpu')
# tensors, tensor_bonds, bond_dims, final_qubits = circ.to_numerical_tn()
# atn = NumericalTensorNetwork(tensors, tensor_bonds, bond_dims, final_qubits)
# tensor_bonds_reorder, _ = atn._simplify('normal')

# order_slicing, slicing_bonds, ctree_new = find_order(
#     tensor_bonds_reorder, atn.bond_dims, 0, 
#     sc_target=30, trials=5, 
#     iters=10, slicing_repeat=1, betas=np.linspace(3.0, 21.0, 61)
# )
# assert len(slicing_bonds) == 0
# scheme, output_bonds = contraction_scheme(ctree_new)
# tensors_reordered = {i: atn.tensors[j] for i, j in enumerate(atn.tensors.keys())}
# result = tensor_contraction(tensors_reordered, scheme)
# qubit_order_final = [int(bond.split('-')[1]) for bond in output_bonds]
# perm_inds = tuple(np.argsort(qubit_order_final))
# amplitudes = result.cpu().numpy().transpose(perm_inds).reshape(-1)
# # for x, y in order_slicing:
# #     atn.contract(reorder_keys[x], reorder_keys[y])
# # left_ind = list(atn.tensors.keys())[0]
# # qubit_order_final = [int(bond.split('-')[1]) for bond in atn.tensor_bonds[left_ind]]
# # perm_inds = tuple(np.argsort(qubit_order_final))
# # amplitudes = atn.tensors[left_ind].permute(perm_inds).reshape(-1)
# for bitstring, amp in correctness_table.items():
#     assert np.allclose(amplitudes[int(bitstring, 2)], amp)
# %%
