import sys
import time
from copy import deepcopy
import numpy as np
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)+'/../'))
from artensor import (
    find_order,
    NumericalTensorNetwork, 
    TensorNetworkCircuit,
)
import time

"""commandline examples:

    For n53m20 single amplitude without memory constraint:
    python circuit_complexity.py -n 53 -m 20 -seq ABCDCDAB -max_bitstrings 1 -sc_target 1000 -trial_num 36 -start_num 0 -repeat_num 1 -alpha 0.0
    For n53m20 1 million amplitudes with maximum size of intermediate tensor to be 2^36:
    python circuit_complexity.py -n 53 -m 20 -seq ABCDCDAB -max_bitstrings 1000000 -sc_target 36 -trial_num 36 -start_num 13 -repeat_num 1 -alpha 0.0
"""


def circuit_complexity(
        n=53, m=12, seq='ABCDCDAB', sc_target=31, 
        max_bitstrings=2**20, trial_num=36, repeat_num=10, start_num=0, 
        alpha=32.0
    ):
    circuit_filename = abspath(dirname(__file__)) + '/circuits/' + \
        f'circuit_n{n}_m{m}_s0_e0_p{seq}.qsim'
        
    circ = TensorNetworkCircuit(circuit_filename, device='cpu')
    tensors, tensor_bonds, bond_dims, final_qubits = circ.to_numerical_tn()
    ntn = NumericalTensorNetwork(tensors, tensor_bonds, bond_dims, final_qubits)

    ## sprase case
    tensor_bonds_reorder, final_qubit_inds = ntn._simplify('sparse')
    bonds_reorder = list(set(sum(tensor_bonds_reorder.values(), start=[])))
    bond_tensors_reorder = {bond: set() for bond in bonds_reorder}
    for i in tensor_bonds_reorder.keys():
        for j in tensor_bonds_reorder[i]:
            bond_tensors_reorder[j].add(i)

    for j in range(start_num, start_num + repeat_num):
        betas = np.linspace(3.0, 21.0, 61)
        t0 = time.time()
        order_slicing, slicing_bonds, ctree_new = find_order(
            deepcopy(tensor_bonds_reorder), 
            deepcopy(ntn.bond_dims), 
            final_qubit_inds, 
            0,
            max_bitstrings,
            sc_target=sc_target, 
            trials=trial_num, 
            iters=50, 
            betas=betas, 
            start_seed=trial_num*j,
            alpha=alpha
        )
        print(f'experiment {j}:')
        print(f'{max_bitstrings} bitstrings results:')
        print('order_slicing =', order_slicing)
        print('slicing_bonds =', slicing_bonds)
        print(f'time: {time.time() - t0}, {len(slicing_bonds)}')
        tc, sc, mc = ctree_new.tree_complexity()
        print('tc {:.5f} overall tc {:.5f} sc {:.5f} mc {:.5f} alpha {:.5f} arithematic intensity {:.2f}'.format(
            tc, np.log10(2**len(slicing_bonds)*10**tc), sc, mc, 10**(tc-mc), alpha)
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=53)
    parser.add_argument("-m", type=int, default=20)
    parser.add_argument("-seq", type=str, default='ABCDCDAB')
    parser.add_argument("-max_bitstrings", type=int, default=1)
    parser.add_argument("-sc_target", type=int, default=100)
    parser.add_argument("-trial_num", type=int, default=36)
    parser.add_argument("-start_num", type=int, default=0)
    parser.add_argument("-repeat_num", type=int, default=20)
    parser.add_argument("-alpha", type=float, default=0.0)
    args = parser.parse_args()

    circuit_complexity(**vars(args))