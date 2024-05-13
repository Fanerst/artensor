import torch
from math import pi, sqrt
from .gates import *

gate_dict = {
    'h': hadamard_gate,
    'cnot': cnot_gate,
    't': t_gate,
    'hz_1_2' : wsqrt_gate,
    'y_1_2' : ysqrt_gate,
    'x_1_2' : xsqrt_gate,
    'fsim' : fsim_gate,
    'fs' : fsim_gate,
    'rz' : rz_gate
}

single_qubit_gates = ['h', 't', 'hz_1_2', 'y_1_2', 'x_1_2', 'rz']
two_qubit_gates = ['cnot', 'fsim', 'fs']

class Tensor:
    def __init__(self, array, inds) -> None:
        self.array = array
        self.inds = inds
        pass

    def __repr__(self) -> str:
        return f"array shape: {tuple(self.array.shape)}, inds: {self.inds}"


class TensorNetworkCircuit:
    def __init__(
            self, circuit_filename, init_state=None, final_state=None, device='cpu', dtype=torch.complex64
        ) -> None:
        self.circuit_filename = circuit_filename
        self.dev = device
        self.dtype = dtype
        self.gate_seq = self._read_circuit()
        if init_state is None:
            self.init_state = '0' * self.n
        else:
            self.init_state = init_state
        assert len(self.init_state) == self.n
        self.final_state = final_state
        assert self.final_state is None or len(self.final_state) == self.n
        self._construct_circuit()
        pass

    def _read_circuit(self):
        with open(self.circuit_filename, 'r') as f:
            lines = f.readlines()
        self.n = int(lines[0].split()[0])
        gate_seq = []
        for line in lines[1:]:
            l = line.split()
            layer_num = int(l[0])
            assert layer_num == len(gate_seq) or layer_num == len(gate_seq) - 1
            if layer_num == len(gate_seq):
                gate_seq.append([])
            if l[1] in single_qubit_gates:
                gate_seq[layer_num].append((
                    gate_dict[l[1]], 
                    (int(l[2]),),
                    tuple((float(l[i]) for i in range(3, len(l))))
                ))
            elif l[1] in two_qubit_gates:
                gate_seq[layer_num].append((
                    gate_dict[l[1]], 
                    (int(l[2]), int(l[3])),
                    tuple((float(l[i]) for i in range(4, len(l))))
                ))
            else:
                raise ValueError(f'Unkown qubit gate {l[1]}')
        return gate_seq

    def _construct_circuit(self):
        # Initital states
        self.circuits_tn = [
            Tensor(
                torch.tensor([1, 0], dtype=self.dtype, device=self.dev)
                if config == '0' else 
                torch.tensor([0, 1], dtype=self.dtype, device=self.dev),
                [f'0-{i}']
            )
            for i, config in enumerate(self.init_state)
        ]
        wire_loc = [0 for i in range(self.n)]
        # Gates
        for layer in self.gate_seq:
            for gate in layer:
                gate_type, exec_qubits, params = gate
                gate_form = gate_type(*params, self.dtype, self.dev)
                inds = sum([
                    [f'{wire_loc[q]+1}-{q}', f'{wire_loc[q]}-{q}'] 
                    for q in exec_qubits
                ], start=[])
                inds = inds[0::2] + inds[1::2]
                self.circuits_tn.append(Tensor(gate_form, inds))
                for q in exec_qubits:
                    wire_loc[q] += 1
        if self.final_state is not None:
            # final_qubits
            self.circuits_tn += [
                Tensor(
                    torch.tensor([1, 0], dtype=self.dtype, device=self.dev) 
                    if self.final_state[i] == 0 else
                    torch.tensor([0, 1], dtype=self.dtype, device=self.dev),
                    [f'{wire_loc[i]}-{i}']
                )
                for i in range(self.n)
            ]

    def to_abstract_tn(self):
        assert 'circuits_tn' in self.__dict__.keys()
        tensor_bonds = {
            i: self.circuits_tn[i].inds for i in range(len(self.circuits_tn))
        }
        bond_dims = {bond: 2.0 for bond in set().union(*tensor_bonds.values())}
        final_qubits = set(
            range(len(tensor_bonds) - self.n, len(tensor_bonds)))
        return tensor_bonds, bond_dims, final_qubits
    
    def to_numerical_tn(self):
        assert 'circuits_tn' in self.__dict__.keys()
        tensors = {
            i: self.circuits_tn[i].array for i in range(len(self.circuits_tn))
        }
        tensor_bonds = {
            i: self.circuits_tn[i].inds for i in range(len(self.circuits_tn))
        }
        bond_dims = {bond: 2.0 for bond in set().union(*tensor_bonds.values())}
        final_qubits = set(
            range(len(tensor_bonds) - self.n, len(tensor_bonds)))
        return tensors, tensor_bonds, bond_dims, final_qubits
    
    def to_einsum(self):
        ALLOW_ACSII = list(range(65, 90)) + list(range(97, 122))
        LETTERS = [chr(ALLOW_ACSII[i]) for i in range(len(ALLOW_ACSII))]
        assert 'circuits_tn' in self.__dict__.keys()
        tensors = {
            i: self.circuits_tn[i].array for i in range(len(self.circuits_tn))
        }
        tensor_bonds = {
            i: self.circuits_tn[i].inds for i in range(len(self.circuits_tn))
        }
        bonds_all = list(set().union(*tensor_bonds.values()))
        einsum_eq = ','.join([
            ''.join([LETTERS[bonds_all.index(bond)] for bond in tensor_bonds[i]]) 
            for i in tensor_bonds.keys()
        ]) + '->' + ''.join([
            LETTERS[bonds_all.index(self.circuits_tn[i].inds[1])]
            for i in range(len(self.circuits_tn)-self.n, len(self.circuits_tn))
        ])
        return tensors, einsum_eq
    
    def state_vec(self,):
        state_vec = torch.zeros((2**self.n,), dtype=self.dtype, device=self.dev)
        state_vec[0] = 1
        state_vec = state_vec.reshape((2,)*self.n)
        state_vec_inds = [f'0-{i}' for i in range(self.n)]
        for tensor in self.circuits_tn[self.n:]:
            array, inds = tensor.array, tensor.inds
            contracted_inds = [ind for ind in state_vec_inds if ind in inds]
            state_vec_inds_new = [
                ind for ind in state_vec_inds + inds 
                if ind not in contracted_inds
            ]
            einsum_eq = einsum_eq_convert((state_vec_inds, inds), state_vec_inds_new)
            assert len(state_vec.shape) == len(state_vec_inds)
            assert len(array.shape) == len(inds)
            # print(state_vec_inds, inds, state_vec_inds_new, einsum_eq)
            state_vec = torch.einsum(einsum_eq, state_vec, array)
            state_vec_inds = state_vec_inds_new
        dims = torch.tensor([int(ind.split('-')[1]) for ind in state_vec_inds])
        perm_dims = torch.argsort(dims)
        return state_vec.permute(tuple(perm_dims))
    
    def to_mps(self,):
        mps = [t.array.reshape(1, 2, 1) for t in self.circuits_tn[:self.n]]
        for tensor in self.circuits_tn[self.n:]:
            array, inds = tensor.array, tensor.inds
            if len(inds) == 2:
                q = int(inds[0].split('-')[1])
                mps[q] = torch.einsum('abc,bd->adc', mps[q], array)
            elif len(inds) == 4:
                q1, q2 = int(inds[0].split('-')[1]), int(inds[1].split('-')[1])
                u, s, vh = torch.linalg.svd(array.permute(0, 2, 1, 3).reshape(4, 4))
                mq1 = (u @ torch.diag(torch.sqrt(s).to(self.dtype))).reshape(2, 2, -1)
                mq2 = (torch.diag(torch.sqrt(s).to(self.dtype)) @ vh).reshape(-1, 2, 2)
                mps[q1] = torch.einsum('abc, bde->adec', mps[q1], mq1).reshape(
                    mps[q1].shape[0], 2, -1
                )
                mps[q2] = torch.einsum('abc, ebd->eadc', mps[q2], mq2).reshape(
                    -1, 2, mps[q2].shape[2]
                )
            
        return mps


ALLOW_ACSII = list(range(65, 90)) + list(range(97, 122))
LETTES = [chr(ALLOW_ACSII[i]) for i in range(len(ALLOW_ACSII))]


def einsum_eq_convert(ixs, iy):
    """
    Generate a einqum eq according to ixs (bonds of contraction tensors) 
    and iy (bonds of resulting tensors)
    """
    uniquelabels = list(set(sum(ixs, start=[]) + iy))
    labelmap = {l:LETTES[i] for i, l in enumerate(uniquelabels)}
    einsum_eq = ",".join(["".join([labelmap[l] for l in ix]) for ix in ixs]) + \
          "->" + "".join([labelmap[l] for l in iy])
    return einsum_eq