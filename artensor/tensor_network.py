from math import log2


class AbstractTensorNetwork:
    def __init__(
            self, tensor_bonds:dict, bond_dims:dict, 
            final_qubits=[], max_bitstring=1
        ) -> None:
        """
        Class of abstract tensor network
        Parameters:
        -----------
        tensor_bonds: dict of lists
            represent bonds in each individual tensor represented by the key of this dict
        bond_dims: dict
            key is the bond, and value is the bond dimension
        final_qubits: list of ints
            tensor id of final qubits, used for multi-bitstring contraction, the final qubit tensor
            will have additional data dimension, the complexity will be calculated by the factor
        max_bitstring: int
            maximum number of bitstrings to calculate during contraction
        -----------
        """
        self.tensor_bonds = tensor_bonds
        self.bond_dims = bond_dims
        self.bond_tensors = {bond: set() for bond in self.bond_dims.keys()} # determine tensors corresponding to each bond
        for i in tensor_bonds.keys():
            for j in tensor_bonds[i]:
                self.bond_tensors[j].add(i)
        self.final_qubits = final_qubits
        if final_qubits:
            self.num_fq = [1 if i in final_qubits else 0 for i in tensor_bonds.keys()]
        else:
            self.num_fq = [0 for i in tensor_bonds.keys()]
        self.max_bitstring = max_bitstring
        self.log2_max_bitstring = log2(max_bitstring)
        self.slicing_bonds = {}
        self.slicing_bond_tensors = {}
        pass
    
    def slicing(self, bond):
        """
        slicing a bond in the tensor network
        """
        assert bond in self.bond_dims.keys()
        assert bond not in self.slicing_bonds.keys()
        dim = self.bond_dims.pop(bond)
        tensors = self.bond_tensors.pop(bond)
        for tensor_id in tensors:
            self.tensor_bonds[tensor_id].remove(bond)
        self.slicing_bonds[bond] = dim
        self.slicing_bond_tensors[bond] = tensors
    
    def add_bond(self, bond):
        """
        adding a bond that already been sliced back to the tensor network
        """
        assert bond not in self.bond_dims.keys()
        assert bond in self.slicing_bonds.keys()
        dim = self.slicing_bonds.pop(bond)
        tensors = self.slicing_bond_tensors.pop(bond)
        self.bond_dims[bond] = dim
        self.bond_tensors[bond] = tensors
        for tensor_id in tensors:
            self.tensor_bonds[tensor_id].append(bond)
        return tensors

    def contract(self, x, y):
        assert x in self.tensor_bonds.keys()
        assert y in self.tensor_bonds.keys()
        bonds_x = set(self.tensor_bonds.pop(x))
        bonds_y = set(self.tensor_bonds.pop(y))
        contracted_bonds = bonds_x & bonds_y
        bonds_new = (bonds_x | bonds_y) - contracted_bonds
        for bond in contracted_bonds:
            self.bond_tensors.pop(bond)
        for bond in bonds_y - contracted_bonds:
            self.bond_tensors[bond].remove(y)
            self.bond_tensors[bond].add(x)
        self.tensor_bonds[x] = list(bonds_new)

    def find_contract_pair(self, tid):
        possible_tid = set().union(
            *[self.bond_tensors[bond] for bond in self.tensor_bonds[tid]]
        )
        possible_tid.discard(tid)
        tid_to_contract = sorted(
            possible_tid, key=lambda x:len(self.tensor_bonds[x])
        )[-1]
        return tid_to_contract

    def _simplify(self):
        dangling_tensor_id = set([
            i for i in self.tensor_bonds.keys() 
            if len(self.tensor_bonds[i]) == 1 and i not in self.final_qubits
        ])
        while len(dangling_tensor_id) > 0:
            new_dangling_id = set([])
            for tensor_id in dangling_tensor_id:
                assert len(self.tensor_bonds[tensor_id]) == 1
                tid_to_contract = self.find_contract_pair(tensor_id)
                self.contract(tid_to_contract, tensor_id)
                if len(self.tensor_bonds[tid_to_contract]) == 1:
                    new_dangling_id.add(tid_to_contract)
            dangling_tensor_id = new_dangling_id
        matrix_tensor_id = set([
            i for i in self.tensor_bonds.keys() 
            if len(self.tensor_bonds[i]) == 2 and i not in self.final_qubits
        ])
        while len(matrix_tensor_id) > 0:
            tensor_id = list(matrix_tensor_id)[0]
            assert len(self.tensor_bonds[tensor_id]) == 2
            tid_to_contract = self.find_contract_pair(tensor_id)
            self.contract(tid_to_contract, tensor_id)
            matrix_tensor_id = set([
                i for i in self.tensor_bonds.keys() 
                if len(self.tensor_bonds[i]) == 2 and i not in self.final_qubits
            ])
        flipped_bond_tensors = {}
        for key, value in self.bond_tensors.items():
            value = tuple(value)
            if value not in flipped_bond_tensors:
                flipped_bond_tensors[value] = [key]
            else:
                flipped_bond_tensors[value].append(key)
        common_bond_tensors = filter(
            lambda x : len(x[0]) > 1 and len(x[1]) > 1,
            flipped_bond_tensors.items()
        )
        for x, y in common_bond_tensors:
            self.contract(*x)

        # for tid in self.final_qubits:
        #     bond_batch = [
        #         bond for bond in self.tensor_bonds[tid] 
        #         if len(self.bond_tensors[bond]) == 1
        #     ]
        #     for bond in bond_batch: self.tensor_bonds[tid].discard(bond)


import torch

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


class NumericalTensorNetwork(AbstractTensorNetwork):
    def __init__(
            self, tensors:dict, tensor_bonds:dict, bond_dims:dict, 
            final_qubits=[], max_bitstring=1
        ) -> None:
        super().__init__(tensor_bonds, bond_dims, final_qubits, max_bitstring)
        self.tensors = tensors
        assert self.tensor_bonds.keys() == self.tensors.keys()
        self.slicing_indices = {}
    
    def slicing(self, bond):
        """
        slicing a bond in the numerical tensor network
        """
        assert bond in self.bond_dims.keys()
        assert bond not in self.slicing_bonds.keys()
        dim = self.bond_dims.pop(bond)
        tensors = self.bond_tensors.pop(bond)
        for tensor_id in tensors:
            bond_ind = self.tensor_bonds[tensor_id].index(bond)
            self.tensor_bonds[tensor_id].pop(bond_ind)
            if bond not in self.slicing_indices.keys():
                self.slicing_indices[bond] = [(tensor_id, bond_ind)]
            else:
                self.slicing_indices[bond].append([(tensor_id, bond_ind)])
        self.slicing_bonds[bond] = dim
        self.slicing_bond_tensors[bond] = tensors
    
    def contract(self, x, y):
        assert x in self.tensor_bonds.keys()
        assert y in self.tensor_bonds.keys()
        bonds_x, bonds_y = self.tensor_bonds.pop(x), self.tensor_bonds.pop(y)
        contracted_bonds = [bond for bond in bonds_x if bond in bonds_y]
        bonds_new = [
            bond for bond in bonds_x + bonds_y if bond not in contracted_bonds
        ]
        for bond in contracted_bonds:
            self.bond_tensors.pop(bond)
        for bond in [bond for bond in bonds_y if bond not in contracted_bonds]:
            self.bond_tensors[bond].remove(y)
            self.bond_tensors[bond].add(x)
        self.tensor_bonds[x] = bonds_new
        # print(self.tensors[x].shape, bonds_x, self.tensors[y].shape, bonds_y)
        # print(bonds_new, einsum_eq_convert((bonds_x, bonds_y), bonds_new))
        self.tensors[x] = torch.einsum(
            einsum_eq_convert((bonds_x, bonds_y), bonds_new), 
            self.tensors.pop(x), self.tensors.pop(y)
        )

    def find_contract_pair(self, tid):
        possible_tid = set().union(
            *[self.bond_tensors[bond] for bond in self.tensor_bonds[tid]]
        )
        possible_tid.discard(tid)
        tid_to_contract = sorted(
            possible_tid, key=lambda x:len(self.tensor_bonds[x])
        )[-1]
        return tid_to_contract

    def _simplify(self):
        dangling_tensor_id = set([
            i for i in self.tensor_bonds.keys() if len(self.tensor_bonds[i]) == 1
        ])
        while len(dangling_tensor_id) > 0:
            new_dangling_id = set([])
            for tensor_id in dangling_tensor_id:
                assert len(self.tensor_bonds[tensor_id]) == 1
                tid_to_contract = self.find_contract_pair(tensor_id)
                self.contract(tid_to_contract, tensor_id)
                if len(self.tensor_bonds[tid_to_contract]) == 1:
                    new_dangling_id.add(tid_to_contract)
            dangling_tensor_id = new_dangling_id
        matrix_tensor_id = set([
            i for i in self.tensor_bonds.keys() 
            if len(self.tensor_bonds[i]) == 2 and i not in self.final_qubits
        ])
        while len(matrix_tensor_id) > 0:
            tensor_id = list(matrix_tensor_id)[0]
            assert len(self.tensor_bonds[tensor_id]) == 2
            tid_to_contract = self.find_contract_pair(tensor_id)
            self.contract(tid_to_contract, tensor_id)
            matrix_tensor_id = set([
                i for i in self.tensor_bonds.keys() 
                if len(self.tensor_bonds[i]) == 2 and i not in self.final_qubits
            ])

    def _exclude_batch_dim(self):
        for tid in self.final_qubits:
            bond_batch = [
                bond for bond in self.tensor_bonds[tid] 
                if len(self.bond_tensors[bond]) == 1
            ]
            for bond in bond_batch: 
                self.tensor_bonds[tid].remove(bond)
                self.bond_tensors.pop(bond)