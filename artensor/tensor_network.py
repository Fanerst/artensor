from math import log2

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for numerical contraction only
    torch = None


class AbstractTensorNetwork:
    def __init__(
            self, tensor_bonds:dict, bond_dims:dict, 
            final_qubits=[], max_bitstring=1, open_bonds=None
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
        self.log2_bond_dims = {bond: log2(dim) for bond, dim in bond_dims.items()}
        self.open_bonds = set(open_bonds or [])
        self._tensor_ids = tuple(tensor_bonds.keys())
        self._bond_ids = tuple(self.bond_dims.keys())
        self.bond_index = {bond: idx for idx, bond in enumerate(self._bond_ids)}
        self.tensor_bitmasks = {tensor_id: 1 << tensor_id for tensor_id in self._tensor_ids}
        self.bond_bitmasks = {bond: 1 << self.bond_index[bond] for bond in self._bond_ids}
        self.bond_log2_dim_array = [0.0] * len(self._bond_ids)
        for bond, value in self.log2_bond_dims.items():
            self.bond_log2_dim_array[self.bond_index[bond]] = value
        self.bond_tensors = {bond: set() for bond in self.bond_dims.keys()} # determine tensors corresponding to each bond
        for i in tensor_bonds.keys():
            for j in tensor_bonds[i]:
                self.bond_tensors[j].add(i)
        self.bond_tensor_masks = {
            bond: sum(self.tensor_bitmasks[tensor_id] for tensor_id in tensor_ids)
            for bond, tensor_ids in self.bond_tensors.items()
        }
        self.tensor_bond_masks = {
            tensor_id: self.bonds_to_mask(bonds)
            for tensor_id, bonds in self.tensor_bonds.items()
        }
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

    def clone(self):
        clone = object.__new__(AbstractTensorNetwork)
        clone.tensor_bonds = {
            tensor_id: bonds.copy()
            for tensor_id, bonds in self.tensor_bonds.items()
        }
        clone.bond_dims = self.bond_dims.copy()
        clone.log2_bond_dims = self.log2_bond_dims.copy()
        clone.open_bonds = self.open_bonds.copy()
        clone._tensor_ids = self._tensor_ids
        clone._bond_ids = self._bond_ids
        clone.bond_index = self.bond_index
        clone.tensor_bitmasks = self.tensor_bitmasks
        clone.bond_bitmasks = self.bond_bitmasks
        clone.bond_log2_dim_array = self.bond_log2_dim_array
        clone.bond_tensors = {
            bond: tensor_ids.copy()
            for bond, tensor_ids in self.bond_tensors.items()
        }
        clone.bond_tensor_masks = self.bond_tensor_masks.copy()
        clone.tensor_bond_masks = self.tensor_bond_masks.copy()
        clone.final_qubits = self.final_qubits
        clone.num_fq = self.num_fq
        clone.max_bitstring = self.max_bitstring
        clone.log2_max_bitstring = self.log2_max_bitstring
        clone.slicing_bonds = self.slicing_bonds.copy()
        clone.slicing_bond_tensors = {
            bond: tensor_ids.copy()
            for bond, tensor_ids in self.slicing_bond_tensors.items()
        }
        return clone

    def bonds_to_mask(self, bonds):
        mask = 0
        for bond in bonds:
            mask |= self.bond_bitmasks[bond]
        return mask

    def sum_log2_dims_mask(self, mask):
        total = 0.0
        while mask:
            lowest_bit = mask & -mask
            bond_index = lowest_bit.bit_length() - 1
            total += self.bond_log2_dim_array[bond_index]
            mask ^= lowest_bit
        return total

    def mask_to_bonds(self, mask):
        bonds = []
        while mask:
            lowest_bit = mask & -mask
            bonds.append(self._bond_ids[lowest_bit.bit_length() - 1])
            mask ^= lowest_bit
        return bonds
    
    def slicing(self, bond):
        """
        slicing a bond in the tensor network
        """
        assert bond in self.bond_dims.keys()
        assert bond not in self.slicing_bonds.keys()
        dim = self.bond_dims.pop(bond)
        self.log2_bond_dims.pop(bond)
        tensors = self.bond_tensors.pop(bond)
        for tensor_id in tensors:
            self.tensor_bonds[tensor_id].remove(bond)
            self.tensor_bond_masks[tensor_id] &= ~self.bond_bitmasks[bond]
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
        self.log2_bond_dims[bond] = log2(dim)
        self.bond_tensors[bond] = tensors
        for tensor_id in tensors:
            self.tensor_bonds[tensor_id].append(bond)
            self.tensor_bond_masks[tensor_id] |= self.bond_bitmasks[bond]
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

    def _simplify(self, strategy='normal'):
        assert strategy in ['normal', 'sparse']
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
        for x, y in sorted(common_bond_tensors):
            self.contract(*x)

        tensor_bonds_reorder = {}
        reorder_dict = {i:j for i, j in enumerate(self.tensor_bonds.keys())}
        final_qubit_inds = [0] * len(self.final_qubits)
        for i, j in reorder_dict.items():
            if j in self.final_qubits:
                assert len(self.tensor_bonds[j]) == 2
                bond1, bond2 = self.tensor_bonds[j]
                assert bond1.split('-')[1] == bond2.split('-')[1]
                final_qubit_inds[int(bond1.split('-')[1])] = i
                if strategy == 'sparse':
                    assert int(bond1.split('-')[0]) > int(bond2.split('-')[0])
                    new_bonds = [bond2]
                else:
                    new_bonds = self.tensor_bonds[j]
            else:
                new_bonds = self.tensor_bonds[j]
            tensor_bonds_reorder[i] = new_bonds
        return tensor_bonds_reorder, final_qubit_inds

        # for tid in self.final_qubits:
        #     bond_batch = [
        #         bond for bond in self.tensor_bonds[tid] 
        #         if len(self.bond_tensors[bond]) == 1
        #     ]
        #     for bond in bond_batch: self.tensor_bonds[tid].discard(bond)

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
        if torch is None:
            raise ImportError("NumericalTensorNetwork requires PyTorch to be installed.")
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

    # def _simplify(self):
    #     dangling_tensor_id = set([
    #         i for i in self.tensor_bonds.keys() if len(self.tensor_bonds[i]) == 1
    #     ])
    #     while len(dangling_tensor_id) > 0:
    #         new_dangling_id = set([])
    #         for tensor_id in dangling_tensor_id:
    #             assert len(self.tensor_bonds[tensor_id]) == 1
    #             tid_to_contract = self.find_contract_pair(tensor_id)
    #             self.contract(tid_to_contract, tensor_id)
    #             if len(self.tensor_bonds[tid_to_contract]) == 1:
    #                 new_dangling_id.add(tid_to_contract)
    #         dangling_tensor_id = new_dangling_id
    #     matrix_tensor_id = set([
    #         i for i in self.tensor_bonds.keys() 
    #         if len(self.tensor_bonds[i]) == 2 and i not in self.final_qubits
    #     ])
    #     while len(matrix_tensor_id) > 0:
    #         tensor_id = list(matrix_tensor_id)[0]
    #         assert len(self.tensor_bonds[tensor_id]) == 2
    #         tid_to_contract = self.find_contract_pair(tensor_id)
    #         self.contract(tid_to_contract, tensor_id)
    #         matrix_tensor_id = set([
    #             i for i in self.tensor_bonds.keys() 
    #             if len(self.tensor_bonds[i]) == 2 and i not in self.final_qubits
    #         ])

    def _exclude_batch_dim(self):
        for tid in self.final_qubits:
            bond_batch = [
                bond for bond in self.tensor_bonds[tid] 
                if len(self.bond_tensors[bond]) == 1
            ]
            for bond in bond_batch: 
                self.tensor_bonds[tid].remove(bond)
                self.bond_tensors.pop(bond)
