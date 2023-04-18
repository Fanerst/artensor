import torch
from .contraction_tree import ContractionTree
import numpy as np
from math import ceil
from traceback import print_exc
import sys


allow_ascii = list(range(65, 90)) + list(range(97, 122))
letters = [chr(allow_ascii[i]) for i in range(len(allow_ascii))]


def einsum_eq_convert(ixs, iy):
    """
    Generate a einqum eq according to ixs (bonds of contraction tensors) and iy (bonds of resulting tensors)
    """
    uniquelabels = list(set(sum(ixs, start=[]) + iy))
    labelmap = {l:letters[i] for i, l in enumerate(uniquelabels)}
    einsum_eq = ",".join(["".join([labelmap[l] for l in ix]) for ix in ixs]) + "->" + "".join([labelmap[l] for l in iy])
    return einsum_eq


def contraction_scheme(ctree:ContractionTree):
    """
    Compile a contraction scheme according to the contraction tree in a depth-first search way
    """
    ctree.mark_rep_tensor()
    stack = [ctree.tree[ctree.all_tensors]]
    scheme = []
    while len(stack):
        vertex = stack.pop()
        if vertex.left and vertex.right:
            if vertex.left.is_leaf():
                ix_left = ctree.tn.tensor_bonds[vertex.left.rep_tensor]
            else:
                ix_left = list(vertex.left.contain_bonds)
            if vertex.right.is_leaf():
                ix_right = ctree.tn.tensor_bonds[vertex.right.rep_tensor]
            else:
                ix_right = list(vertex.right.contain_bonds)
            if vertex.rep_tensor == vertex.left.rep_tensor:
                order = (vertex.left.rep_tensor, vertex.right.rep_tensor)
                ixs = (ix_left, ix_right)
            elif vertex.rep_tensor == vertex.right.rep_tensor:
                order = (vertex.right.rep_tensor, vertex.left.rep_tensor)
                ixs = (ix_right, ix_left)
            else:
                raise ValueError('Incorrect rep tensor mark process.')
            iy = list(vertex.contain_bonds)
            if vertex == ctree.tree[ctree.all_tensors]:
                output_bonds = iy
            einsum_eq = einsum_eq_convert(ixs, iy)
            scheme.append((order, einsum_eq))
            if vertex.left.sc > vertex.right.sc:
                stack += [vertex.left, vertex.right]
            else:
                stack += [vertex.right, vertex.left]
    scheme.reverse()
    return scheme, output_bonds


def tensor_contraction(tensors, scheme):
    """
    perform the tensor contraction
    """
    for s in scheme:
        i, j = s[0]
        einsum_eq = s[1]
        try:
            tensors[i] = torch.einsum(einsum_eq, tensors[i], tensors[j])
        except:
            print(s, len(tensors[i].shape), len(tensors[j].shape))
            print_exc()
            sys.exit(1)
    
    return tensors[i]


def tensordot2einsum(len_i, len_j, idxi_j, idxj_i, permute=None):
    if idxi_j and idxj_i:
        len_ij = len(idxi_j)
    else:
        len_ij = 0
    if permute:
        assert len(permute) == len_i + len_j - 2 * len_ij
    eq_i = []
    eq_i_uncontract = []
    for i in range(len_i):
        eq_i.append(letters[i])
        if i not in idxi_j:
            eq_i_uncontract.append(letters[i])
    eq_j = ['' for i in range(len_j)]
    eq_j_uncontract = list(letters[len_i:(len_i + len_j - len_ij)])
    for i, j in zip(idxi_j, idxj_i):
        eq_j[j] = eq_i[i]
    count = len_i
    for i in range(len_j):
        if i not in idxj_i:
            eq_j[i] = letters[count]
            count += 1
    eq_i = ''.join(eq_i)
    eq_j = ''.join(eq_j)
    eq_result = eq_i_uncontract + eq_j_uncontract
    eq_result = ''.join([eq_result[i] for i in permute]) if permute else ''.join(eq_result)
    
    einsum_eq = eq_i + ',' + eq_j + '->' + eq_result
    return einsum_eq


def index_select(bitstrings, inds):
    '''
    select bitstrings in specific indices

    :param bitstrings: list of bitstring
    :param inds: indices to be selected
    '''
    return [''.join(bitstring[i] for i in inds) for bitstring in bitstrings]


def combine_bitstring(bitstring_i, bitstring_j, loc_i, loc_j):
    '''
    combine bitstring with two partial ones with their location

    :param bitstring_i: bitstring i to be combined
    :param bitstring_j: bitstring j to be combined
    :param loc_i: location of bitstring i
    :param loc_j: location of bitstring j
    '''
    return ''.join([bitstring_i[loc_i.index(k)] if k in loc_i else bitstring_j[loc_j.index(k)] for k in range(len(loc_i) + len(loc_j))])


def tensor_contraction_sparse(tensors, contraction_scheme, scientific_notation=False):
    if scientific_notation:
        factor = torch.tensor(0, dtype=tensors[0].dtype, device=tensors[0].device)

    for step in contraction_scheme:
        i, j = step[0]
        batch_i, batch_j = step[2]
        try:
            if len(batch_i) > 1:
                tensors[i] = [tensors[i]]
                for k in range(len(batch_i)-1, -1, -1):
                    if k != 0:
                        if step[3]:
                            tensors[i].insert(
                                1, 
                                torch.einsum(
                                    step[1],
                                    tensors[i][0][batch_i[k]], 
                                    tensors[j][batch_j[k]], 
                                ).reshape(step[3])
                            )
                        else:
                            tensors[i].insert(
                                1, 
                                torch.einsum(
                                    step[1],
                                    tensors[i][0][batch_i[k]], 
                                    tensors[j][batch_j[k]])
                            )
                    else:
                        if step[3]:
                            tensors[i][0] = torch.einsum(
                                step[1],
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]], 
                            ).reshape(step[3])
                        else:
                            tensors[i][0] = torch.einsum(
                                step[1],
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]], 
                            )
                tensors[j] = []
                tensors[i] = torch.cat(tensors[i], dim=0)
            elif len(step) > 3 and len(batch_i) == len(batch_j) == 1:
                tensors[i] = tensors[i][batch_i[0]]
                tensors[j] = tensors[j][batch_j[0]]
                tensors[i] = torch.einsum(step[1], tensors[i], tensors[j])
            elif len(step) > 3:
                tensors[i] = torch.einsum(
                    step[1],
                    tensors[i],
                    tensors[j],
                ).reshape(step[3])
                if len(batch_i) == 1:
                    tensors[i] = tensors[i][batch_i[0]]
                tensors[j] = []
            else:
                tensors[i] = torch.einsum(step[1], tensors[i], tensors[j])
                tensors[j] = []
        except:
            print(step)
            print_exc()
            sys.exit(1)

        if scientific_notation:
            norm_factor = tensors[i].abs().max()
            tensors[i] /= norm_factor
            factor += torch.log10(norm_factor)

    if scientific_notation:
        return factor, tensors[i]
    else:
        return tensors[i]


def contraction_scheme_sparse(ctree:ContractionTree, bitstrings=None, sc_target=31):
    """
    Compile a contraction scheme according to the contraction tree in a depth-first search way
    """
    import time
    order = ctree.tree_order_dfs() # ctree.tree_to_order() # 
    tensor_bonds = ctree.tn.tensor_bonds
    contraction_scheme = []
    final_qubits = ctree.tn.final_qubits
    if type(final_qubits) == frozenset or type(final_qubits) == set:
        final_qubits = sorted(list(final_qubits))
    tensor_bitstrings = [np.array([0, 1]) for _ in range(len(final_qubits))]
    tensor_info = [([], np.array([-1])) if k not in final_qubits else ([final_qubits.index(k)], tensor_bitstrings[final_qubits.index(k)]) for k in tensor_bonds.keys()]
    
    for edge in order:
        i, j = edge
        bond_i, bond_j = tensor_bonds[i], tensor_bonds[j]

        common_indices = sorted(frozenset(bond_i) & frozenset(bond_j))
        uncontract_indices = []
        for ind in common_indices:
            for x in tensor_bonds.keys():
                if x == i or x == j or len(tensor_bonds[x]) == 0:
                    continue
                if ind in tensor_bonds[x]:
                    uncontract_indices.append(ind)
                    break
        contract_indices = [ind for ind in common_indices if ind not in uncontract_indices]

        idxi_j = []
        idxj_i = []
        for idx in contract_indices:
            idxi_j.append(bond_i.index(idx))
            idxj_i.append(bond_j.index(idx))
        tensor_bonds[i] = [bond_i[m] for m in range(len(bond_i)) if m not in idxi_j] 
        tensor_bonds[i] += [bond_j[n] for n in range(len(bond_j)) if n not in idxj_i and bond_j[n] not in tensor_bonds[i]]
        tensor_bonds[j] = []
    
        cat_batch_flag = False
        tmp_final_qubit_ids = sorted(tensor_info[i][0] + tensor_info[j][0])
        if len(tmp_final_qubit_ids) == 0:
            batch_seq = [[torch.tensor([0])], [torch.tensor([0])]]
            tmp_bitstrings_rep = np.array([-1])
        elif len(tensor_info[i][0]) > 0 and len(tensor_info[j][0]) == 0:
            batch_seq = [[torch.tensor([k for k in range(len(tensor_info[i][1]))])], [torch.tensor([0])]]
            tmp_bitstrings_rep = tensor_info[i][1]
        elif len(tensor_info[j][0]) > 0 and len(tensor_info[i][0]) == 0:
            batch_seq = [[torch.tensor([0])], [torch.tensor([k for k in range(len(tensor_info[j][1]))])]]
            tmp_bitstrings_rep = tensor_info[j][1]
        else:
            loc_i, loc_j = [tmp_final_qubit_ids.index(item) for item in tensor_info[i][0]], [tmp_final_qubit_ids.index(item) for item in tensor_info[j][0]]
            idx = int(len(tensor_info[i][1]) > len(tensor_info[j][1]))
            tmp_bitstrings = np.unique(index_select(bitstrings, tmp_final_qubit_ids))
            if len(tmp_bitstrings) == 2 ** len(tmp_final_qubit_ids) or len(tmp_final_qubit_ids) + len(tensor_bonds[i]) <= sc_target:
                tmp_bitstrings_rep = np.array([
                    int(combine_bitstring(np.binary_repr(x, len(tensor_info[i][0])), np.binary_repr(y, len(tensor_info[j][0])), loc_i, loc_j), 2) for x in tensor_info[i][1] for y in tensor_info[j][1] 
                ])
                if len(tmp_bitstrings) != len(tmp_bitstrings_rep):
                    remain_ind = np.sort(np.array([np.argwhere(tmp_bitstrings_rep == int(string, 2))[0][0] for string in tmp_bitstrings])) # sorted([tmp_bitstrings_rep.index(int(string, 2)) for string in tmp_bitstrings])
                    tmp_bitstrings_rep = np.array([tmp_bitstrings_rep[ind] for ind in remain_ind])
                    batch_seq = [[torch.tensor(remain_ind)], []]
                else:
                    batch_seq = [[], []]
            else:
                bitstrings_partials = np.stack(
                    [np.array([int(string, 2) for string in index_select(tmp_bitstrings, loc_i)]),
                     np.array([int(string, 2) for string in index_select(tmp_bitstrings, loc_j)])]
                )
                tmp_bitstrings_rep = np.array([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in tmp_bitstrings])
                batch_seq_init = np.array([[np.argwhere(tensor_info[i][1] == b_i)[0][0], np.argwhere(tensor_info[j][1] == b_j)[0][0]] for b_i, b_j in zip(bitstrings_partials[0], bitstrings_partials[1])])
                sort_inds = np.argsort(batch_seq_init[:, 1-idx])
                batch_seq_init = batch_seq_init[sort_inds].T.reshape(2, -1)

                batch_seq = [
                    [torch.from_numpy(batch_seq_init[0])],
                    [torch.from_numpy(batch_seq_init[1])],
                ]
                # print(len(batch_seq[0][0]), torch.max(batch_seq[0][0]), len(tensor_info[i][1]), torch.max(batch_seq[1][0]), len(tensor_info[j][1]))
                assert torch.max(batch_seq[0][0]) < len(tensor_info[i][1])
                assert torch.max(batch_seq[1][0]) < len(tensor_info[j][1])
                num_seperated_seq = 2 ** ceil(max(0, np.log2(len(tmp_bitstrings_rep)) + max(len(bond_i), len(bond_j)) - (sc_target - 2)))
                # print((i, j), len(tmp_bitstrings_rep), len(bond_i), len(bond_j), sc_target, num_seperated_seq)
                if num_seperated_seq > 1:
                    seq_length = int(len(tmp_bitstrings_rep) / num_seperated_seq)
                    if len(tmp_bitstrings_rep) % num_seperated_seq > 0:
                        num_seperated_seq += 1
                    batch_seq = [
                        [batch_seq[0][0][i*seq_length:(i+1)*seq_length] for i in range(num_seperated_seq)], 
                        [batch_seq[1][0][i*seq_length:(i+1)*seq_length] for i in range(num_seperated_seq)]
                    ]
                cat_batch_flag = True

                tmp_bitstrings_rep = tmp_bitstrings_rep[sort_inds]
            assert len(tmp_bitstrings_rep) == len(tmp_bitstrings)

        iy = []
        if len(tensor_info[j][0]):
            permute_dim_j = 1
            if cat_batch_flag:
                ix_right = [-3] + bond_j
                iy.insert(0, -3)
            else:
                ix_right = [-2] + bond_j
                iy.insert(0, -2)
        else:
            permute_dim_j = 0
            ix_right = bond_j
        if len(tensor_info[i][0]):
            permute_dim_i = 1
            if cat_batch_flag:
                ix_left = [-3] + bond_i
            else:
                ix_left = [-1] + bond_i
                iy.insert(0, -1)
        else:
            permute_dim_i = 0
            ix_left = bond_i
        iy = iy + tensor_bonds[i]
        einsum_eq = einsum_eq_convert((ix_left, ix_right), iy)
        if permute_dim_i and permute_dim_j:
            next_tensor_shape = (len(tmp_bitstrings_rep),) + (2,) * len(tensor_bonds[i])
            if cat_batch_flag:
                contraction_scheme.append((edge, einsum_eq, batch_seq, None, next_tensor_shape))
            else:
                rshape = (-1,) + (2,) * len(tensor_bonds[i])
                contraction_scheme.append((edge, einsum_eq, batch_seq, rshape, next_tensor_shape))
        else:
            contraction_scheme.append((edge, einsum_eq, batch_seq))
        
        tensor_info[i] = (tmp_final_qubit_ids, tmp_bitstrings_rep)
    
        if edge == order[-1]:
            tmp_bitstrings = [np.binary_repr(n, len(final_qubits)) for n in tensor_info[i][1]]

    return contraction_scheme, tensor_bonds[i], tmp_bitstrings