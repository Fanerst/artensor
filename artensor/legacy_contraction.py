import torch
from .core import ContractionTree
import numpy as np
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
    stack = [ctree.tree[ctree.all_tensors]]
    scheme = []
    while len(stack):
        vertex = stack.pop()
        if vertex.left and vertex.right:
            rep_tensors = [min(vertex.left.contain_tensors), min(vertex.right.contain_tensors)]
            order = (min(rep_tensors), max(rep_tensors))
            if vertex.left.is_leaf():
                ix_left = ctree.tn.tensor_bonds[rep_tensors[0]]
            else:
                ix_left = list(vertex.left.contain_bonds)
            if vertex.right.is_leaf():
                ix_right = ctree.tn.tensor_bonds[rep_tensors[1]]
            else:
                ix_right = list(vertex.right.contain_bonds)
            ixs, iy = (ix_left, ix_right), list(vertex.contain_bonds)
            einsum_eq = einsum_eq_convert(ixs, iy)
            scheme.append((order, einsum_eq))
            if vertex.left.sc > vertex.right.sc:
                stack += [vertex.left, vertex.right]
            else:
                stack += [vertex.right, vertex.left]
    scheme.reverse()
    return scheme


def tensor_contraction(tensors, scheme):
    """
    perform the tensor contraction
    """
    for s in scheme:
        i, j = s[0]
        einsum_eq = s[1]
        tensors[i] = torch.einsum(einsum_eq, tensors[i], tensors[j])
    
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

# def sort_bitstrings_wrt_loc(bitstrings, loc_i, loc_j, rep_i, repj):
#     return sorted(bitstrings, key=lambda x:(''.join([x[loc] for loc in loc_i]), ''.join([x[loc] for loc in loc_j])))
def sort_bitstrings_wrt_loc(bitstrings, loc_i, loc_j, dict_i, dict_j):
    # dict_i = {np.binary_repr(x, len(loc_i)):rep_i.index(x) for x in rep_i}
    # dict_j = {np.binary_repr(x, len(loc_j)):rep_j.index(x) for x in rep_j}
    return sorted(bitstrings, key=lambda x:(dict_i[''.join([x[loc] for loc in loc_i])], dict_j[''.join([x[loc] for loc in loc_j])]))

def contraction_scheme_sparsestate(tensor_bonds, order, final_qubits=[], bitstrings=[], batch_loc=0):
    '''
    construct a scheme by given order with multiple bitstings

    :param tensor_bonds: list of equations represent the tensor network
    :param order: given order
    :param final_qubits: ids of which tensor is a final qubit
    :param bitstrings: bitstrings will be sampled in the simulation
    :param batch_loc: 0 or -1, select where to put the batch dimension, 0 for the 0th dimension and -1 for the last dimentsion

    :return contraction_scheme: constructed contraction scheme
    :return tensor_bonds[i]: equation of final tensor
    :return tmp_bitstrings: ordered corresponding bitstrings to amplitudes calculated according to contraction scheme
    '''
    contraction_scheme = []
    if type(final_qubits) == frozenset or type(final_qubits) == set:
        final_qubits = sorted(list(final_qubits))

    # tensor_info = []
    # for k in range(len(tensor_bonds)):
    #     if k not in final_qubits:
    #         tensor_info.append(([], np.array([-1])))
    #     else:
    #         final_idx = [final_qubits.index(k)]
    #         x = np.unique(index_select(bitstrings, final_idx))
    #         print(k, final_idx, x)
    #         if len(x) == 2:
    #             tensor_info.append((final_idx, np.array([0, 1])))
    #         elif x[0] == '0':
    #             tensor_info.append((final_idx, np.array([0])))
    #         else:
    #             assert x[0] == '1'
    #             tensor_info.append((final_idx, np.array([1])))

    tensor_bitstrings = [np.array([0, 1]) for _ in range(len(final_qubits))]
    tensor_info = [([], np.array([-1])) if k not in final_qubits else ([final_qubits.index(k)], tensor_bitstrings[final_qubits.index(k)]) for k in range(len(tensor_bonds))]
    
    for edge in order:
        i, j = edge
        bond_i, bond_j = tensor_bonds[i], tensor_bonds[j]

        tmp_final_qubit_ids = sorted(tensor_info[i][0] + tensor_info[j][0])
        if len(tmp_final_qubit_ids) == 0:
            batch_sep_sorted = [[torch.tensor([0])], [torch.tensor([0])]]
            tmp_bitstrings_rep = np.array([-1])
        elif len(tensor_info[i][0]) > 0  and len(tensor_info[j][0]) == 0:
            batch_sep_sorted = [[torch.tensor([k for k in range(len(tensor_info[i][1]))])], [torch.tensor([0])]]
            tmp_bitstrings_rep = tensor_info[i][1]
        elif len(tensor_info[j][0]) > 0  and len(tensor_info[i][0]) == 0:
            batch_sep_sorted = [[torch.tensor([0])], [torch.tensor([k for k in range(len(tensor_info[j][1]))])]]
            tmp_bitstrings_rep = tensor_info[j][1]
        else:
            loc_i, loc_j = [tmp_final_qubit_ids.index(item) for item in tensor_info[i][0]], [tmp_final_qubit_ids.index(item) for item in tensor_info[j][0]]
            idx = int(len(tensor_info[i][1]) > len(tensor_info[j][1]))
            tmp_bitstrings = np.unique(index_select(bitstrings, tmp_final_qubit_ids))
            tmp_bitstrings_rep = np.array([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in tmp_bitstrings])
            bitstrings_partial_i, bitstrings_partial_j = index_select(tmp_bitstrings, loc_i), index_select(tmp_bitstrings, loc_j)
            bitstrings_rep_i, bitstrings_rep_j = np.array([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in bitstrings_partial_i]), np.array([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in bitstrings_partial_j])
            batch_sep = np.array([[np.argwhere(tensor_info[i][1] == b_i)[0][0], np.argwhere(tensor_info[j][1] == b_j)[0][0]] for b_i, b_j in zip(bitstrings_rep_i, bitstrings_rep_j)])
            sort_inds = np.argsort(batch_sep[:, idx])
            batch_sep = batch_sep[sort_inds].T.reshape(2, -1)
            uni, inds = np.unique(batch_sep[idx], return_index=True)
            inds = list(inds) + [len(batch_sep[idx])]
            batch_sep_sorted = [[], []]
            for k in range(len(uni)):
                batch_sep_sorted[idx].append(torch.tensor([uni[k]]))
                batch_sep_sorted[1-idx].append(torch.from_numpy(batch_sep[1-idx][inds[k]:inds[k+1]]))
            tmp_bitstrings_rep = tmp_bitstrings_rep[sort_inds]

        common_indices = sorted(frozenset(bond_i) & frozenset(bond_j))

        idxi_j = []
        idxj_i = []
        for idx in common_indices:
            idxi_j.append(bond_i.index(idx))
            idxj_i.append(bond_j.index(idx))
        tensor_bonds[i] = [bond_i[m] for m in range(len(bond_i)) if m not in idxi_j] + [bond_j[n] for n in range(len(bond_j)) if n not in idxj_i]

        permute_dim_i = 1 if len(tensor_info[i][0]) else 0
        permute_dim_j = 1 if len(tensor_info[j][0]) else 0
        idxi_j = [ind + 1 for ind in idxi_j] if len(tensor_info[i][0]) and batch_loc == 0 else idxi_j
        idxj_i = [ind + 1 for ind in idxj_i] if len(tensor_info[j][0]) and batch_loc == 0 else idxj_i
        if permute_dim_j == 1 and batch_loc == 0:
            # print(permute_dim_i, permute_dim_j, len(bond_i), len(bond_j), len(common_indices))
            permute_seq = list(range(permute_dim_i + permute_dim_j + len(bond_i) + len(bond_j) - 2 * len(common_indices)))
            si = permute_seq.pop(permute_dim_i + len(bond_i) - len(common_indices))
            permute_seq.insert(permute_dim_i, si)
            rshape = (-1,) + (2,) * (len(bond_i) + len(bond_j) - 2 * len(common_indices))
            next_tensor_shape = (len(tmp_bitstrings_rep),) + (2,) * (len(bond_i) + len(bond_j) - 2 * len(common_indices))
            contraction_scheme.append((edge, 'tensordot', idxi_j, idxj_i, batch_sep_sorted, permute_seq, rshape, next_tensor_shape))
        elif permute_dim_i == 1 and batch_loc == -1:
            # print(permute_dim_i, permute_dim_j, len(bond_i), len(bond_j), len(common_indices))
            permute_seq = list(range(permute_dim_i + permute_dim_j + len(bond_i) + len(bond_j) - 2 * len(common_indices)))
            si = permute_seq.pop(permute_dim_i + len(bond_i) - len(common_indices)-1)
            if permute_dim_j:
                permute_seq.insert(-1, si)
            else:
                permute_seq.append(si)
            rshape = (2,) * (len(bond_i) + len(bond_j) - 2 * len(common_indices)) + (-1,)
            next_tensor_shape = (2,) * (len(bond_i) + len(bond_j) - 2 * len(common_indices)) + (len(tmp_bitstrings_rep),)
            contraction_scheme.append((edge, 'tensordot', idxi_j, idxj_i, batch_sep_sorted, permute_seq, rshape, next_tensor_shape))
        else:
            contraction_scheme.append((edge, 'tensordot', idxi_j, idxj_i, batch_sep_sorted))
        
        tensor_info[i] = (tmp_final_qubit_ids, tmp_bitstrings_rep)
    
        if edge == order[-1]:
            tmp_bitstrings = [np.binary_repr(n, len(final_qubits)) for n in tensor_info[i][1]]

    return contraction_scheme, tensor_bonds[i], tmp_bitstrings


def contraction_scheme_sparsestate_1(tensor_bonds, order, final_qubits=None, bitstrings=None, batch_loc=0):
    '''
    construct a scheme by given order with multiple bitstings

    :param tensor_bonds: list of equations represent the tensor network
    :param order: given order
    :param final_qubits: ids of which tensor is a final qubit
    :param bitstrings: bitstrings will be sampled in the simulation
    :param batch_loc: 0 or -1, select where to put the batch dimension, 0 for the 0th dimension and -1 for the last dimentsion

    :return contraction_scheme: constructed contraction scheme
    :return tensor_bonds[i]: equation of final tensor
    :return tmp_bitstrings: ordered corresponding bitstrings to amplitudes calculated according to contraction scheme
    '''
    contraction_scheme = []
    if type(final_qubits) == frozenset or type(final_qubits) == set:
        final_qubits = sorted(list(final_qubits))
    tensor_bitstrings = [[0, 1] for _ in range(len(final_qubits))]
    tensor_info = [([], [-1]) if k not in final_qubits else ([final_qubits.index(k)], tensor_bitstrings[final_qubits.index(k)]) for k in range(len(tensor_bonds))]
    
    for edge in order:
        i, j = edge
        bond_i, bond_j = tensor_bonds[i], tensor_bonds[j]

        tmp_final_qubit_ids = sorted(tensor_info[i][0] + tensor_info[j][0])
        if len(tmp_final_qubit_ids) == 0:
            batch_sep_sorted = [[torch.tensor([0])], [torch.tensor([0])]]
            tmp_bitstrings_rep = [-1]
        elif len(tensor_info[i][0]) > 0  and len(tensor_info[j][0]) == 0:
            batch_sep_sorted = [[torch.tensor([k for k in range(len(tensor_info[i][1]))])], [torch.tensor([0])]]
            tmp_bitstrings_rep = tensor_info[i][1]
        elif len(tensor_info[j][0]) > 0  and len(tensor_info[i][0]) == 0:
            batch_sep_sorted = [[torch.tensor([0])], [torch.tensor([k for k in range(len(tensor_info[j][1]))])]]
            tmp_bitstrings_rep = tensor_info[j][1]
        else:
            loc_i, loc_j = [tmp_final_qubit_ids.index(item) for item in tensor_info[i][0]], [tmp_final_qubit_ids.index(item) for item in tensor_info[j][0]]
            idx = int(len(tensor_info[i][1]) > len(tensor_info[j][1]))
            tmp_bitstrings = np.unique(index_select(bitstrings, tmp_final_qubit_ids))
            if idx == 0:
                if len(tmp_bitstrings) == 2 ** len(tmp_final_qubit_ids):
                    batch_sep_sorted = [
                        [torch.tensor([item]) for item in range(len(tensor_info[i][1]))], 
                        [torch.arange(len(tensor_info[j][1]))] * len(tensor_info[i][1])
                    ]
                    tmp_bitstrings_rep = [
                        int(combine_bitstring(np.binary_repr(x, len(tensor_info[i][0])), np.binary_repr(y, len(tensor_info[j][0])), loc_i, loc_j), 2) for x in tensor_info[i][1] for y in tensor_info[j][1] 
                    ]
                else:
                    dict_i = {np.binary_repr(x, len(loc_i)):tensor_info[i][1].index(x) for x in tensor_info[i][1]}
                    dict_j = {np.binary_repr(x, len(loc_j)):tensor_info[j][1].index(x) for x in tensor_info[j][1]}
                    tmp_bitstrings_sorted = sort_bitstrings_wrt_loc(tmp_bitstrings, loc_i, loc_j, dict_i, dict_j)
                    bitstrings_partial_i, bitstrings_partial_j = index_select(tmp_bitstrings_sorted, loc_i), index_select(tmp_bitstrings_sorted, loc_j)
                    tmp_bitstrings_rep = [int(bitstring, 2) for bitstring in tmp_bitstrings_sorted]
                    batch_sep_sorted = [[], []]
                    for k in range(len(tmp_bitstrings_sorted)):
                        tmp_bi_rep = dict_i[bitstrings_partial_i[k]]
                        if tmp_bi_rep not in batch_sep_sorted[idx]:
                            batch_sep_sorted[idx].append(tmp_bi_rep)
                            batch_sep_sorted[1-idx].append([])
                        batch_sep_sorted[1-idx][-1].append(dict_j[bitstrings_partial_j[k]])
                    batch_sep_sorted = [[torch.tensor([i]) for i in batch_sep_sorted[0]], [torch.tensor(i) for i in batch_sep_sorted[1]]]
                    # print(i, j, tensor_info[i], tensor_info[j])
                    # print(tmp_bitstrings_rep)
                    # print(batch_sep_sorted)
                    # print('-'*20)
                    # tmp_bitstrings_rep_unsort = set([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in tmp_bitstrings])
                    # tmp_bitstrings_rep = []
                    # batch_sep_sorted = [[], []]
                    # for idi in range(len(tensor_info[i][1])):
                    #     x = tensor_info[i][1][idi]
                    #     batch_sep_sorted[idx].append(torch.tensor([idi]))
                    #     batch_sep_sorted[1-idx].append([])
                    #     for idj in range(len(tensor_info[j][1])):
                    #         y = tensor_info[j][1][idj]
                    #         combined_bitstring_rep = int(combine_bitstring(np.binary_repr(x, len(tensor_info[i][0])), np.binary_repr(y, len(tensor_info[j][0])), loc_i, loc_j), 2)
                    #         if combined_bitstring_rep in tmp_bitstrings_rep_unsort:
                    #             tmp_bitstrings_rep.append(combined_bitstring_rep)
                    #             batch_sep_sorted[1-idx][idi].append(idj)
                    #     batch_sep_sorted[1-idx][idi] = torch.tensor(batch_sep_sorted[1-idx][idi])
                    # print(tmp_bitstrings_rep)
                    # print(batch_sep_sorted)
                    # print('-'*40)
            else:
                if len(tmp_bitstrings) == 2 ** len(tmp_final_qubit_ids):
                    batch_sep_sorted = [
                        [torch.arange(len(tensor_info[i][1]))] * len(tensor_info[j][1]),
                        [torch.tensor([item]) for item in range(len(tensor_info[j][1]))], 
                    ]
                    tmp_bitstrings_rep = [
                        int(combine_bitstring(np.binary_repr(x, len(tensor_info[i][0])), np.binary_repr(y, len(tensor_info[j][0])), loc_i, loc_j), 2) for y in tensor_info[j][1] for x in tensor_info[i][1]
                    ]
                else:
                    dict_i = {np.binary_repr(x, len(loc_i)):tensor_info[i][1].index(x) for x in tensor_info[i][1]}
                    dict_j = {np.binary_repr(x, len(loc_j)):tensor_info[j][1].index(x) for x in tensor_info[j][1]}
                    tmp_bitstrings_sorted = sort_bitstrings_wrt_loc(tmp_bitstrings, loc_j, loc_i, dict_j, dict_i)
                    bitstrings_partial_i, bitstrings_partial_j = index_select(tmp_bitstrings_sorted, loc_i), index_select(tmp_bitstrings_sorted, loc_j)
                    tmp_bitstrings_rep = [int(bitstring, 2) for bitstring in tmp_bitstrings_sorted]
                    batch_sep_sorted = [[], []]
                    for k in range(len(tmp_bitstrings_sorted)):
                        tmp_bj_rep = dict_j[bitstrings_partial_j[k]]
                        if tmp_bj_rep not in batch_sep_sorted[idx]:
                            batch_sep_sorted[idx].append(tmp_bj_rep)
                            batch_sep_sorted[1-idx].append([])
                        batch_sep_sorted[1-idx][-1].append(dict_i[bitstrings_partial_i[k]])
                    batch_sep_sorted = [[torch.tensor(i) for i in batch_sep_sorted[0]], [torch.tensor([i]) for i in batch_sep_sorted[1]]]
                    # print(i, j, tensor_info[i], tensor_info[j])
                    # print(tmp_bitstrings_rep)
                    # print(batch_sep_sorted)
                    # print('-'*20)
                    # tmp_bitstrings_rep_unsort = set([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in tmp_bitstrings])
                    # tmp_bitstrings_rep = []
                    # batch_sep_sorted = [[], []]
                    # for idj in range(len(tensor_info[j][1])):
                    #     y = tensor_info[j][1][idj]
                    #     batch_sep_sorted[idx].append(torch.tensor([idj]))
                    #     batch_sep_sorted[1-idx].append([])
                    #     for idi in range(len(tensor_info[i][1])):
                    #         x = tensor_info[i][1][idi]
                    #         combined_bitstring_rep = int(combine_bitstring(np.binary_repr(x, len(tensor_info[i][0])), np.binary_repr(y, len(tensor_info[j][0])), loc_i, loc_j), 2)
                    #         if combined_bitstring_rep in tmp_bitstrings_rep_unsort:
                    #             tmp_bitstrings_rep.append(combined_bitstring_rep)
                    #             batch_sep_sorted[1-idx][idj].append(idi)
                    #     batch_sep_sorted[1-idx][idj] = torch.tensor(batch_sep_sorted[1-idx][idj])
                    # print(tmp_bitstrings_rep)
                    # print(batch_sep_sorted)
                    # print('-'*40)
            assert len(tmp_bitstrings_rep) == len(tmp_bitstrings)
            # else:
            #     tmp_bitstrings_rep = np.array([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in tmp_bitstrings])
            #     bitstrings_partial_i, bitstrings_partial_j = index_select(tmp_bitstrings, loc_i), index_select(tmp_bitstrings, loc_j)
            #     bitstrings_rep_i, bitstrings_rep_j = np.array([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in bitstrings_partial_i]), np.array([int(bitstring, 2) if len(bitstring) > 0 else -1 for bitstring in bitstrings_partial_j])
            #     batch_sep = np.array([[np.argwhere(tensor_info[i][1] == b_i)[0][0], np.argwhere(tensor_info[j][1] == b_j)[0][0]] for b_i, b_j in zip(bitstrings_rep_i, bitstrings_rep_j)])
            #     sort_inds = np.argsort(batch_sep[:, idx])
            #     batch_sep = batch_sep[sort_inds].T.reshape(2, -1)
            #     uni, inds = np.unique(batch_sep[idx], return_index=True)
            #     inds = list(inds) + [len(batch_sep[idx])]
            #     batch_sep_sorted = [[], []]
            #     for k in range(len(uni)):
            #         batch_sep_sorted[idx].append(torch.tensor([uni[k]]))
            #         batch_sep_sorted[1-idx].append(torch.from_numpy(batch_sep[1-idx][inds[k]:inds[k+1]]))
            #     tmp_bitstrings_rep = tmp_bitstrings_rep[sort_inds]

        common_indices = sorted(frozenset(bond_i) & frozenset(bond_j))

        idxi_j = []
        idxj_i = []
        for idx in common_indices:
            idxi_j.append(bond_i.index(idx))
            idxj_i.append(bond_j.index(idx))
        tensor_bonds[i] = [bond_i[m] for m in range(len(bond_i)) if m not in idxi_j] + [bond_j[n] for n in range(len(bond_j)) if n not in idxj_i]

        permute_dim_i = 1 if len(tensor_info[i][0]) else 0
        permute_dim_j = 1 if len(tensor_info[j][0]) else 0
        idxi_j = [ind + 1 for ind in idxi_j] if len(tensor_info[i][0]) and batch_loc == 0 else idxi_j
        idxj_i = [ind + 1 for ind in idxj_i] if len(tensor_info[j][0]) and batch_loc == 0 else idxj_i
        if permute_dim_j == 1 and batch_loc == 0:
            # print(permute_dim_i, permute_dim_j, len(bond_i), len(bond_j), len(common_indices))
            permute_seq = list(range(permute_dim_i + permute_dim_j + len(bond_i) + len(bond_j) - 2 * len(common_indices)))
            si = permute_seq.pop(permute_dim_i + len(bond_i) - len(common_indices))
            permute_seq.insert(permute_dim_i, si)
            rshape = (-1,) + (2,) * (len(bond_i) + len(bond_j) - 2 * len(common_indices))
            next_tensor_shape = (len(tmp_bitstrings_rep),) + (2,) * (len(bond_i) + len(bond_j) - 2 * len(common_indices))
            contraction_scheme.append((edge, 'tensordot', idxi_j, idxj_i, batch_sep_sorted, permute_seq, rshape, next_tensor_shape))
        elif permute_dim_i == 1 and batch_loc == -1:
            # print(permute_dim_i, permute_dim_j, len(bond_i), len(bond_j), len(common_indices))
            permute_seq = list(range(permute_dim_i + permute_dim_j + len(bond_i) + len(bond_j) - 2 * len(common_indices)))
            si = permute_seq.pop(permute_dim_i + len(bond_i) - len(common_indices)-1)
            if permute_dim_j:
                permute_seq.insert(-1, si)
            else:
                permute_seq.append(si)
            rshape = (2,) * (len(bond_i) + len(bond_j) - 2 * len(common_indices)) + (-1,)
            next_tensor_shape = (2,) * (len(bond_i) + len(bond_j) - 2 * len(common_indices)) + (len(tmp_bitstrings_rep),)
            contraction_scheme.append((edge, 'tensordot', idxi_j, idxj_i, batch_sep_sorted, permute_seq, rshape, next_tensor_shape))
        else:
            contraction_scheme.append((edge, 'tensordot', idxi_j, idxj_i, batch_sep_sorted))
        
        tensor_info[i] = (tmp_final_qubit_ids, tmp_bitstrings_rep)
    
        if edge == order[-1]:
            tmp_bitstrings = [np.binary_repr(n, len(final_qubits)) for n in tensor_info[i][1]]

    return contraction_scheme, tensor_bonds[i], tmp_bitstrings


def tensor_contraction_sparsestate(tensors, contraction_scheme):
    '''
    contraction the tensor network according to contraction scheme

    :param tensors: numerical tensors of the tensor network
    :param contraction_scheme: list of contraction step, defintion of entries in each step:
                               step[0]: locations of tensors to be contracted
                               step[1]: set to be 'tensordot' here, maybe more operations in the future
                               step[2] and step[3], indices arguments of tensordot
                               step[4]: batch dimension of the contraction
                               step[5]: optional, if the second tensor has batch dimension, then here is the permute sequence
                               step[6]: optional, if the second tensor has batch dimension, then here is the reshape sequence

    :return tensors[i]: the final resulting amplitudes
    '''
    
    for step in contraction_scheme:
        i, j = step[0]
        assert step[1] == 'tensordot'
        batch_i, batch_j = step[4]
        if len(batch_i) > 1:
            tensors[i] = [tensors[i]]
            for k in range(len(batch_i)-1, -1, -1):
                if k != 0:
                    try:
                        tensors[i].insert(
                            1, 
                            torch.tensordot(
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]], 
                                (step[2], step[3])
                            ).permute(step[5]).reshape(step[6])
                        )
                    except:
                        print(step[0], tensors[i][0][batch_i[k]].shape, tensors[j][batch_j[k]].shape, step[2:])
                        print_exc()
                        sys.exit(1)
                else:
                    # tensors[i][0] = tensors[i][0][batch_i[k]]
                    try:
                        tensors[i][0] = torch.tensordot(
                            tensors[i][0][batch_i[k]], 
                            tensors[j][batch_j[k]], 
                            (step[2], step[3])
                        ).permute(step[5]).reshape(step[6])
                    except:
                        print(step[0], len(batch_i), tensors[i][0].shape, tensors[j].shape, step[2:])
                        for k in range(len(tensors)):
                            if type(tensors[k]) is not list:
                                print(k, np.log2(np.prod(tensors[k].shape)))
                        print_exc()
                        sys.exit(1)
            tensors[j] = []
            try:
                tensors[i] = torch.cat(tensors[i], dim=0)
            except:
                print(step, len(tensors[i]))
                print_exc()
                sys.exit(1)
        elif len(step) > 5:
            try:
                tensors[i] = torch.tensordot(
                    tensors[i], 
                    tensors[j], 
                    (step[2], step[3])
                ).permute(step[5]).reshape(step[6])
            except:
                print(step[0], len(batch_i), tensors[i].shape, tensors[j].shape, step[2:])
                print_exc()
                sys.exit(1)
            tensors[j] = []
        else:
            try:
                # torch.cuda.empty_cache()
                tensors[i] = torch.tensordot(tensors[i], tensors[j], (step[2], step[3]))
            except:
                print(step[0], len(batch_i), tensors[i].shape, tensors[j].shape, step[2], step[3])
                print_exc()
                sys.exit(1)
            tensors[j] = []

    return tensors[i]


def contraction_scheme_sparse_einsum(ctree:ContractionTree, bitstrings=None, sc_target=31):
    """
    Compile a contraction scheme according to the contraction tree in a depth-first search way
    """
    order = ctree.tree_order_dfs()
    tensor_bonds = ctree.tn.tensor_bonds
    contraction_scheme = []
    final_qubits = ctree.tn.final_qubits
    # print(f'final qubits: {final_qubits}')
    if type(final_qubits) == frozenset or type(final_qubits) == set:
        final_qubits = sorted(list(final_qubits))
    tensor_bitstrings = [[0, 1] for _ in range(len(final_qubits))]
    tensor_info = [([], [-1]) if k not in final_qubits else ([final_qubits.index(k)], tensor_bitstrings[final_qubits.index(k)]) for k in range(len(tensor_bonds))]
    
    for edge in order:
        i, j = edge
        bond_i, bond_j = tensor_bonds[i], tensor_bonds[j]

        common_indices = sorted(frozenset(bond_i) & frozenset(bond_j))
        uncontract_indices = []
        for ind in common_indices:
            for x in range(len(tensor_bonds)):
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
            batch_sep_sorted = [[torch.tensor([0])], [torch.tensor([0])]]
            tmp_bitstrings_rep = [-1]
        elif len(tensor_info[i][0]) > 0 and len(tensor_info[j][0]) == 0:
            batch_sep_sorted = [[torch.tensor([k for k in range(len(tensor_info[i][1]))])], [torch.tensor([0])]]
            tmp_bitstrings_rep = tensor_info[i][1]
        elif len(tensor_info[j][0]) > 0 and len(tensor_info[i][0]) == 0:
            batch_sep_sorted = [[torch.tensor([0])], [torch.tensor([k for k in range(len(tensor_info[j][1]))])]]
            tmp_bitstrings_rep = tensor_info[j][1]
        else:
            loc_i, loc_j = [tmp_final_qubit_ids.index(item) for item in tensor_info[i][0]], [tmp_final_qubit_ids.index(item) for item in tensor_info[j][0]]
            idx = int(len(tensor_info[i][1]) > len(tensor_info[j][1]))
            tmp_bitstrings = np.unique(index_select(bitstrings, tmp_final_qubit_ids))
            if len(tmp_bitstrings) == 2 ** len(tmp_final_qubit_ids) or len(tmp_final_qubit_ids) + len(tensor_bonds[i]) <= sc_target:
                tmp_bitstrings_rep = [
                    int(combine_bitstring(np.binary_repr(x, len(tensor_info[i][0])), np.binary_repr(y, len(tensor_info[j][0])), loc_i, loc_j), 2) for x in tensor_info[i][1] for y in tensor_info[j][1] 
                ]
                if len(tmp_bitstrings) != len(tmp_bitstrings_rep):
                    remain_ind = sorted([tmp_bitstrings_rep.index(int(string, 2)) for string in tmp_bitstrings])
                    tmp_bitstrings_rep = [tmp_bitstrings_rep[ind] for ind in remain_ind]
                    batch_sep_sorted = [[torch.tensor(remain_ind)], []]
                else:
                    batch_sep_sorted = [
                        [], #[torch.tensor([item]) for item in range(len(tensor_info[i][1]))], 
                        [] #[torch.arange(len(tensor_info[j][1]))] * len(tensor_info[i][1])
                    ]
            else:
                if idx == 0:
                    dict_i = {np.binary_repr(x, len(loc_i)):tensor_info[i][1].index(x) for x in tensor_info[i][1]}
                    dict_j = {np.binary_repr(x, len(loc_j)):tensor_info[j][1].index(x) for x in tensor_info[j][1]}
                    tmp_bitstrings_sorted = sort_bitstrings_wrt_loc(tmp_bitstrings, loc_i, loc_j, dict_i, dict_j)
                    bitstrings_partial_i, bitstrings_partial_j = index_select(tmp_bitstrings_sorted, loc_i), index_select(tmp_bitstrings_sorted, loc_j)
                    tmp_bitstrings_rep = [int(bitstring, 2) for bitstring in tmp_bitstrings_sorted]
                    batch_sep_sorted = [[], []]
                    for k in range(len(tmp_bitstrings_sorted)):
                        tmp_bi_rep = dict_i[bitstrings_partial_i[k]]
                        if tmp_bi_rep not in batch_sep_sorted[idx]:
                            batch_sep_sorted[idx].append(tmp_bi_rep)
                            batch_sep_sorted[1-idx].append([])
                        batch_sep_sorted[1-idx][-1].append(dict_j[bitstrings_partial_j[k]])
                    if np.log2(len(tmp_bitstrings_rep)) + len(bond_i) <= sc_target and np.log2(len(tmp_bitstrings_rep)) + len(bond_j) <= sc_target:
                        batch_sep_sorted = [
                            [torch.cat([torch.tensor([batch_sep_sorted[0][i]] * len(batch_sep_sorted[1][i])) for i in range(len(batch_sep_sorted[0]))])],
                            [torch.cat([torch.tensor(batch_sep_sorted[1][i]) for i in range(len(batch_sep_sorted[1]))])]
                        ]
                        cat_batch_flag = True
                    else:
                        batch_sep_sorted = [[torch.tensor([i]) for i in batch_sep_sorted[0]], [torch.tensor(i) for i in batch_sep_sorted[1]]]
            # else:
            #     if len(tmp_bitstrings) == 2 ** len(tmp_final_qubit_ids):
            #         batch_sep_sorted = [
            #             [], #[torch.arange(len(tensor_info[i][1]))] * len(tensor_info[j][1]),
            #             [] # [torch.tensor([item]) for item in range(len(tensor_info[j][1]))], 
            #         ]
            #         tmp_bitstrings_rep = [
            #             int(combine_bitstring(np.binary_repr(x, len(tensor_info[i][0])), np.binary_repr(y, len(tensor_info[j][0])), loc_i, loc_j), 2) for x in tensor_info[i][1] for y in tensor_info[j][1] 
            #         ]
            #         # tmp_bitstrings_rep = [
            #         #     int(combine_bitstring(np.binary_repr(x, len(tensor_info[i][0])), np.binary_repr(y, len(tensor_info[j][0])), loc_i, loc_j), 2) for y in tensor_info[j][1] for x in tensor_info[i][1]
            #         # ]
                else:
                    dict_i = {np.binary_repr(x, len(loc_i)):tensor_info[i][1].index(x) for x in tensor_info[i][1]}
                    dict_j = {np.binary_repr(x, len(loc_j)):tensor_info[j][1].index(x) for x in tensor_info[j][1]}
                    tmp_bitstrings_sorted = sort_bitstrings_wrt_loc(tmp_bitstrings, loc_j, loc_i, dict_j, dict_i)
                    bitstrings_partial_i, bitstrings_partial_j = index_select(tmp_bitstrings_sorted, loc_i), index_select(tmp_bitstrings_sorted, loc_j)
                    tmp_bitstrings_rep = [int(bitstring, 2) for bitstring in tmp_bitstrings_sorted]
                    batch_sep_sorted = [[], []]
                    for k in range(len(tmp_bitstrings_sorted)):
                        tmp_bj_rep = dict_j[bitstrings_partial_j[k]]
                        if tmp_bj_rep not in batch_sep_sorted[idx]:
                            batch_sep_sorted[idx].append(tmp_bj_rep)
                            batch_sep_sorted[1-idx].append([])
                        batch_sep_sorted[1-idx][-1].append(dict_i[bitstrings_partial_i[k]])
                    if np.log2(len(tmp_bitstrings_rep)) + len(bond_i) <= sc_target and np.log2(len(tmp_bitstrings_rep)) + len(bond_j) <= sc_target:
                        batch_sep_sorted = [
                            [torch.cat([torch.tensor(batch_sep_sorted[0][i]) for i in range(len(batch_sep_sorted[0]))])],
                            [torch.cat([torch.tensor([batch_sep_sorted[1][i]] * len(batch_sep_sorted[0][i])) for i in range(len(batch_sep_sorted[1]))])] 
                        ]
                        cat_batch_flag = True
                    else:
                        batch_sep_sorted = [[torch.tensor(i) for i in batch_sep_sorted[0]], [torch.tensor([i]) for i in batch_sep_sorted[1]]]
            assert len(tmp_bitstrings_rep) == len(tmp_bitstrings)

            # lshape, rshape = (len(tensor_info[i][1]), ) + (len(bond_i),), (len(tensor_info[j][1]), ) + (len(bond_j),)
            # print(lshape, rshape)
            # batch_size = max(sum([len(t) for t in batch_sep_sorted[0]]), sum([len(t) for t in batch_sep_sorted[1]]))
            # batch_lshape, batch_rshape = batch_size * 2 ** len(bond_i), batch_size * 2 ** len(bond_j)
            # print(f'batch lshape: {np.log2(batch_lshape)} batch rshape {np.log2(batch_rshape)}')

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
                contraction_scheme.append((edge, einsum_eq, batch_sep_sorted, None, cat_batch_flag, next_tensor_shape))
            else:
                rshape = (-1,) + (2,) * len(tensor_bonds[i])
                contraction_scheme.append((edge, einsum_eq, batch_sep_sorted, rshape, cat_batch_flag, next_tensor_shape))
            # print(f'cat_batch_flag : {cat_batch_flag}')
        else:
            contraction_scheme.append((edge, einsum_eq, batch_sep_sorted))
        
        # print(ix_left, ix_right, iy, 
        #     [contraction_scheme[-1][i] if i != 2 else [[len(t) for t in batch_sep_sorted[0]], [len(t) for t in batch_sep_sorted[1]]] for i in range(len(contraction_scheme[-1]))]
        # )
        # print((tensor_info[i][0], len(tensor_info[i][1])), (tensor_info[j][0], len(tensor_info[j][1])))
        # print((tmp_final_qubit_ids, len(tmp_bitstrings_rep)))
        # print('-'*40)
        tensor_info[i] = (tmp_final_qubit_ids, tmp_bitstrings_rep)
    
        if edge == order[-1]:
            tmp_bitstrings = [np.binary_repr(n, len(final_qubits)) for n in tensor_info[i][1]]

    return contraction_scheme, tensor_bonds[i], tmp_bitstrings


def tensor_contraction_einsum(tensors, contraction_scheme, scientific_notation=False):
    if scientific_notation:
        factor = torch.tensor(0, dtype=tensors[0].dtype, device=tensors[0].device)

    for step in contraction_scheme:
        i, j = step[0]
        batch_i, batch_j = step[2]
        if len(batch_i) > 1 and not step[4]:
            tensors[i] = [tensors[i]]
            for k in range(len(batch_i)-1, -1, -1):
                if k != 0:
                    try:
                        tensors[i].insert(
                            1, 
                            torch.einsum(
                                step[1],
                                tensors[i][0][batch_i[k]], 
                                tensors[j][batch_j[k]], 
                            ).reshape(step[3])
                        )
                    except:
                        print(step, tensors[i][0][batch_i[k]].shape, tensors[j][batch_j[k]].shape)
                        print_exc()
                        sys.exit(1)
                else:
                    try:
                        tensors[i][0] = torch.einsum(
                            step[1],
                            tensors[i][0][batch_i[k]], 
                            tensors[j][batch_j[k]], 
                        ).reshape(step[3])
                    except:
                        print(step, len(batch_i), tensors[i][0].shape, tensors[j].shape)
                        for k in range(len(tensors)):
                            if type(tensors[k]) is not list:
                                print(k, np.log2(np.prod(tensors[k].shape)))
                        print_exc()
                        sys.exit(1)
            tensors[j] = []
            try:
                tensors[i] = torch.cat(tensors[i], dim=0)
            except:
                print(step, len(tensors[i]))
                print_exc()
                sys.exit(1)
        elif len(step) > 3 and step[4]:
            tensors[i] = tensors[i][batch_i[0]]
            tensors[j] = tensors[j][batch_j[0]]
            try:
                tensors[i] = torch.einsum(step[1], tensors[i], tensors[j])
            except:
                print(tensors[i].shape, tensors[j].shape)
                print(step)
                print_exc()
                sys.exit(1)
        elif len(step) > 3:
            try:
                tensors[i] = torch.einsum(
                    step[1],
                    tensors[i], 
                    tensors[j], 
                ).reshape(step[3])
                if len(batch_i) == 1:
                    tensors[i] = tensors[i][batch_i[0]]
            except:
                print(step, len(batch_i), tensors[i].shape, tensors[j].shape)
                print_exc()
                sys.exit(1)
            tensors[j] = []
        else:
            try:
                tensors[i] = torch.einsum(step[1], tensors[i], tensors[j])
            except:
                print(step, len(batch_i), tensors[i].shape, tensors[j].shape)
                print_exc()
                sys.exit(1)
            tensors[j] = []
        if scientific_notation:
            norm_factor = tensors[i].abs().max()
            tensors[i] /= norm_factor
            factor += torch.log10(norm_factor)

    if scientific_notation:
        return factor, tensors[i]
    else:
        return tensors[i]


def tensor_contraction_sparsestate_1(tensors, contraction_scheme):
    '''
    contraction the tensor network according to contraction scheme

    :param tensors: numerical tensors of the tensor network
    :param contraction_scheme: list of contraction step, defintion of entries in each step:
                               step[0]: locations of tensors to be contracted
                               step[1]: set to be 'tensordot' here, maybe more operations in the future
                               step[2] and step[3], indices arguments of tensordot
                               step[4]: batch dimension of the contraction
                               step[5]: optional, if the second tensor has batch dimension, then here is the permute sequence
                               step[6]: optional, if the second tensor has batch dimension, then here is the reshape sequence

    :return tensors[i]: the final resulting amplitudes
    '''
    device = tensors[0].device
    for step in contraction_scheme:
        i, j = step[0]
        assert step[1] == 'tensordot'
        batch_i, batch_j = step[4]
        if len(batch_i) > 1:
            tensors[i] = [tensors[i]]
            for k in range(len(batch_i)-1, -1, -1):
                if k != 0:
                    try:
                        tensors[i].insert(
                            1, 
                            torch.tensordot(
                                tensors[i][0].index_select(-1, batch_i[k].to(device)), 
                                tensors[j].index_select(-1, batch_j[k].to(device)), 
                                (step[2], step[3])
                            ).permute(step[5]).reshape(step[6])
                        )
                    except:
                        print(step[0], tensors[i][0].shape, tensors[j].shape, step[2:])
                        print_exc()
                        sys.exit(1)
                else:
                    # tensors[i][0] = tensors[i][0][batch_i[k]]
                    try:
                        tensors[i][0] = torch.tensordot(
                            tensors[i][0].index_select(-1, batch_i[k].to(device)), 
                            tensors[j].index_select(-1, batch_j[k].to(device)), 
                            (step[2], step[3])
                        ).permute(step[5]).reshape(step[6])
                    except:
                        print(step[0], len(batch_i), tensors[i][0].shape, tensors[j].shape, step[2:])
                        for k in range(len(tensors)):
                            if type(tensors[k]) is not list:
                                print(k, np.log2(np.prod(tensors[k].shape)))
                        print_exc()
                        sys.exit(1)
            tensors[j] = []
            try:
                tensors[i] = torch.cat(tensors[i], dim=-1)
            except:
                print(step, len(tensors[i]))
                print_exc()
                sys.exit(1)
        elif len(step) > 5:
            try:
                tensors[i] = torch.tensordot(
                    tensors[i], 
                    tensors[j], 
                    (step[2], step[3])
                ).permute(step[5]).reshape(step[6])
            except:
                print(step[0], len(batch_i), tensors[i].shape, tensors[j].shape, step[2:])
                print_exc()
                sys.exit(1)
            tensors[j] = []
        else:
            try:
                # torch.cuda.empty_cache()
                tensors[i] = torch.tensordot(tensors[i], tensors[j], (step[2], step[3]))
            except:
                print(step[0], len(batch_i), tensors[i].shape, tensors[j].shape, step[2], step[3])
                print_exc()
                sys.exit(1)
            tensors[j] = []

    return tensors[i]


def contraction_scheme_sparse_einsum_1(ctree:ContractionTree, bitstrings=None, sc_target=31):
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
    tensor_info = [([], np.array([-1])) if k not in final_qubits else ([final_qubits.index(k)], tensor_bitstrings[final_qubits.index(k)]) for k in range(len(tensor_bonds))]
    
    for edge in order:
        i, j = edge
        bond_i, bond_j = tensor_bonds[i], tensor_bonds[j]

        common_indices = sorted(frozenset(bond_i) & frozenset(bond_j))
        uncontract_indices = []
        for ind in common_indices:
            for x in range(len(tensor_bonds)):
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
                sort_inds = np.argsort(batch_seq_init[:, idx])
                batch_seq_init = batch_seq_init[sort_inds].T.reshape(2, -1)

                uni, inds = np.unique(batch_seq_init[idx], return_index=True)
                # if (i, j) == (0, 182):
                #     print(batch_seq_init)
                #     print(uni)
                #     print(inds)
                #     print(type(tensor_info[i][1]), type(tensor_info[j][1]))
                inds = list(inds) + [len(batch_seq_init[idx])]
                batch_seq = [[], []]
                for k in range(len(uni)):
                    batch_seq[idx].append([uni[k]])
                    seq_inds = batch_seq_init[1-idx][inds[k]:inds[k+1]].argsort()
                    batch_seq[1-idx].append(batch_seq_init[1-idx][inds[k]:inds[k+1]][seq_inds])
                    sort_inds[inds[k]:inds[k+1]] = sort_inds[inds[k]:inds[k+1]][seq_inds]
                tmp_bitstrings_rep = tmp_bitstrings_rep[sort_inds]
                # tmp_bitstrings_num = np.array([int(string, 2) for string in tmp_bitstrings])
                # tmp_bitstrings_rep = []
                # batch_seq = [[], []]
                # m, n = [i, j][idx], [i, j][1-idx]
                # loc_m, loc_n = [loc_i, loc_j][idx], [loc_i, loc_j][1-idx]
                # t_combine, t_determine = 0, 0
                # for x in range(len(tensor_info[m][1])):
                #     seq = []
                #     for y in range(len(tensor_info[n][1])):
                #         t0 = time.time()
                #         combine_num = int(
                #             combine_bitstring(
                #                 np.binary_repr(tensor_info[m][1][x], len(tensor_info[m][0])), np.binary_repr(tensor_info[n][1][y], len(tensor_info[n][0])), loc_m, loc_n
                #             ), 2
                #         )
                #         t1 = time.time()
                #         if combine_num in tmp_bitstrings_num:
                #             tmp_bitstrings_rep.append(combine_num) 
                #             seq.append(y)
                #         t2 = time.time()
                #         t_combine += t1 - t0
                #         t_determine += t2 - t1
                #     batch_seq[idx].append([x])
                #     batch_seq[1-idx].append(seq)
                # print(i, j, tensor_info[i][0], tensor_info[j][0], len(tensor_info[i][1]), len(tensor_info[j][1]), t_combine, t_determine)
                if np.log2(len(tmp_bitstrings_rep)) + max(len(bond_i), len(bond_j)) <= sc_target-1: # and np.log2(len(tmp_bitstrings_rep)) + len(bond_j) <= sc_target:
                    if idx == 0:
                        batch_seq = [
                            [torch.cat([torch.tensor(batch_seq[0][i] * len(batch_seq[1][i])) for i in range(len(batch_seq[0]))])],
                            [torch.cat([torch.tensor(batch_seq[1][i]) for i in range(len(batch_seq[1]))])]
                        ]
                    else:
                        batch_seq = [
                            [torch.cat([torch.tensor(seq) for seq in batch_seq[0]])],
                            [torch.cat([torch.tensor(batch_seq[1][i] * len(batch_seq[0][i])) for i in range(len(batch_seq[1]))])],
                        ]
                    cat_batch_flag = True
                else:
                    batch_seq = [[torch.tensor(i) for i in batch_seq[0]], [torch.tensor(i) for i in batch_seq[1]]]

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
                contraction_scheme.append((edge, einsum_eq, batch_seq, None, cat_batch_flag, next_tensor_shape))
            else:
                rshape = (-1,) + (2,) * len(tensor_bonds[i])
                contraction_scheme.append((edge, einsum_eq, batch_seq, rshape, cat_batch_flag, next_tensor_shape))
            # print(f'cat_batch_flag : {cat_batch_flag}')
        else:
            contraction_scheme.append((edge, einsum_eq, batch_seq))
        
        # print(ix_left, ix_right, iy, 
        #     [contraction_scheme[-1][i] if i != 2 else [[len(t) for t in batch_sep_sorted[0]], [len(t) for t in batch_sep_sorted[1]]] for i in range(len(contraction_scheme[-1]))]
        # )
        # print((tensor_info[i][0], len(tensor_info[i][1])), (tensor_info[j][0], len(tensor_info[j][1])))
        # print((tmp_final_qubit_ids, len(tmp_bitstrings_rep)))
        # print('-'*40)
        tensor_info[i] = (tmp_final_qubit_ids, tmp_bitstrings_rep)
    
        if edge == order[-1]:
            tmp_bitstrings = [np.binary_repr(n, len(final_qubits)) for n in tensor_info[i][1]]

    return contraction_scheme, tensor_bonds[i], tmp_bitstrings