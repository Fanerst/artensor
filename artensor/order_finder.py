from math import log10
import multiprocessing as mp
import numpy as np
import sys
from copy import deepcopy
from .greedy import GreedyOrderFinder
from .contraction_tree import ContractionTree, ContractionVertex, get_tc_sc_contraction
from .tensor_network import AbstractTensorNetwork


def score_fn(tc, sc, mc, sc_target=30.0, alpha=32.0, sc_weight=2.0):
    """
    Score function for finding order
    """
    return log10(alpha * 10 ** mc + 10 ** tc) + \
        sc_weight * log10(2) * max(0, sc - sc_target)


def simulate_annealing(
        tensor_network, sc_target=-1, trials=10, iters=50, betas=np.linspace(0.1, 10, 100), 
        slicing_repeat=4, start_seed=0, alpha=32.0
    ):
    greedy_order = GreedyOrderFinder(tensor_network)
    init_tree = [
        ContractionTree(
            deepcopy(tensor_network), 
            greedy_order('min_dim', start_seed + i)[0], 
            0
        )
        for i in range(trials)
    ]
    args = [
        (
            init_tree[i].copy(), sc_target, init_tree[i].tree_complexity(), 
            iters, betas, start_seed + i, slicing_repeat, alpha
        ) for i in range(trials)]
    p = mp.Pool(trials)
    results = p.starmap(sa_trial, args)
    p.close()
    results_slicing = [
        (result[0][1] + len(result[1].tn.slicing_bonds) * log10(2), result[1]) 
        for result in results
    ]
    best_result, best_tree = sorted(results_slicing, key=lambda info:info[0])[0]

    return best_tree.tree_to_order(), best_tree.tn.slicing_bonds


def sa_trial(
        tree, sc_target, init_result, iters, betas, seed, 
        slicing_repeat=4, alpha=32.0
    ):
    init_tc, init_sc, init_mc = init_result
    init_score = score_fn(init_tc, init_sc, init_mc, sc_target, alpha)
    best_result = [(init_score, init_tc, init_sc, init_mc), tree.copy()]
    sub_root = tree.tree[tree.all_tensors]
    rng = np.random.RandomState(seed)
    for beta in betas:
        for iter in range(iters):
            tree_update(
                sub_root, tree, 3, beta, init_sc, rng, 
                sc_target=sc_target, alpha=alpha
            )
            tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
            result = (
                score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target, alpha), 
                tc_tmp, sc_tmp, mc_tmp
            )
            if result[0] < best_result[0][0]:
                best_result = [result, tree.copy()]
    
    result = best_result[1].tree_complexity()
    optimized_sc = result[1]
    slicing_loop = 0
    while slicing_loop < slicing_repeat * (optimized_sc - sc_target) or best_result[0][2] > sc_target:
        tree = best_result[1]
        current_tc, current_sc, current_mc = tree.tree_complexity()
        if current_sc > sc_target:
            scores_slicing = []
            for bond in tree.select_slicing_bonds():
                tc_slicing, sc_slicing, mc_slicing = tree.slicing_tree_complexity_new(bond)
                scores_slicing.append(
                    (
                        bond, 
                        score_fn(tc_slicing, sc_slicing, mc_slicing, sc_target, alpha), 
                        tc_slicing, sc_slicing, mc_slicing
                    )
                )
            slicing_bond = sorted(scores_slicing, key=lambda info:info[1])[0][0]
            tree.slicing(slicing_bond)
        elif len(tree.tn.slicing_bonds) > 0:
            bond_add = rng.choice(list(tree.tn.slicing_bonds.keys()))
            tree.add_bond(bond_add)
        tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
        result = (
            score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target, alpha), 
            tc_tmp, sc_tmp, mc_tmp
        )
        best_result = (result, tree.copy())
        for beta in betas[-10:]:
            for iter in range(iters):
                sub_root = tree.tree[tree.all_tensors]
                tree_update(
                    sub_root, tree, 3, beta, sc_target, rng, 
                    sc_target=sc_target, alpha=alpha
                )
                tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
                result = (
                    score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target, alpha), 
                    tc_tmp, sc_tmp, mc_tmp
                )
                if result[0] < best_result[0][0]:
                    best_result = (result, tree.copy())
        slicing_loop += 1
    return best_result


def old_tree_stats(vertex, local_tree_leaves):
    """
    Given subroot and subtree, determine the order of it, only useful when the subtree size is 3
    """
    if vertex.left not in local_tree_leaves:
        branch = vertex.left
    elif vertex.right not in local_tree_leaves:
        branch = vertex.right
    else:
        print(
            vertex.left.contain_tensors, 
            vertex.right.contain_tensors, 
            [v.contain_tensors for v in local_tree_leaves]
        )
        raise ValueError('something wrong with the local tree')
    sc = max([leaf.sc for leaf in local_tree_leaves] + [vertex.sc, branch.sc])
    tc = log10(2**branch.tc + 2**vertex.tc)
    mc = log10(2**branch.mc + 2**vertex.mc)

    first_contract = sorted((local_tree_leaves.index(branch.left), local_tree_leaves.index(branch.right)))
    if first_contract == [0, 2]:
        return [(0,2), (0,1)], tc, sc, mc
    elif first_contract == [0, 1]:
        return [(0,1), (0,2)], tc, sc, mc
    else:
        assert first_contract == [1, 2]
        return [(1,2), (0,1)], tc, sc, mc


def tree_update(vertex, tree, size, beta, initial_sc, rng, sc_target=30.0, alpha=32.0):
    """
    Local update of the contraction tree in a recursive way.
    For each step, get the size 3 subtree of current contraction vertex and find out the possible
    alternative 2 other contraction orders to update and randomly choose one, the update probability
    is calculated according to their score ratio
    """
    local_tree_leaves, local_tree = tree.spanning_tree(vertex, size)
    if len(local_tree_leaves) > 2:
        # tc_tree, sc_tree, mc_tree = tree.tree_complexity(local_tree, vertex)
        order_old, tc_tree, sc_tree, mc_tree = old_tree_stats(vertex, local_tree_leaves)
        reference_score = score_fn(tc_tree, sc_tree, mc_tree, sc_target, alpha)
        order_pool = [o for o in [[(0,2),(0,1)], [(0,1),(0,2)], [(1,2),(0,1)]] if o != order_old]
        order_new = order_pool[rng.choice(2)]
        left_new, right_new = local_tree_leaves[order_new[0][0]], local_tree_leaves[order_new[0][1]]
        branch_new = ContractionVertex(
            left_new.contain_tensors | right_new.contain_tensors, 
            tree.tn, left_new, right_new
        )
        third_leaf = order_new[1][1] if order_new[0] != (1,2) else 0
        tc_vertex, sc_vertex, multiconfig_factor, contain_bonds, mc_vertex, contract_bonds, all_bonds = get_tc_sc_contraction(
            tree.tn, branch_new, local_tree_leaves[third_leaf]
        )
        # print(tc_vertex, sc_vertex, multiconfig_factor, contain_bonds, mc_vertex, contract_bonds, all_bonds)
        sc_new = max([leaf.sc for leaf in local_tree_leaves] + [sc_vertex, branch_new.sc])
        tc_new = log10(2**branch_new.tc + 2**tc_vertex)
        mc_new = log10(2**branch_new.mc + 2**mc_vertex)
        # tc_new, sc_new, mc_new = tree.tree_complexity_new_order(local_tree_leaves, order_new)
        score_new = score_fn(tc_new, sc_new, mc_new, sc_target, alpha)


        if score_new < reference_score or rng.rand() < np.exp(-beta * (score_new-reference_score)):
            for leaf in local_tree:
                assert leaf.contain_tensors in tree.tree.keys(), print(leaf.contain_tensors, len(tree.tree.keys()))
            # tree.apply_order(order_new, local_tree_leaves, local_tree, vertex)
            vertex.tc, vertex.sc, vertex.multiconfig_factor, vertex.contain_bonds, vertex.mc, vertex.contract_bonds, vertex.all_bonds = \
            tc_vertex, sc_vertex, multiconfig_factor, contain_bonds, mc_vertex, contract_bonds, all_bonds
            if vertex.left not in local_tree_leaves:
                tree.tree.pop(vertex.left.contain_tensors)
            elif vertex.right not in local_tree_leaves:
                tree.tree.pop(vertex.right.contain_tensors)
            vertex.left = branch_new
            vertex.right = local_tree_leaves[third_leaf]
            tree.tree[branch_new.contain_tensors] = branch_new

        for next_vertex in [vertex.left, vertex.right]:
            tree_update(next_vertex, tree, size, beta, initial_sc, rng, sc_target, alpha)


def find_order(
        tensor_bonds, bond_dims, final_qubits=[], seed=0, max_bitstrings=1, 
        **simulated_annnealing_args
    ):
    """
    Function wrapper for finding the contraction order of a given tensor network
    """
    tensor_network = AbstractTensorNetwork(
        deepcopy(tensor_bonds), 
        deepcopy(bond_dims),
        final_qubits,
        max_bitstrings)
    # greedy_order = GreedyOrderFinder(tensor_network)
    # order, tc, sc = greedy_order('min_dim', seed)
    # ctree = ContractionTree(deepcopy(tensor_network), order, seed)
    sys.setrecursionlimit(16385)
    order_slicing, slicing_bonds = simulate_annealing(
        deepcopy(tensor_network), **simulated_annnealing_args
    )

    for bond in slicing_bonds:
        tensor_network.slicing(bond)

    ctree_new = ContractionTree(tensor_network, order_slicing, seed)

    return order_slicing, slicing_bonds, ctree_new