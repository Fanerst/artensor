from math import log10
import multiprocessing as mp
import numpy as np
import sys
from copy import deepcopy
from .greedy import GreedyOrderFinder
from .contraction_tree import ContractionTree, candidate_local_tree_score, local_tree_score
from .tensor_network import AbstractTensorNetwork


def compress_bond_labels(tensor_bonds, bond_dims):
    bond_labels = sorted(bond_dims)
    bond_to_int = {bond: idx for idx, bond in enumerate(bond_labels)}
    int_to_bond = {idx: bond for bond, idx in bond_to_int.items()}
    compressed_tensor_bonds = {
        tensor_id: [bond_to_int[bond] for bond in bonds]
        for tensor_id, bonds in tensor_bonds.items()
    }
    compressed_bond_dims = {
        bond_to_int[bond]: dim
        for bond, dim in bond_dims.items()
    }
    return compressed_tensor_bonds, compressed_bond_dims, int_to_bond


def score_fn(tc, sc, mc, sc_target=30.0, alpha=32.0, sc_weight=2.0):
    """
    Score function for finding order
    """
    if tc >= mc:
        lead = tc
        tail = alpha * 10 ** (mc - tc)
    else:
        lead = mc
        tail = (10 ** (tc - mc)) / alpha
        return log10(alpha) + lead + log10(1 + tail) + \
            sc_weight * log10(2) * max(0, sc - sc_target)
    return lead + log10(1 + tail) + \
        sc_weight * log10(2) * max(0, sc - sc_target)


def snapshot_tree(tree):
    return tree.tree_to_order(), tuple(sorted(tree.tn.slicing_bonds))


def restore_tree(base_tensor_network, snapshot):
    order, slicing_bonds = snapshot
    tensor_network = deepcopy(base_tensor_network)
    for bond in slicing_bonds:
        tensor_network.slicing(bond)
    return ContractionTree(tensor_network, order, 0)


def reduce_slices(tree, sc_target, alpha):
    """
    Greedily add back sliced bonds while keeping space complexity within the target.
    """
    improved = True
    while improved and tree.tn.slicing_bonds:
        improved = False
        best_choice = None
        best_result = None
        for bond in list(tree.tn.slicing_bonds.keys()):
            tree.add_bond(bond)
            tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
            if sc_tmp <= sc_target:
                result = (
                    score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target, alpha),
                    tc_tmp,
                    sc_tmp,
                    mc_tmp,
                )
                if best_result is None or result[0] < best_result[0]:
                    best_choice = bond
                    best_result = result
            tree.slicing(bond)
        if best_choice is not None:
            tree.add_bond(best_choice)
            improved = True
    return tree


def select_ranked_slicing_bond(tree, current_sc, sc_target, alpha, candidate_limit=4):
    candidate_bonds = tree.ranked_slicing_bonds(current_sc, limit=candidate_limit)
    slicing_scores = []
    for bond in candidate_bonds:
        tc_slicing, sc_slicing, mc_slicing = tree.slicing_tree_complexity_new(bond)
        slicing_scores.append(
            (
                score_fn(tc_slicing, sc_slicing, mc_slicing, sc_target, alpha),
                bond,
                tc_slicing,
                sc_slicing,
                mc_slicing,
            )
        )
    return min(slicing_scores, key=lambda item: item[0])[1]


def simulate_annealing(
        tensor_network, sc_target=-1, trials=10, iters=50, betas=np.linspace(0.1, 10, 100), 
        slicing_repeat=4, start_seed=0, alpha=32.0, update_mode="optimized"
    ):
    greedy_order = GreedyOrderFinder(tensor_network)
    # order, tc, sc = greedy_order('min_dim', seed)
    # ctree = ContractionTree(deepcopy(tensor_network), order, seed)
    # init_result = tree.tree_complexity()
    # args = [
    #     (
    #         tree.copy(), sc_target, init_result, iters, betas, start_seed + i, 
    #         slicing_repeat, alpha
    #     ) for i in range(trials)]
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
            iters, betas, start_seed + i, slicing_repeat, alpha, update_mode
        ) for i in range(trials)]
    if update_mode == "optimized" and trials == 1:
        results = [sa_trial(*args[0])]
    else:
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
        slicing_repeat=4, alpha=32.0, update_mode="optimized"
    ):
    init_tc, init_sc, init_mc = init_result
    init_score = score_fn(init_tc, init_sc, init_mc, sc_target, alpha)
    base_tensor_network = deepcopy(tree.tn)
    if update_mode == "legacy":
        best_result = [(init_score, init_tc, init_sc, init_mc), tree.copy()]
    else:
        best_result = [(init_score, init_tc, init_sc, init_mc), snapshot_tree(tree)]
    rng = np.random.RandomState(seed)
    checkpoint_interval = 1 if update_mode == "legacy" else 10
    for beta in betas:
        for iter in range(iters):
            if update_mode == "legacy":
                tree_update_legacy(
                    tree.tree[tree.all_tensors], tree, 3, beta, init_sc, rng,
                    sc_target=sc_target, alpha=alpha
                )
            else:
                tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=sc_target, alpha=alpha)
            if (iter + 1) % checkpoint_interval != 0 and iter + 1 != iters:
                continue
            tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
            result = (
                score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target, alpha), 
                tc_tmp, sc_tmp, mc_tmp
            )
            if result[0] < best_result[0][0]:
                best_result = [result, tree.copy()] if update_mode == "legacy" else [result, snapshot_tree(tree)]
    
    best_tree = best_result[1] if update_mode == "legacy" else restore_tree(base_tensor_network, best_result[1])
    result = best_tree.tree_complexity()
    optimized_sc = result[1]
    if update_mode == "optimized":
        tree = restore_tree(base_tensor_network, best_result[1])
        current_tc, current_sc, current_mc = tree.tree_complexity()
        while current_sc > sc_target:
            slicing_bond = select_ranked_slicing_bond(tree, current_sc, sc_target, alpha)
            tree.slicing(slicing_bond)
            refine_betas = betas[-min(3, len(betas)):]
            refine_iters = max(1, min(2, iters))
            for beta in refine_betas:
                for _ in range(refine_iters):
                    tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=sc_target, alpha=alpha)
            current_tc, current_sc, current_mc = tree.tree_complexity()
        reduce_slices(tree, sc_target, alpha)
        current_tc, current_sc, current_mc = tree.tree_complexity()
        result = (
            score_fn(current_tc, current_sc, current_mc, sc_target, alpha),
            current_tc,
            current_sc,
            current_mc,
        )
        best_result = (result, snapshot_tree(tree))
        refine_betas = betas[-min(4, len(betas)):]
        refine_iters = max(1, min(2, iters))
        for beta in refine_betas:
            for iter in range(refine_iters):
                tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=sc_target, alpha=alpha)
                if iter + 1 != refine_iters:
                    continue
                tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
                result = (
                    score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target, alpha),
                    tc_tmp,
                    sc_tmp,
                    mc_tmp,
                )
                if result[0] < best_result[0][0]:
                    best_result = (result, snapshot_tree(tree))
    else:
        slicing_loop = 0
        slicing_ratio = slicing_repeat
        while slicing_loop < slicing_ratio * max(0, optimized_sc - sc_target) or best_result[0][2] > sc_target:
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
                slicing_bond = sorted(scores_slicing, key=lambda info: info[1])[0][0]
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
                    tree_update_legacy(
                        tree.tree[tree.all_tensors], tree, 3, beta, sc_target, rng,
                        sc_target=sc_target, alpha=alpha
                    )
                    if (iter + 1) % checkpoint_interval != 0 and iter + 1 != iters:
                        continue
                    tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
                    result = (
                        score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target, alpha), 
                        tc_tmp, sc_tmp, mc_tmp
                    )
                    if result[0] < best_result[0][0]:
                        best_result = (result, tree.copy())
            slicing_loop += 1
    if update_mode == "legacy":
        return best_result
    return best_result[0], restore_tree(base_tensor_network, best_result[1])


def determine_old_order(vertex, local_tree_leaves):
    """
    Given subroot and subtree, determine the order of it, only useful when the subtree size is 3
    """
    if vertex.left not in local_tree_leaves:
        branch = vertex.left
    elif vertex.right not in local_tree_leaves:
        branch = vertex.right
    else:
        print(vertex.left, vertex.right, local_tree_leaves)
        raise ValueError('something wrong with the local tree')
    first_contract = sorted((local_tree_leaves.index(branch.left), local_tree_leaves.index(branch.right)))
    if first_contract == [0, 2]:
        return [(0,2), (0,1)]
    elif first_contract == [0, 1]:
        return [(0,1), (0,2)]
    else:
        assert first_contract == [1, 2]
        return [(1,2), (0,1)]


def tree_update(vertex, tree, beta, rng, sc_target=30.0, alpha=32.0):
    """
    Apply a lightweight local tree rotation update recursively.
    """
    if vertex is None or not (vertex.left and vertex.right):
        return

    local_updates = tree.iter_local_updates(vertex)
    if local_updates:
        candidate_moves = []
        for side, branch, outer, first, second in local_updates:
            tc_tree, sc_tree, mc_tree = local_tree_score(branch, vertex, (first, second, outer))
            reference_score = score_fn(tc_tree, sc_tree, mc_tree, sc_target, alpha)
            if side == "left":
                candidates = (
                    candidate_local_tree_score(tree.tn, first, outer, second),
                    candidate_local_tree_score(tree.tn, second, outer, first),
                )
            else:
                candidates = (
                    candidate_local_tree_score(tree.tn, outer, second, first),
                    candidate_local_tree_score(tree.tn, outer, first, second),
                )
            for choice, (tc_new, sc_new, mc_new) in enumerate(candidates):
                score_new = score_fn(tc_new, sc_new, mc_new, sc_target, alpha)
                candidate_moves.append(
                    (score_new - reference_score, side, branch, outer, first, second, choice)
                )
        delta_score, side, branch, outer, first, second, choice = candidate_moves[rng.choice(len(candidate_moves))]
        if rng.rand() < np.exp(-beta * delta_score):
            tree.apply_local_update(vertex, side, branch, outer, first, second, choice)

    for next_vertex in (vertex.left, vertex.right):
        tree_update(next_vertex, tree, beta, rng, sc_target, alpha)


def tree_update_legacy(vertex, tree, size, beta, initial_sc, rng, sc_target=30.0, alpha=32.0):
    """
    Local update of the contraction tree in a recursive way.
    For each step, get the size 3 subtree of current contraction vertex and find out the possible
    alternative 2 other contraction orders to update and randomly choose one, the update probability
    is calculated according to their score ratio
    """
    local_tree_leaves, local_tree = tree.spanning_tree(vertex, size)
    if len(local_tree_leaves) > 2:
        tc_tree, sc_tree, mc_tree = tree.tree_complexity(local_tree, vertex)
        reference_score = score_fn(tc_tree, sc_tree, mc_tree, sc_target, alpha)
        order_old = determine_old_order(vertex, local_tree_leaves)
        order_pool = [[(0,2),(0,1)], [(0,1),(0,2)], [(1,2),(0,1)]]

        order_pool.remove(order_old)
        order_new = order_pool[rng.choice(2)]
        tc_new, sc_new, mc_new = tree.tree_complexity_new_order(local_tree_leaves, order_new)
        score_new = score_fn(tc_new, sc_new, mc_new, sc_target, alpha)


        if rng.rand() < np.exp(-beta * (score_new-reference_score)):
            tree.apply_order(order_new, local_tree_leaves, local_tree, vertex)

        for next_vertex in [vertex.left, vertex.right]:
            tree_update_legacy(next_vertex, tree, size, beta, initial_sc, rng, sc_target, alpha)


def find_order(
        tensor_bonds, bond_dims, final_qubits=[], seed=0, max_bitstrings=1, 
        **simulated_annnealing_args
    ):
    """
    Function wrapper for finding the contraction order of a given tensor network
    """
    compressed_tensor_bonds, compressed_bond_dims, int_to_bond = compress_bond_labels(
        deepcopy(tensor_bonds),
        deepcopy(bond_dims),
    )
    tensor_network = AbstractTensorNetwork(
        compressed_tensor_bonds,
        compressed_bond_dims,
        final_qubits,
        max_bitstrings)
    # greedy_order = GreedyOrderFinder(tensor_network)
    # order, tc, sc = greedy_order('min_dim', seed)
    # ctree = ContractionTree(deepcopy(tensor_network), order, seed)
    sys.setrecursionlimit(16385)
    order_slicing, slicing_bonds = simulate_annealing(
        deepcopy(tensor_network), **simulated_annnealing_args
    )

    for bond in list(slicing_bonds):
        tensor_network.slicing(bond)

    ctree_new = ContractionTree(tensor_network, order_slicing, seed)

    slicing_bonds = {int_to_bond[bond]: dim for bond, dim in slicing_bonds.items()}

    return order_slicing, slicing_bonds, ctree_new
