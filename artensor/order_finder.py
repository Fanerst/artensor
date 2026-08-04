from math import log10
import multiprocessing as mp
import numpy as np
import sys
from .greedy import GreedyOrderFinder, MultiCostGreedyOrderFinder
from .contraction_tree import ContractionTree
from .tensor_network import AbstractTensorNetwork


def score_fn(tc, sc, mc, sc_target=30.0, alpha=32.0, sc_weight=2.0):
    """
    Score function for finding order
    """
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if alpha == 0:
        memory_time = tc
    else:
        memory_term = log10(alpha) + mc
        maximum = max(tc, memory_term)
        memory_time = maximum + log10(
            10 ** (tc - maximum) + 10 ** (memory_term - maximum)
        )
    return memory_time + sc_weight * log10(2) * max(0, sc - sc_target)


def _sliced_result_key(metrics, slicing_bonds, sc_target, alpha):
    """Rank a feasible sliced result by its complete weighted workload."""
    tc, sc, mc = metrics
    slice_overhead = sum(log10(dim) for dim in slicing_bonds.values())
    total_tc = tc + slice_overhead
    total_mc = mc + slice_overhead
    return score_fn(total_tc, sc, total_mc, sc_target, alpha), total_tc


def simulate_annealing(
        tensor_network, sc_target=-1, trials=10, iters=50, betas=np.linspace(0.1, 10, 100), 
        slicing_repeat=4, start_seed=0, alpha=32.0, workers=None,
        use_compiled=True, greedy_strategy="min_dim", greedy_repeats=16,
        greedy_max_time=0.0,
    ):
    if trials < 1:
        raise ValueError("trials must be positive")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    greedy_order = GreedyOrderFinder(tensor_network, use_compiled=use_compiled)
    if greedy_strategy == "multi_cost":
        multi_cost_greedy = MultiCostGreedyOrderFinder(
            tensor_network, use_compiled=use_compiled
        )

        def initial_order(seed):
            return multi_cost_greedy(
                seed=seed,
                max_repeats=greedy_repeats,
                max_time=greedy_max_time,
            ).order
    elif greedy_strategy in {"min_dim", "max_reduce"}:
        def initial_order(seed):
            return greedy_order(greedy_strategy, seed)[0]
    else:
        raise ValueError(
            "greedy_strategy must be 'min_dim', 'max_reduce', or 'multi_cost'"
        )
    native_core = None
    if use_compiled:
        try:
            from . import _order_core as native_core
        except ImportError:
            native_core = None

    # Dense/no-slicing searches can stay entirely in the native core.  Besides
    # avoiding process startup, this postpones construction of the public
    # ContractionTree until find_order prepares the final result.
    native_initial_orders = None
    if native_core is not None:
        native_results = []
        for trial in range(trials):
            trial_seed = start_seed + trial
            trial_order = initial_order(trial_seed)
            order, tc, sc, mc = native_core.anneal_order(
                tensor_network,
                trial_order,
                list(betas),
                int(iters),
                int(trial_seed),
                float(sc_target),
                float(alpha),
                2.0,
            )
            trial_score = score_fn(tc, sc, mc, sc_target, alpha)
            native_results.append((trial_score, sc, order))
        if all(sc <= sc_target for _, sc, _ in native_results):
            _, _, best_order = min(native_results, key=lambda result: result[0])
            return best_order, {}
        # Reuse the native-optimized topologies as the starting point for
        # slicing.  Previously every trial repeated the complete anneal after
        # this prepass whenever even one trial still exceeded ``sc_target``.
        native_initial_orders = [result[2] for result in native_results]

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
            tensor_network.copy(),
            (
                native_initial_orders[i]
                if native_initial_orders is not None
                else initial_order(start_seed + i)
            ),
            0
        )
        for i in range(trials)
    ]
    args = [
        (
            init_tree[i], sc_target, init_tree[i].tree_complexity(),
            iters, betas, start_seed + i, slicing_repeat, alpha, use_compiled,
            native_initial_orders is not None,
        ) for i in range(trials)]
    if workers is None:
        workers = trials
    workers = max(1, min(int(workers), trials))
    if workers == 1:
        results = [sa_trial(*arg) for arg in args]
    else:
        with mp.Pool(workers) as pool:
            results = pool.starmap(sa_trial, args)
    results_slicing = [
        (
            _sliced_result_key(
                result[0][1:], result[1].tn.slicing_bonds,
                sc_target, alpha,
            ),
            result[1],
        )
        for result in results
    ]
    _, best_tree = min(results_slicing, key=lambda info: info[0])

    return best_tree.tree_to_order(), best_tree.tn.slicing_bonds


def sa_trial(
        tree, sc_target, init_result, iters, betas, seed, 
        slicing_repeat=4, alpha=32.0, use_compiled=True,
        initial_optimized=False,
    ):
    init_tc, init_sc, init_mc = init_result
    init_score = score_fn(init_tc, init_sc, init_mc, sc_target, alpha)
    rng = np.random.RandomState(seed)
    native_core = None
    if use_compiled:
        try:
            from . import _order_core as native_core
        except ImportError:
            native_core = None

    if native_core is not None and not initial_optimized:
        order, tc_tmp, sc_tmp, mc_tmp = native_core.anneal_order(
            tree.tn,
            tree.tree_to_order(),
            list(betas),
            int(iters),
            int(seed),
            float(sc_target),
            float(alpha),
            2.0,
        )
        tree = ContractionTree(tree.tn, order)
        result = (
            score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target, alpha),
            tc_tmp, sc_tmp, mc_tmp,
        )
        current_result = (result, tree)
    else:
        current_result = ((init_score, init_tc, init_sc, init_mc), tree)
        if native_core is None:
            best_python_result = (
                current_result[0], current_result[1].copy()
            )
            sub_root = tree.tree[tree.all_tensors]
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
                    if result[0] < best_python_result[0][0]:
                        best_python_result = (result, tree.copy())
            current_result = best_python_result

    optimized_sc = current_result[0][2]
    best_feasible = None

    def retain_if_feasible(candidate):
        nonlocal best_feasible
        metrics, candidate_tree = candidate
        if metrics[2] > sc_target:
            return
        key = _sliced_result_key(
            metrics[1:], candidate_tree.tn.slicing_bonds,
            sc_target, alpha,
        )
        if best_feasible is None or key < best_feasible[0]:
            best_feasible = (key, (metrics, candidate_tree.copy()))

    retain_if_feasible(current_result)
    slicing_loop = 0
    while (
        slicing_loop < slicing_repeat * max(0.0, optimized_sc - sc_target)
        or current_result[0][2] > sc_target
    ):
        tree = current_result[1]
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
        state_result = (result, tree)
        if native_core is not None:
            order, tc_tmp, sc_tmp, mc_tmp = native_core.anneal_order(
                tree.tn,
                tree.tree_to_order(),
                list(betas[-10:]),
                int(iters),
                int(seed + slicing_loop + 1),
                float(sc_target),
                float(alpha),
                2.0,
            )
            tree = ContractionTree(tree.tn, order)
            result = (
                score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target, alpha),
                tc_tmp, sc_tmp, mc_tmp,
            )
            if result[0] < state_result[0][0]:
                state_result = (result, tree)
        else:
            best_python_result = (state_result[0], state_result[1].copy())
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
                    if result[0] < best_python_result[0][0]:
                        best_python_result = (result, tree.copy())
            state_result = best_python_result
        current_result = state_result
        retain_if_feasible(current_result)
        slicing_loop += 1
    if best_feasible is not None:
        return best_feasible[1]
    return current_result


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


def tree_update(vertex, tree, size, beta, initial_sc, rng, sc_target=30.0, alpha=32.0):
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
            tree_update(next_vertex, tree, size, beta, initial_sc, rng, sc_target, alpha)


def find_order(
        tensor_bonds, bond_dims, final_qubits=None, seed=0, max_bitstrings=1,
        **simulated_annnealing_args
    ):
    """
    Function wrapper for finding the contraction order of a given tensor network
    """
    tensor_network = AbstractTensorNetwork(
        tensor_bonds,
        bond_dims,
        final_qubits,
        max_bitstrings)
    # greedy_order = GreedyOrderFinder(tensor_network)
    # order, tc, sc = greedy_order('min_dim', seed)
    # ctree = ContractionTree(deepcopy(tensor_network), order, seed)
    sys.setrecursionlimit(16385)
    simulated_annnealing_args.setdefault("start_seed", seed)
    order_slicing, slicing_bonds = simulate_annealing(
        tensor_network, **simulated_annnealing_args
    )

    for bond in slicing_bonds:
        tensor_network.slicing(bond)

    ctree_new = ContractionTree(tensor_network, order_slicing, seed)

    return order_slicing, slicing_bonds, ctree_new
