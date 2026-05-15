import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().with_name('benchmark_qec_instance.py')
spec = importlib.util.spec_from_file_location('benchmark_qec_instance_mod', MODULE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def beam_restore_slices_with_refinement(
    tree,
    base_tensor_network,
    sc_target,
    alpha,
    refine_betas,
    refine_iters,
    max_rounds,
    beam_width,
    expand_width,
):
    def snapshot_key(snapshot):
        order, slicing_bonds = snapshot
        return tuple(slicing_bonds)

    best_snapshot = base.snapshot_tree(tree)
    best_key = base.post_target_key(tree, sc_target, alpha)
    frontier = [(best_key, best_snapshot)]
    seen = {snapshot_key(best_snapshot): best_key}

    for round_idx in range(max_rounds):
        candidate_states = []
        round_improved = False
        for _, state_snapshot in frontier:
            state_tree = base.restore_tree(base_tensor_network, state_snapshot)
            current_key = base.post_target_key(state_tree, sc_target, alpha)
            bonds = list(state_tree.tn.slicing_bonds.keys())
            if not bonds:
                continue

            scored = []
            for bond in bonds:
                trial_tree = base.restore_tree(base_tensor_network, state_snapshot)
                trial_tree.add_bond(bond)
                for beta in refine_betas:
                    for _ in range(refine_iters):
                        base.tree_update(
                            trial_tree.tree[trial_tree.all_tensors],
                            trial_tree,
                            beta,
                            np.random.RandomState(0),
                            sc_target=sc_target,
                            alpha=alpha,
                        )
                base.reduce_slices(trial_tree, sc_target, alpha)
                tc, sc, mc = trial_tree.tree_complexity()
                if sc > sc_target:
                    continue
                candidate_key = base.post_target_key(trial_tree, sc_target, alpha)
                scored.append((candidate_key, bond, base.snapshot_tree(trial_tree), tc, sc, mc))
            scored.sort(key=lambda item: item[0])
            for candidate_key, bond, snapshot, tc, sc, mc in scored[:expand_width]:
                key = snapshot_key(snapshot)
                if key in seen and seen[key] <= candidate_key:
                    continue
                seen[key] = candidate_key
                candidate_states.append((candidate_key, snapshot, bond, tc, sc, mc))
                if candidate_key < current_key:
                    round_improved = True

        if not candidate_states:
            base.log_event(
                'beam_restore_round_done',
                round=round_idx + 1,
                improved=False,
                remaining_slices=len(base.restore_tree(base_tensor_network, best_snapshot).tn.slicing_bonds),
                tc=best_key[2],
                sc=sc_target,
                mc=best_key[3],
            )
            break

        candidate_states.sort(key=lambda item: item[0])
        frontier = [(item[0], item[1]) for item in candidate_states[:beam_width]]
        if frontier[0][0] < best_key:
            best_key, best_snapshot = frontier[0]
        best_tree = base.restore_tree(base_tensor_network, best_snapshot)
        tc, sc, mc = best_tree.tree_complexity()
        base.log_event(
            'beam_restore_round_done',
            round=round_idx + 1,
            improved=round_improved,
            remaining_slices=len(best_tree.tn.slicing_bonds),
            tc=tc,
            sc=sc,
            mc=mc,
        )
        if not round_improved:
            break

    return base.restore_tree(base_tensor_network, best_snapshot)


def main():
    parser = argparse.ArgumentParser(description='Run QEC benchmark with beam restore refinement.')
    parser.add_argument('equation', type=Path)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sc-target', type=float, default=33.0)
    parser.add_argument('--iters', type=int, default=0)
    parser.add_argument('--beta-start', type=float, default=0.1)
    parser.add_argument('--beta-stop', type=float, default=10.0)
    parser.add_argument('--beta-steps', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=64.0)
    parser.add_argument('--greedy-alpha', type=float, default=0.1)
    parser.add_argument('--greedy-strategy', default='paper_skewed')
    parser.add_argument('--post-target-betas', type=int, default=4)
    parser.add_argument('--post-target-rounds', type=int, default=6)
    parser.add_argument('--slice-replace-rounds', type=int, default=8)
    parser.add_argument('--slice-candidate-limit', type=int, default=8)
    parser.add_argument('--replace-candidate-limit', type=int, default=8)
    parser.add_argument('--restore-refine-rounds', type=int, default=6)
    parser.add_argument('--restore-refine-betas', type=int, default=4)
    parser.add_argument('--beam-rounds', type=int, default=4)
    parser.add_argument('--beam-width', type=int, default=4)
    parser.add_argument('--beam-expand-width', type=int, default=4)
    args = parser.parse_args()

    run_start = time.perf_counter()
    base.log_event(
        'start',
        equation=str(args.equation),
        seed=args.seed,
        sc_target=args.sc_target,
        iters=args.iters,
        beta_start=args.beta_start,
        beta_stop=args.beta_stop,
        beta_steps=args.beta_steps,
        alpha=args.alpha,
        greedy_alpha=args.greedy_alpha,
        greedy_strategy=args.greedy_strategy,
        post_target_betas=args.post_target_betas,
        post_target_rounds=args.post_target_rounds,
        slice_replace_rounds=args.slice_replace_rounds,
        slice_candidate_limit=args.slice_candidate_limit,
        replace_candidate_limit=args.replace_candidate_limit,
        restore_refine_rounds=args.restore_refine_rounds,
        restore_refine_betas=args.restore_refine_betas,
        beam_rounds=args.beam_rounds,
        beam_width=args.beam_width,
        beam_expand_width=args.beam_expand_width,
    )

    tensor_bonds, bond_dims, output = base.parse_qec_equation(args.equation)
    base.log_event('parsed', num_tensors=len(tensor_bonds), num_bonds=len(bond_dims), output_label=output)
    compressed_tensor_bonds, compressed_bond_dims, int_to_bond = base.compress_bond_labels(
        dict(tensor_bonds),
        dict(bond_dims),
    )
    bond_to_int = {bond: idx for idx, bond in int_to_bond.items()}
    tensor_network = base.AbstractTensorNetwork(
        compressed_tensor_bonds,
        compressed_bond_dims,
        open_bonds=[bond_to_int[output]],
    )
    base.log_event('network_built')

    betas = np.linspace(args.beta_start, args.beta_stop, args.beta_steps)
    greedy_order = base.GreedyOrderFinder(tensor_network)
    order, greedy_tc, greedy_sc = greedy_order(args.greedy_strategy, args.seed, alpha=args.greedy_alpha)
    tree = base.ContractionTree(base.deepcopy(tensor_network), order, 0)
    tc, sc, mc = tree.tree_complexity()
    base.log_event('greedy_done', tc=greedy_tc, sc=greedy_sc, tree_tc=tc, tree_sc=sc, tree_mc=mc, order_len=len(order), selected_seed=args.seed)

    rng = np.random.RandomState(args.seed)
    base_tensor_network = base.deepcopy(tree.tn)
    best_result = [(base.score_fn(tc, sc, mc, args.sc_target, args.alpha), tc, sc, mc), base.snapshot_tree(tree)]

    plateau_rounds = 0
    last_rebuild_sc = sc
    for beta in betas:
        for _ in range(args.iters):
            base.tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=args.sc_target, alpha=args.alpha)
        tc, sc, mc = tree.tree_complexity()
        result = (base.score_fn(tc, sc, mc, args.sc_target, args.alpha), tc, sc, mc)
        improved = result[0] < best_result[0][0]
        if improved:
            best_result = [result, base.snapshot_tree(tree)]
        if last_rebuild_sc - sc >= 8.0:
            plateau_rounds = 0
            last_rebuild_sc = sc
        else:
            plateau_rounds += 1
        if plateau_rounds >= 3:
            changed, _ = base.optimize_peak_subtree(tree, args.sc_target, args.alpha, subtree_size=5, vertex_limit=3)
            plateau_rounds = 0
            tc, sc, mc = tree.tree_complexity()
            last_rebuild_sc = sc
            result = (base.score_fn(tc, sc, mc, args.sc_target, args.alpha), tc, sc, mc)
            if result[0] < best_result[0][0]:
                best_result = [result, base.snapshot_tree(tree)]
            base.log_event('peak_rebuild_done', changed=changed, tc=tc, sc=sc, mc=mc)

    tree = base.restore_tree(base_tensor_network, best_result[1])
    tc, sc, mc = tree.tree_complexity()
    base.log_event('restore_best', tc=tc, sc=sc, mc=mc)

    slicing_step = 0
    while sc > args.sc_target:
        slicing_bond = base.select_ranked_slicing_bond(tree, sc, args.sc_target, args.alpha, candidate_limit=args.slice_candidate_limit)
        tree.slicing(slicing_bond)
        tc, sc, mc = tree.tree_complexity()
        slicing_step += 1
        base.log_event('slice_step_done', step=slicing_step, bond=int(slicing_bond), tc=tc, sc=sc, mc=mc, num_slices=len(tree.tn.slicing_bonds))

    if tree.tn.slicing_bonds:
        base.reduce_slices_with_logging(tree, args.sc_target, args.alpha)
        for round_idx in range(args.slice_replace_rounds):
            changed = base.replace_slices(tree, args.sc_target, args.alpha, candidate_limit=args.replace_candidate_limit)
            base.log_event('slice_replace_done', round=round_idx + 1, changed=changed, remaining_slices=len(tree.tn.slicing_bonds))
            if not changed:
                break
            base.reduce_slices_with_logging(tree, args.sc_target, args.alpha)
        if args.post_target_betas > 0:
            refine_betas = betas[-min(args.post_target_betas, len(betas)):]
            best_post_key = base.post_target_key(tree, args.sc_target, args.alpha)
            best_post_snapshot = base.snapshot_tree(tree)
            for round_idx in range(max(1, args.post_target_rounds)):
                for beta in refine_betas:
                    for _ in range(max(1, min(2, args.iters))):
                        base.tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=args.sc_target, alpha=args.alpha)
                base.reduce_slices_with_logging(tree, args.sc_target, args.alpha)
                if args.slice_replace_rounds > 0:
                    base.replace_slices(tree, args.sc_target, args.alpha, candidate_limit=args.replace_candidate_limit)
                    base.reduce_slices_with_logging(tree, args.sc_target, args.alpha)
                candidate_key = base.post_target_key(tree, args.sc_target, args.alpha)
                if candidate_key < best_post_key:
                    best_post_key = candidate_key
                    best_post_snapshot = base.snapshot_tree(tree)
                tc, sc, mc = tree.tree_complexity()
                base.log_event('post_target_round_done', round=round_idx + 1, tc=tc, sc=sc, mc=mc, num_slices=len(tree.tn.slicing_bonds))
            tree = base.restore_tree(base_tensor_network, best_post_snapshot)
        if args.restore_refine_rounds > 0 and tree.tn.slicing_bonds:
            refine_betas = betas[-min(max(1, args.restore_refine_betas), len(betas)):]
            refine_iters = max(1, min(2, args.iters))
            tree = beam_restore_slices_with_refinement(
                tree,
                base_tensor_network,
                args.sc_target,
                args.alpha,
                refine_betas,
                refine_iters,
                args.beam_rounds,
                args.beam_width,
                args.beam_expand_width,
            )

    tc, sc, mc = tree.tree_complexity()
    result = {
        'equation': str(args.equation),
        'output_label': output,
        'seed': args.seed,
        'time_s': time.perf_counter() - run_start,
        'order_len': len(tree.tree_to_order()),
        'slices': len(tree.tn.slicing_bonds),
        'tc': tc,
        'sc': sc,
        'mc': mc,
        'greedy_alpha': args.greedy_alpha,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
