import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().with_name('benchmark_qec_instance.py')
spec = importlib.util.spec_from_file_location('benchmark_qec_instance_mod', MODULE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


BEAM_WIDTH = 4
BEAM_EXPAND = 4


def beam_restore_slices_with_refinement(
    tree,
    base_tensor_network,
    sc_target,
    alpha,
    refine_betas,
    refine_iters,
    rng,
    rounds,
):
    def state_id(snapshot):
        order, slicing_bonds = snapshot
        return tuple(slicing_bonds)

    initial_snapshot = base.snapshot_tree(tree)
    initial_key = base.post_target_key(tree, sc_target, alpha)
    frontier = [(initial_key, initial_snapshot)]
    best_key = initial_key
    best_snapshot = initial_snapshot
    seen = {state_id(initial_snapshot): initial_key}

    for round_idx in range(rounds):
        candidates = []
        for _, snapshot in frontier:
            state_tree = base.restore_tree(base_tensor_network, snapshot)
            bonds = list(state_tree.tn.slicing_bonds.keys())
            local_candidates = []
            for bond in bonds:
                trial_tree = base.restore_tree(base_tensor_network, snapshot)
                trial_tree.add_bond(bond)
                local_rng = np.random.RandomState((int(bond) * 1009 + round_idx * 9173) % (2**32 - 1))
                for beta in refine_betas:
                    for _ in range(refine_iters):
                        base.tree_update(
                            trial_tree.tree[trial_tree.all_tensors],
                            trial_tree,
                            beta,
                            local_rng,
                            sc_target=sc_target,
                            alpha=alpha,
                        )
                base.reduce_slices(trial_tree, sc_target, alpha)
                tc, sc, mc = trial_tree.tree_complexity()
                if sc > sc_target:
                    continue
                candidate_key = base.post_target_key(trial_tree, sc_target, alpha)
                local_candidates.append((candidate_key, bond, base.snapshot_tree(trial_tree), tc, sc, mc))
            local_candidates.sort(key=lambda item: item[0])
            candidates.extend(local_candidates[:BEAM_EXPAND])

        if not candidates:
            tc, sc, mc = base.restore_tree(base_tensor_network, best_snapshot).tree_complexity()
            base.log_event(
                'restore_refine_done',
                round=round_idx + 1,
                restored_bond=None,
                remaining_slices=len(base.restore_tree(base_tensor_network, best_snapshot).tn.slicing_bonds),
                tc=tc,
                sc=sc,
                mc=mc,
            )
            break

        candidates.sort(key=lambda item: item[0])
        next_frontier = []
        improved = False
        logged_bond = None
        logged_tree = None
        for candidate_key, bond, snapshot, tc, sc, mc in candidates:
            sid = state_id(snapshot)
            prev = seen.get(sid)
            if prev is not None and prev <= candidate_key:
                continue
            seen[sid] = candidate_key
            next_frontier.append((candidate_key, snapshot))
            if candidate_key < best_key:
                best_key = candidate_key
                best_snapshot = snapshot
                improved = True
                logged_bond = bond
                logged_tree = (tc, sc, mc, len(snapshot[1]))
            if len(next_frontier) >= BEAM_WIDTH:
                break

        if not next_frontier:
            tc, sc, mc = base.restore_tree(base_tensor_network, best_snapshot).tree_complexity()
            base.log_event(
                'restore_refine_done',
                round=round_idx + 1,
                restored_bond=None,
                remaining_slices=len(base.restore_tree(base_tensor_network, best_snapshot).tn.slicing_bonds),
                tc=tc,
                sc=sc,
                mc=mc,
            )
            break

        frontier = next_frontier
        if logged_tree is None:
            best_tree = base.restore_tree(base_tensor_network, best_snapshot)
            tc, sc, mc = best_tree.tree_complexity()
            remaining = len(best_tree.tn.slicing_bonds)
        else:
            tc, sc, mc, remaining = logged_tree
        base.log_event(
            'restore_refine_done',
            round=round_idx + 1,
            restored_bond=None if logged_bond is None else int(logged_bond),
            remaining_slices=remaining,
            tc=tc,
            sc=sc,
            mc=mc,
        )
        if not improved:
            break

    return base.restore_tree(base_tensor_network, best_snapshot)


def main():
    global BEAM_WIDTH, BEAM_EXPAND
    parser = argparse.ArgumentParser(description='Run benchmark_qec_instance with beam restore refinement.')
    parser.add_argument('--beam-width', type=int, default=4)
    parser.add_argument('--beam-expand-width', type=int, default=4)
    args, remaining = parser.parse_known_args()
    BEAM_WIDTH = args.beam_width
    BEAM_EXPAND = args.beam_expand_width
    base.restore_slices_with_refinement = beam_restore_slices_with_refinement
    sys.argv = [str(MODULE_PATH)] + remaining
    base.main()


if __name__ == '__main__':
    main()
