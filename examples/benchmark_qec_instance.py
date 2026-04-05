import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artensor.order_finder import (
    compress_bond_labels,
    optimize_peak_subtree,
    post_target_key,
    replace_slices,
    reduce_slices,
    restore_tree,
    score_fn,
    select_ranked_slicing_bond,
    should_start_slicing,
    snapshot_tree,
    tree_update,
)
from artensor.tensor_network import AbstractTensorNetwork
from artensor.greedy import GreedyOrderFinder
from artensor.contraction_tree import ContractionTree


def parse_qec_equation(path):
    data = Path(path).read_text(encoding="utf-8").strip()
    lhs, rhs = data.split("->")
    inputs = lhs.split(",")
    output = rhs[0]
    labels = sorted(set("".join(inputs)))
    tensor_bonds = {i: sorted(set(term)) for i, term in enumerate(inputs)}
    bond_dims = {label: 2 for label in labels}
    bond_dims[output] = 2 ** 8
    return tensor_bonds, bond_dims, output


def log_event(event, **fields):
    payload = {"event": event, **fields}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def reduce_slices_with_logging(tree, sc_target, alpha):
    improved = True
    round_idx = 0
    while improved and tree.tn.slicing_bonds:
        round_idx += 1
        round_start = time.perf_counter()
        improved = False
        best_choice = None
        best_result = None
        checked = 0
        for bond in list(tree.tn.slicing_bonds.keys()):
            checked += 1
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
            tc_now, sc_now, mc_now = tree.tree_complexity()
            log_event(
                "reduce_round_done",
                round=round_idx,
                elapsed_s=time.perf_counter() - round_start,
                checked=checked,
                restored_bond=int(best_choice),
                remaining_slices=len(tree.tn.slicing_bonds),
                tc=tc_now,
                sc=sc_now,
                mc=mc_now,
            )
        else:
            tc_now, sc_now, mc_now = tree.tree_complexity()
            log_event(
                "reduce_round_done",
                round=round_idx,
                elapsed_s=time.perf_counter() - round_start,
                checked=checked,
                restored_bond=None,
                remaining_slices=len(tree.tn.slicing_bonds),
                tc=tc_now,
                sc=sc_now,
                mc=mc_now,
            )
    return tree


def main():
    parser = argparse.ArgumentParser(description="Benchmark artensor on a QEC equation file.")
    parser.add_argument("equation", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sc-target", type=float, default=33.0)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--beta-start", type=float, default=0.1)
    parser.add_argument("--beta-stop", type=float, default=10.0)
    parser.add_argument("--beta-steps", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--greedy-alpha", type=float, default=0.0)
    parser.add_argument("--greedy-strategy", default="min_dim")
    parser.add_argument("--greedy-seed-start", type=int, default=None)
    parser.add_argument("--greedy-seed-count", type=int, default=1)
    parser.add_argument("--slicing-repeat", type=int, default=8)
    parser.add_argument("--disable-slicing", action="store_true")
    parser.add_argument("--min-sc-before-slicing", type=float, default=None)
    parser.add_argument("--max-slice-steps", type=int, default=None)
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--peak-rebuild-patience", type=int, default=3)
    parser.add_argument("--peak-rebuild-size", type=int, default=5)
    parser.add_argument("--peak-rebuild-vertex-limit", type=int, default=3)
    parser.add_argument("--peak-rebuild-min-sc-delta", type=float, default=8.0)
    parser.add_argument("--post-target-betas", type=int, default=0)
    parser.add_argument("--post-target-rounds", type=int, default=0)
    parser.add_argument("--slice-replace-rounds", type=int, default=0)
    parser.add_argument("--slice-candidate-limit", type=int, default=4)
    parser.add_argument("--replace-candidate-limit", type=int, default=4)
    args = parser.parse_args()

    run_start = time.perf_counter()
    log_event(
        "start",
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
        greedy_seed_start=args.greedy_seed_start,
        greedy_seed_count=args.greedy_seed_count,
        disable_slicing=args.disable_slicing,
        min_sc_before_slicing=args.min_sc_before_slicing,
        max_slice_steps=args.max_slice_steps,
        max_slices=args.max_slices,
        peak_rebuild_patience=args.peak_rebuild_patience,
        peak_rebuild_size=args.peak_rebuild_size,
        peak_rebuild_vertex_limit=args.peak_rebuild_vertex_limit,
        peak_rebuild_min_sc_delta=args.peak_rebuild_min_sc_delta,
        post_target_betas=args.post_target_betas,
        post_target_rounds=args.post_target_rounds,
        slice_replace_rounds=args.slice_replace_rounds,
        slice_candidate_limit=args.slice_candidate_limit,
        replace_candidate_limit=args.replace_candidate_limit,
    )

    stage_start = time.perf_counter()
    tensor_bonds, bond_dims, output = parse_qec_equation(args.equation)
    log_event(
        "parsed",
        elapsed_s=time.perf_counter() - stage_start,
        num_tensors=len(tensor_bonds),
        num_bonds=len(bond_dims),
        output_label=output,
    )

    stage_start = time.perf_counter()
    compressed_tensor_bonds, compressed_bond_dims, int_to_bond = compress_bond_labels(
        deepcopy(tensor_bonds),
        deepcopy(bond_dims),
    )
    bond_to_int = {bond: idx for idx, bond in int_to_bond.items()}
    tensor_network = AbstractTensorNetwork(
        compressed_tensor_bonds,
        compressed_bond_dims,
        open_bonds=[bond_to_int[output]],
    )
    log_event("network_built", elapsed_s=time.perf_counter() - stage_start)

    betas = np.linspace(args.beta_start, args.beta_stop, args.beta_steps)
    greedy_order = GreedyOrderFinder(tensor_network)

    stage_start = time.perf_counter()
    seed_start = args.seed if args.greedy_seed_start is None else args.greedy_seed_start
    greedy_candidates = []
    for offset in range(max(1, args.greedy_seed_count)):
        greedy_seed = seed_start + offset
        candidate_start = time.perf_counter()
        order, greedy_tc, greedy_sc = greedy_order(
            args.greedy_strategy,
            greedy_seed,
            alpha=args.greedy_alpha,
        )
        tree = ContractionTree(deepcopy(tensor_network), order, 0)
        tc, sc, mc = tree.tree_complexity()
        greedy_candidates.append(((sc, tc, mc, greedy_seed), order, tree, greedy_tc, greedy_sc, greedy_seed))
        log_event(
            "greedy_candidate_done",
            elapsed_s=time.perf_counter() - candidate_start,
            greedy_seed=greedy_seed,
            greedy_tc=greedy_tc,
            greedy_sc=greedy_sc,
            tree_tc=tc,
            tree_sc=sc,
            tree_mc=mc,
            order_len=len(order),
            greedy_strategy=args.greedy_strategy,
        )
    _, order, tree, greedy_tc, greedy_sc, selected_greedy_seed = min(
        greedy_candidates, key=lambda item: item[0]
    )
    tc, sc, mc = tree.tree_complexity()
    log_event(
        "greedy_done",
        elapsed_s=time.perf_counter() - stage_start,
        tc=greedy_tc,
        sc=greedy_sc,
        tree_tc=tc,
        tree_sc=sc,
        tree_mc=mc,
        order_len=len(order),
        greedy_strategy=args.greedy_strategy,
        selected_seed=selected_greedy_seed,
    )
    log_event(
        "tree_built",
        elapsed_s=0.0,
        tc=tc,
        sc=sc,
        mc=mc,
    )

    rng = np.random.RandomState(args.seed)
    base_tensor_network = deepcopy(tree.tn)
    best_result = [(score_fn(tc, sc, mc, args.sc_target, args.alpha), tc, sc, mc), snapshot_tree(tree)]

    plateau_rounds = 0
    last_rebuild_sc = sc
    for beta in betas:
        stage_start = time.perf_counter()
        for _ in range(args.iters):
            tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=args.sc_target, alpha=args.alpha)
        tc, sc, mc = tree.tree_complexity()
        result = (score_fn(tc, sc, mc, args.sc_target, args.alpha), tc, sc, mc)
        improved = result[0] < best_result[0][0]
        if improved:
            best_result = [result, snapshot_tree(tree)]
        if args.peak_rebuild_min_sc_delta is None:
            plateau_rounds = 0 if improved else plateau_rounds + 1
        else:
            if last_rebuild_sc - sc >= args.peak_rebuild_min_sc_delta:
                plateau_rounds = 0
                last_rebuild_sc = sc
            else:
                plateau_rounds += 1
        if args.peak_rebuild_patience is not None and plateau_rounds >= args.peak_rebuild_patience:
            rebuild_start = time.perf_counter()
            changed, rebuild_result = optimize_peak_subtree(
                tree,
                args.sc_target,
                args.alpha,
                subtree_size=args.peak_rebuild_size,
                vertex_limit=args.peak_rebuild_vertex_limit,
            )
            plateau_rounds = 0
            tc, sc, mc = tree.tree_complexity()
            last_rebuild_sc = sc
            result = (score_fn(tc, sc, mc, args.sc_target, args.alpha), tc, sc, mc)
            if result[0] < best_result[0][0]:
                best_result = [result, snapshot_tree(tree)]
            log_event(
                "peak_rebuild_done",
                elapsed_s=time.perf_counter() - rebuild_start,
                changed=changed,
                tc=tc,
                sc=sc,
                mc=mc,
                best_score=best_result[0][0],
            )
        log_event(
            "anneal_beta_done",
            beta=float(beta),
            elapsed_s=time.perf_counter() - stage_start,
            tc=tc,
            sc=sc,
            mc=mc,
            best_score=best_result[0][0],
        )

    stage_start = time.perf_counter()
    tree = restore_tree(base_tensor_network, best_result[1])
    tc, sc, mc = tree.tree_complexity()
    log_event(
        "restore_best",
        elapsed_s=time.perf_counter() - stage_start,
        tc=tc,
        sc=sc,
        mc=mc,
    )

    slicing_step = 0
    allow_slicing = should_start_slicing(
        sc,
        args.sc_target,
        min_sc_before_slicing=args.min_sc_before_slicing,
        disable_slicing=args.disable_slicing,
    )
    if not allow_slicing and sc > args.sc_target:
        log_event(
            "slicing_skipped",
            reason="disabled" if args.disable_slicing else "above_min_sc_before_slicing",
            tc=tc,
            sc=sc,
            mc=mc,
            min_sc_before_slicing=args.min_sc_before_slicing,
        )
    while allow_slicing and sc > args.sc_target:
        if args.max_slice_steps is not None and slicing_step >= args.max_slice_steps:
            log_event(
                "slice_limit_reached",
                limit_type="max_slice_steps",
                limit=args.max_slice_steps,
                tc=tc,
                sc=sc,
                mc=mc,
                num_slices=len(tree.tn.slicing_bonds),
            )
            break
        if args.max_slices is not None and len(tree.tn.slicing_bonds) >= args.max_slices:
            log_event(
                "slice_limit_reached",
                limit_type="max_slices",
                limit=args.max_slices,
                tc=tc,
                sc=sc,
                mc=mc,
                num_slices=len(tree.tn.slicing_bonds),
            )
            break
        stage_start = time.perf_counter()
        slicing_bond = select_ranked_slicing_bond(
            tree,
            sc,
            args.sc_target,
            args.alpha,
            candidate_limit=args.slice_candidate_limit,
        )
        tree.slicing(slicing_bond)
        refine_betas = betas[-min(3, len(betas)):]
        refine_iters = max(1, min(2, args.iters))
        for beta in refine_betas:
            for _ in range(refine_iters):
                tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=args.sc_target, alpha=args.alpha)
        tc, sc, mc = tree.tree_complexity()
        slicing_step += 1
        log_event(
            "slice_step_done",
            step=slicing_step,
            elapsed_s=time.perf_counter() - stage_start,
            bond=int(slicing_bond),
            tc=tc,
            sc=sc,
            mc=mc,
            num_slices=len(tree.tn.slicing_bonds),
        )

    stage_start = time.perf_counter()
    before_reduce = len(tree.tn.slicing_bonds)
    if before_reduce:
        reduce_slices_with_logging(tree, args.sc_target, args.alpha)
        if args.slice_replace_rounds > 0:
            for round_idx in range(args.slice_replace_rounds):
                changed = replace_slices(
                    tree,
                    args.sc_target,
                    args.alpha,
                    candidate_limit=args.replace_candidate_limit,
                )
                log_event(
                    "slice_replace_done",
                    round=round_idx + 1,
                    changed=changed,
                    remaining_slices=len(tree.tn.slicing_bonds),
                )
                if not changed:
                    break
                reduce_slices_with_logging(tree, args.sc_target, args.alpha)
        if args.post_target_betas > 0:
            refine_betas = betas[-min(args.post_target_betas, len(betas)):]
            refine_rounds = max(1, args.post_target_rounds)
            best_post_key = post_target_key(tree, args.sc_target, args.alpha)
            best_post_snapshot = snapshot_tree(tree)
            for round_idx in range(refine_rounds):
                round_start = time.perf_counter()
                for beta in refine_betas:
                    for _ in range(max(1, min(2, args.iters))):
                        tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=args.sc_target, alpha=args.alpha)
                reduce_slices_with_logging(tree, args.sc_target, args.alpha)
                if args.slice_replace_rounds > 0:
                    replace_slices(
                        tree,
                        args.sc_target,
                        args.alpha,
                        candidate_limit=args.replace_candidate_limit,
                    )
                    reduce_slices_with_logging(tree, args.sc_target, args.alpha)
                tc, sc, mc = tree.tree_complexity()
                candidate_key = post_target_key(tree, args.sc_target, args.alpha)
                if candidate_key < best_post_key:
                    best_post_key = candidate_key
                    best_post_snapshot = snapshot_tree(tree)
                log_event(
                    "post_target_round_done",
                    round=round_idx + 1,
                    elapsed_s=time.perf_counter() - round_start,
                    tc=tc,
                    sc=sc,
                    mc=mc,
                    num_slices=len(tree.tn.slicing_bonds),
                )
            tree = restore_tree(base_tensor_network, best_post_snapshot)
    tc, sc, mc = tree.tree_complexity()
    slicing_bonds = {int_to_bond[bond]: dim for bond, dim in tree.tn.slicing_bonds.items()}
    log_event(
        "reduce_slices_done",
        elapsed_s=time.perf_counter() - stage_start,
        before_reduce=before_reduce,
        after_reduce=len(slicing_bonds),
        tc=tc,
        sc=sc,
        mc=mc,
    )

    result = {
        "equation": str(args.equation),
        "output_label": output,
        "seed": args.seed,
        "time_s": time.perf_counter() - run_start,
        "order_len": len(order),
        "slices": len(slicing_bonds),
        "tc": tc,
        "sc": sc,
        "mc": mc,
        "greedy_alpha": args.greedy_alpha,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
