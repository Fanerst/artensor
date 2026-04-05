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
    reduce_slices,
    restore_tree,
    score_fn,
    select_ranked_slicing_bond,
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
    tensor_bonds = {i: list(set(term)) for i, term in enumerate(inputs)}
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
    parser.add_argument("--slicing-repeat", type=int, default=8)
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
    order, greedy_tc, greedy_sc = greedy_order("min_dim", args.seed, alpha=args.greedy_alpha)
    log_event(
        "greedy_done",
        elapsed_s=time.perf_counter() - stage_start,
        tc=greedy_tc,
        sc=greedy_sc,
        order_len=len(order),
    )

    stage_start = time.perf_counter()
    tree = ContractionTree(deepcopy(tensor_network), order, 0)
    tc, sc, mc = tree.tree_complexity()
    log_event(
        "tree_built",
        elapsed_s=time.perf_counter() - stage_start,
        tc=tc,
        sc=sc,
        mc=mc,
    )

    rng = np.random.RandomState(args.seed)
    base_tensor_network = deepcopy(tree.tn)
    best_result = [(score_fn(tc, sc, mc, args.sc_target, args.alpha), tc, sc, mc), snapshot_tree(tree)]

    for beta in betas:
        stage_start = time.perf_counter()
        for _ in range(args.iters):
            tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=args.sc_target, alpha=args.alpha)
        tc, sc, mc = tree.tree_complexity()
        result = (score_fn(tc, sc, mc, args.sc_target, args.alpha), tc, sc, mc)
        if result[0] < best_result[0][0]:
            best_result = [result, snapshot_tree(tree)]
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
    while sc > args.sc_target:
        stage_start = time.perf_counter()
        slicing_bond = select_ranked_slicing_bond(tree, sc, args.sc_target, args.alpha)
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
    reduce_slices_with_logging(tree, args.sc_target, args.alpha)
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
