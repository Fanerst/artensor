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


def main():
    parser = argparse.ArgumentParser(description="Profile artensor stages on a QEC equation file.")
    parser.add_argument("equation", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sc-target", type=float, default=33.0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--beta-start", type=float, default=0.1)
    parser.add_argument("--beta-stop", type=float, default=10.0)
    parser.add_argument("--beta-steps", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--greedy-alpha", type=float, default=0.1)
    args = parser.parse_args()

    timings = {}

    t0 = time.perf_counter()
    tensor_bonds, bond_dims, output = parse_qec_equation(args.equation)
    timings["parse"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    compressed_tensor_bonds, compressed_bond_dims, int_to_bond = compress_bond_labels(
        deepcopy(tensor_bonds), deepcopy(bond_dims)
    )
    bond_to_int = {bond: idx for idx, bond in int_to_bond.items()}
    compressed_open_bonds = [bond_to_int[output]]
    tensor_network = AbstractTensorNetwork(
        compressed_tensor_bonds,
        compressed_bond_dims,
        open_bonds=compressed_open_bonds,
    )
    timings["network_build"] = time.perf_counter() - t0

    betas = np.linspace(args.beta_start, args.beta_stop, args.beta_steps)

    t0 = time.perf_counter()
    greedy_order = GreedyOrderFinder(tensor_network)
    order, greedy_tc, greedy_sc = greedy_order("min_dim", args.seed, alpha=args.greedy_alpha)
    timings["greedy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree = ContractionTree(deepcopy(tensor_network), order, 0)
    timings["tree_build"] = time.perf_counter() - t0

    rng = np.random.RandomState(args.seed)
    base_tensor_network = deepcopy(tree.tn)
    init_tc, init_sc, init_mc = tree.tree_complexity()
    best_result = [
        (score_fn(init_tc, init_sc, init_mc, args.sc_target, args.alpha), init_tc, init_sc, init_mc),
        snapshot_tree(tree),
    ]

    beta_stats = []
    t0 = time.perf_counter()
    for beta in betas:
        beta_start = time.perf_counter()
        for _ in range(args.iters):
            tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=args.sc_target, alpha=args.alpha)
        tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
        result = (score_fn(tc_tmp, sc_tmp, mc_tmp, args.sc_target, args.alpha), tc_tmp, sc_tmp, mc_tmp)
        if result[0] < best_result[0][0]:
            best_result = [result, snapshot_tree(tree)]
        beta_stats.append(
            {
                "beta": float(beta),
                "time_s": time.perf_counter() - beta_start,
                "tc": tc_tmp,
                "sc": sc_tmp,
                "mc": mc_tmp,
            }
        )
    timings["anneal_total"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree = restore_tree(base_tensor_network, best_result[1])
    timings["restore_best"] = time.perf_counter() - t0

    slicing_steps = []
    t0 = time.perf_counter()
    current_tc, current_sc, current_mc = tree.tree_complexity()
    while current_sc > args.sc_target:
        step_start = time.perf_counter()
        slicing_bond = select_ranked_slicing_bond(tree, current_sc, args.sc_target, args.alpha)
        tree.slicing(slicing_bond)
        refine_start = time.perf_counter()
        refine_betas = betas[-min(3, len(betas)):]
        refine_iters = max(1, min(2, args.iters))
        for beta in refine_betas:
            for _ in range(refine_iters):
                tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=args.sc_target, alpha=args.alpha)
        refine_time = time.perf_counter() - refine_start
        current_tc, current_sc, current_mc = tree.tree_complexity()
        slicing_steps.append(
            {
                "bond": int(slicing_bond),
                "step_time_s": time.perf_counter() - step_start,
                "refine_time_s": refine_time,
                "tc": current_tc,
                "sc": current_sc,
                "mc": current_mc,
                "num_slices": len(tree.tn.slicing_bonds),
            }
        )
    timings["slice_down_total"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    before_reduce = len(tree.tn.slicing_bonds)
    reduce_slices(tree, args.sc_target, args.alpha)
    timings["reduce_slices"] = time.perf_counter() - t0

    result = {
        "equation": str(args.equation),
        "seed": args.seed,
        "greedy_alpha": args.greedy_alpha,
        "alpha": args.alpha,
        "sc_target": args.sc_target,
        "iters": args.iters,
        "beta_steps": args.beta_steps,
        "timings": timings,
        "greedy_result": {"tc": greedy_tc, "sc": greedy_sc},
        "best_unsliced": {
            "tc": best_result[0][1],
            "sc": best_result[0][2],
            "mc": best_result[0][3],
        },
        "slicing": {
            "before_reduce": before_reduce,
            "after_reduce": len(tree.tn.slicing_bonds),
            "steps": len(slicing_steps),
            "last_step": slicing_steps[-1] if slicing_steps else None,
        },
        "beta_stats": beta_stats,
        "final": {
            "tc": tree.tree_complexity()[0],
            "sc": tree.tree_complexity()[1],
            "mc": tree.tree_complexity()[2],
        },
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
