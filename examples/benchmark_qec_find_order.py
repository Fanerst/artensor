import argparse
import json
import sys
import time
from math import log10
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artensor.contraction_tree import ContractionTree
from artensor.greedy import GreedyOrderFinder
from artensor.order_finder import compress_bond_labels, score_fn, tree_update_legacy
from artensor.tensor_network import AbstractTensorNetwork


def parse_qec_equation(path: Path):
    data = path.read_text(encoding="utf-8").strip()
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


def run_legacy_trial(tensor_network, betas, args, trial_idx):
    trial_seed = args.seed + trial_idx
    trial_start = time.perf_counter()
    greedy_order = GreedyOrderFinder(tensor_network)

    stage_start = time.perf_counter()
    order, greedy_tc, greedy_sc = greedy_order(
        args.greedy_strategy,
        trial_seed,
        alpha=args.greedy_alpha,
    )
    log_event(
        "greedy_done",
        trial=trial_idx,
        seed=trial_seed,
        elapsed_s=time.perf_counter() - stage_start,
        tc=greedy_tc,
        sc=greedy_sc,
        order_len=len(order),
        greedy_strategy=args.greedy_strategy,
    )

    stage_start = time.perf_counter()
    tree = ContractionTree(tensor_network.clone(), order, 0)
    init_tc, init_sc, init_mc = tree.tree_complexity()
    best_result = (
        (score_fn(init_tc, init_sc, init_mc, args.sc_target, args.alpha), init_tc, init_sc, init_mc),
        tree.copy(),
    )
    log_event(
        "tree_built",
        trial=trial_idx,
        seed=trial_seed,
        elapsed_s=time.perf_counter() - stage_start,
        tc=init_tc,
        sc=init_sc,
        mc=init_mc,
    )

    rng = np.random.RandomState(trial_seed)
    for beta in betas:
        beta_start = time.perf_counter()
        for iter_idx in range(args.iters):
            tree_update_legacy(
                tree.tree[tree.all_tensors],
                tree,
                3,
                beta,
                init_sc,
                rng,
                sc_target=args.sc_target,
                alpha=args.alpha,
            )
            tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
            result = (
                score_fn(tc_tmp, sc_tmp, mc_tmp, args.sc_target, args.alpha),
                tc_tmp,
                sc_tmp,
                mc_tmp,
            )
            if result[0] < best_result[0][0]:
                best_result = (result, tree.copy())
        tc_now, sc_now, mc_now = tree.tree_complexity()
        log_event(
            "anneal_beta_done",
            trial=trial_idx,
            seed=trial_seed,
            beta=float(beta),
            elapsed_s=time.perf_counter() - beta_start,
            tc=tc_now,
            sc=sc_now,
            mc=mc_now,
            best_score=best_result[0][0],
        )

    tree = best_result[1]
    optimized_tc, optimized_sc, optimized_mc = tree.tree_complexity()
    log_event(
        "restore_best",
        trial=trial_idx,
        seed=trial_seed,
        elapsed_s=time.perf_counter() - trial_start,
        tc=optimized_tc,
        sc=optimized_sc,
        mc=optimized_mc,
        num_slices=len(tree.tn.slicing_bonds),
    )

    slicing_loop = 0
    slicing_ratio = args.slicing_repeat
    print(slicing_ratio * max(0, optimized_sc - args.sc_target), best_result[0][2])
    while slicing_loop < slicing_ratio * max(0, optimized_sc - args.sc_target) or best_result[0][2] > args.sc_target:
        loop_start = time.perf_counter()
        tree = best_result[1]
        current_tc, current_sc, current_mc = tree.tree_complexity()
        action = "noop"
        bond = None
        if current_sc > args.sc_target:
            scores_slicing = []
            for candidate_bond in tree.select_slicing_bonds():
                tc_slicing, sc_slicing, mc_slicing = tree.slicing_tree_complexity_new(candidate_bond)
                scores_slicing.append(
                    (
                        candidate_bond,
                        score_fn(tc_slicing, sc_slicing, mc_slicing, args.sc_target, args.alpha),
                        tc_slicing,
                        sc_slicing,
                        mc_slicing,
                    )
                )
            bond = sorted(scores_slicing, key=lambda info: info[1])[0][0]
            tree.slicing(bond)
            action = "slice"
        elif tree.tn.slicing_bonds:
            bond = rng.choice(list(tree.tn.slicing_bonds.keys()))
            tree.add_bond(bond)
            action = "restore"

        tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
        result = (
            score_fn(tc_tmp, sc_tmp, mc_tmp, args.sc_target, args.alpha),
            tc_tmp,
            sc_tmp,
            mc_tmp,
        )
        best_result = (result, tree.copy())

        refine_betas = betas[-min(10, len(betas)):]
        for beta in refine_betas:
            for iter_idx in range(args.iters):
                tree_update_legacy(
                    tree.tree[tree.all_tensors],
                    tree,
                    3,
                    beta,
                    args.sc_target,
                    rng,
                    sc_target=args.sc_target,
                    alpha=args.alpha,
                )
                tc_refine, sc_refine, mc_refine = tree.tree_complexity()
                result = (
                    score_fn(tc_refine, sc_refine, mc_refine, args.sc_target, args.alpha),
                    tc_refine,
                    sc_refine,
                    mc_refine,
                )
                if result[0] < best_result[0][0]:
                    best_result = (result, tree.copy())

        loop_tc, loop_sc, loop_mc = best_result[1].tree_complexity()
        log_event(
            "slicing_loop_done",
            trial=trial_idx,
            seed=trial_seed,
            loop=slicing_loop + 1,
            elapsed_s=time.perf_counter() - loop_start,
            action=action,
            bond=int(bond) if bond is not None else None,
            tc=loop_tc,
            sc=loop_sc,
            mc=loop_mc,
            num_slices=len(best_result[1].tn.slicing_bonds),
            best_score=best_result[0][0],
        )
        slicing_loop += 1

    final_tc, final_sc, final_mc = best_result[1].tree_complexity()
    log_event(
        "trial_done",
        trial=trial_idx,
        seed=trial_seed,
        elapsed_s=time.perf_counter() - trial_start,
        tc=final_tc,
        sc=final_sc,
        mc=final_mc,
        slices=len(best_result[1].tn.slicing_bonds),
        objective=best_result[0][0],
    )
    return best_result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run QEC order finding through the legacy find_order-style slicing loop "
            "while keeping the current greedy initializer, with stepwise logging."
        )
    )
    parser.add_argument("equation", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sc-target", type=float, default=33.0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--beta-start", type=float, default=0.1)
    parser.add_argument("--beta-stop", type=float, default=10.0)
    parser.add_argument("--beta-steps", type=int, default=20)
    parser.add_argument("--slicing-repeat", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--greedy-alpha", type=float, default=0.1)
    parser.add_argument("--greedy-strategy", default="paper_skewed")
    parser.add_argument("--result-out", type=Path, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    log_event(
        "start",
        equation=str(args.equation),
        seed=args.seed,
        sc_target=args.sc_target,
        trials=args.trials,
        iters=args.iters,
        beta_start=args.beta_start,
        beta_stop=args.beta_stop,
        beta_steps=args.beta_steps,
        slicing_repeat=args.slicing_repeat,
        alpha=args.alpha,
        greedy_alpha=args.greedy_alpha,
        greedy_strategy=args.greedy_strategy,
        update_mode="legacy",
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
        tensor_bonds,
        bond_dims,
    )
    bond_to_int = {bond: idx for idx, bond in int_to_bond.items()}
    tensor_network = AbstractTensorNetwork(
        compressed_tensor_bonds,
        compressed_bond_dims,
        final_qubits=[],
        max_bitstring=1,
        open_bonds=[bond_to_int[output]],
    )
    log_event("network_built", elapsed_s=time.perf_counter() - stage_start)

    betas = np.linspace(args.beta_start, args.beta_stop, args.beta_steps)
    trial_results = [
        run_legacy_trial(tensor_network, betas, args, trial_idx)
        for trial_idx in range(args.trials)
    ]
    best_trial_idx, best_result = min(
        enumerate(trial_results),
        key=lambda item: item[1][0][1] + len(item[1][1].tn.slicing_bonds) * log10(2),
    )
    best_tree = best_result[1]
    final_tc, final_sc, final_mc = best_tree.tree_complexity()
    order = best_tree.tree_to_order()
    slicing_bonds = {int_to_bond[bond]: dim for bond, dim in best_tree.tn.slicing_bonds.items()}

    log_event(
        "best_trial_selected",
        trial=best_trial_idx,
        seed=args.seed + best_trial_idx,
        tc=final_tc,
        sc=final_sc,
        mc=final_mc,
        slices=len(slicing_bonds),
        objective=best_result[0][0],
    )

    result = {
        "equation": str(args.equation),
        "output_label": output,
        "seed": args.seed,
        "time_s": time.perf_counter() - started,
        "order_len": len(order),
        "slices": len(slicing_bonds),
        "tc": final_tc,
        "sc": final_sc,
        "mc": final_mc,
        "sc_target": args.sc_target,
        "trials": args.trials,
        "iters": args.iters,
        "beta_start": args.beta_start,
        "beta_stop": args.beta_stop,
        "beta_steps": args.beta_steps,
        "slicing_repeat": args.slicing_repeat,
        "alpha": args.alpha,
        "greedy_alpha": args.greedy_alpha,
        "greedy_strategy": args.greedy_strategy,
        "update_mode": "legacy",
        "slicing_bonds": slicing_bonds,
    }
    if args.result_out is not None:
        args.result_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_event("result", **result)


if __name__ == "__main__":
    main()
