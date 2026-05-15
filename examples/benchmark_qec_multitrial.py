import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artensor.contraction_tree import ContractionTree
from artensor.greedy import GreedyOrderFinder
from artensor.order_finder import (
    compress_bond_labels,
    optimize_peak_subtree,
    post_target_key,
    replace_slices,
    reduce_slices,
    restore_tree,
    score_fn,
    select_ranked_slicing_bond,
    snapshot_tree,
    tree_update,
)
from artensor.tensor_network import AbstractTensorNetwork


_WORKER_NETWORK_DATA = None
_WORKER_OUTPUT_BOND = None
_WORKER_CONFIG = None


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
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def parse_csv_list(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def prescreen_objective(item):
    return (
        item["tree_sc"],
        item["tree_tc"],
        item["tree_mc"],
        item["seed"],
    )


def select_prescreen_candidates(results, top_k, sc_target, alpha):
    if top_k >= len(results):
        return sorted(results, key=prescreen_objective)

    selected = []
    seen = set()

    def add_item(item):
        key = (item["seed"], item["strategy"], item["greedy_alpha"])
        if key in seen:
            return False
        seen.add(key)
        selected.append(item)
        return True

    groups = {}
    for item in results:
        group_key = (item["strategy"], item["greedy_alpha"])
        groups.setdefault(group_key, []).append(item)

    group_order = sorted(groups)
    per_group = max(1, top_k // max(1, len(group_order)))
    for group_key in group_order:
        ranked_group = sorted(
            groups[group_key],
            key=lambda item: (
                score_fn(item["tree_tc"], item["tree_sc"], item["tree_mc"], sc_target, alpha),
                item["tree_sc"],
                item["tree_tc"],
                item["seed"],
            ),
        )
        for item in ranked_group[:per_group]:
            if len(selected) >= top_k:
                return selected[:top_k]
            add_item(item)

    diversity_buckets = [
        sorted(results, key=lambda item: (item["tree_sc"], item["tree_tc"], item["seed"])),
        sorted(results, key=lambda item: (item["tree_tc"], item["tree_sc"], item["seed"])),
        sorted(results, key=lambda item: (item["tree_mc"], item["tree_sc"], item["seed"])),
        sorted(
            results,
            key=lambda item: (
                score_fn(item["tree_tc"], item["tree_sc"], item["tree_mc"], sc_target, alpha),
                item["tree_sc"],
                item["seed"],
            ),
        ),
    ]
    for bucket in diversity_buckets:
        for item in bucket:
            if len(selected) >= top_k:
                return selected[:top_k]
            add_item(item)
    return selected[:top_k]


def init_worker(network_data, output_bond, config):
    global _WORKER_NETWORK_DATA, _WORKER_OUTPUT_BOND, _WORKER_CONFIG
    _WORKER_NETWORK_DATA = network_data
    _WORKER_OUTPUT_BOND = output_bond
    _WORKER_CONFIG = config


def make_tensor_network():
    compressed_tensor_bonds, compressed_bond_dims = _WORKER_NETWORK_DATA
    return AbstractTensorNetwork(
        deepcopy(compressed_tensor_bonds),
        deepcopy(compressed_bond_dims),
        open_bonds=[_WORKER_OUTPUT_BOND],
    )


def restore_slices_with_refinement_plain(
    tree,
    base_tensor_network,
    sc_target,
    alpha,
    refine_betas,
    refine_iters,
    rng,
    rounds,
):
    for _ in range(rounds):
        current_key = post_target_key(tree, sc_target, alpha)
        best_candidate = None
        for bond in list(tree.tn.slicing_bonds.keys()):
            snapshot = snapshot_tree(tree)
            trial_tree = restore_tree(base_tensor_network, snapshot)
            trial_tree.add_bond(bond)
            for beta in refine_betas:
                for _ in range(refine_iters):
                    tree_update(
                        trial_tree.tree[trial_tree.all_tensors],
                        trial_tree,
                        beta,
                        rng,
                        sc_target=sc_target,
                        alpha=alpha,
                    )
            reduce_slices(trial_tree, sc_target, alpha)
            tc, sc, mc = trial_tree.tree_complexity()
            if sc > sc_target:
                continue
            candidate_key = post_target_key(trial_tree, sc_target, alpha)
            if candidate_key < current_key and (
                best_candidate is None or candidate_key < best_candidate[0]
            ):
                best_candidate = (candidate_key, snapshot_tree(trial_tree))
        if best_candidate is None:
            break
        tree = restore_tree(base_tensor_network, best_candidate[1])
    return tree


def prescreen_seed(candidate):
    seed, strategy, greedy_alpha = candidate
    cfg = _WORKER_CONFIG
    tensor_network = make_tensor_network()
    greedy_order = GreedyOrderFinder(tensor_network)
    t0 = time.perf_counter()
    order, greedy_tc, greedy_sc = greedy_order(
        strategy,
        seed,
        alpha=greedy_alpha,
    )
    tree = ContractionTree(tensor_network.clone(), order, 0)
    tc, sc, mc = tree.tree_complexity()
    return {
        "seed": seed,
        "strategy": strategy,
        "greedy_alpha": greedy_alpha,
        "elapsed_s": time.perf_counter() - t0,
        "greedy_tc": greedy_tc,
        "greedy_sc": greedy_sc,
        "tree_tc": tc,
        "tree_sc": sc,
        "tree_mc": mc,
        "order_len": len(order),
    }


def run_full_trial(candidate):
    seed, strategy, greedy_alpha = candidate
    cfg = _WORKER_CONFIG
    betas = np.linspace(cfg["beta_start"], cfg["beta_stop"], cfg["beta_steps"])
    rng = np.random.RandomState(seed)
    trial_start = time.perf_counter()

    tensor_network = make_tensor_network()
    greedy_order = GreedyOrderFinder(tensor_network)
    order, greedy_tc, greedy_sc = greedy_order(
        strategy,
        seed,
        alpha=greedy_alpha,
    )
    tree = ContractionTree(tensor_network.clone(), order, 0)
    tc, sc, mc = tree.tree_complexity()
    base_tensor_network = deepcopy(tree.tn)
    best_result = [(score_fn(tc, sc, mc, cfg["sc_target"], cfg["alpha"]), tc, sc, mc), snapshot_tree(tree)]

    plateau_rounds = 0
    last_rebuild_sc = sc
    for beta in betas:
        for _ in range(cfg["iters"]):
            tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=cfg["sc_target"], alpha=cfg["alpha"])
        tc, sc, mc = tree.tree_complexity()
        result = (score_fn(tc, sc, mc, cfg["sc_target"], cfg["alpha"]), tc, sc, mc)
        improved = result[0] < best_result[0][0]
        if improved:
            best_result = [result, snapshot_tree(tree)]
        if cfg["peak_rebuild_min_sc_delta"] is None:
            plateau_rounds = 0 if improved else plateau_rounds + 1
        else:
            if last_rebuild_sc - sc >= cfg["peak_rebuild_min_sc_delta"]:
                plateau_rounds = 0
                last_rebuild_sc = sc
            else:
                plateau_rounds += 1
        if cfg["peak_rebuild_patience"] is not None and plateau_rounds >= cfg["peak_rebuild_patience"]:
            changed, _ = optimize_peak_subtree(
                tree,
                cfg["sc_target"],
                cfg["alpha"],
                subtree_size=cfg["peak_rebuild_size"],
                vertex_limit=cfg["peak_rebuild_vertex_limit"],
            )
            plateau_rounds = 0
            last_rebuild_sc = tree.tree_complexity()[1]
            if changed:
                tc, sc, mc = tree.tree_complexity()
                result = (score_fn(tc, sc, mc, cfg["sc_target"], cfg["alpha"]), tc, sc, mc)
                if result[0] < best_result[0][0]:
                    best_result = [result, snapshot_tree(tree)]

    tree = restore_tree(base_tensor_network, best_result[1])
    tc, sc, mc = tree.tree_complexity()

    slicing_step = 0
    while sc > cfg["sc_target"]:
        slicing_bond = select_ranked_slicing_bond(
            tree,
            sc,
            cfg["sc_target"],
            cfg["alpha"],
            candidate_limit=cfg["slice_candidate_limit"],
        )
        tree.slicing(slicing_bond)
        refine_betas = betas[-min(3, len(betas)):]
        refine_iters = max(1, min(2, cfg["iters"]))
        for beta in refine_betas:
            for _ in range(refine_iters):
                tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=cfg["sc_target"], alpha=cfg["alpha"])
        tc, sc, mc = tree.tree_complexity()
        slicing_step += 1

    before_reduce = len(tree.tn.slicing_bonds)
    if before_reduce:
        reduce_slices(tree, cfg["sc_target"], cfg["alpha"])
        for _ in range(cfg["slice_replace_rounds"]):
            changed = replace_slices(
                tree,
                cfg["sc_target"],
                cfg["alpha"],
                candidate_limit=cfg["replace_candidate_limit"],
            )
            if not changed:
                break
            reduce_slices(tree, cfg["sc_target"], cfg["alpha"])
        if cfg["post_target_betas"] > 0:
            refine_betas = betas[-min(cfg["post_target_betas"], len(betas)):]
            best_post_key = post_target_key(tree, cfg["sc_target"], cfg["alpha"])
            best_post_snapshot = snapshot_tree(tree)
            for _ in range(max(1, cfg["post_target_rounds"])):
                for beta in refine_betas:
                    for _ in range(max(1, min(2, cfg["iters"]))):
                        tree_update(tree.tree[tree.all_tensors], tree, beta, rng, sc_target=cfg["sc_target"], alpha=cfg["alpha"])
                reduce_slices(tree, cfg["sc_target"], cfg["alpha"])
                if cfg["slice_replace_rounds"] > 0:
                    replace_slices(
                        tree,
                        cfg["sc_target"],
                        cfg["alpha"],
                        candidate_limit=cfg["replace_candidate_limit"],
                    )
                    reduce_slices(tree, cfg["sc_target"], cfg["alpha"])
                candidate_key = post_target_key(tree, cfg["sc_target"], cfg["alpha"])
                if candidate_key < best_post_key:
                    best_post_key = candidate_key
                    best_post_snapshot = snapshot_tree(tree)
            tree = restore_tree(base_tensor_network, best_post_snapshot)
        if cfg["restore_refine_rounds"] > 0 and tree.tn.slicing_bonds:
            refine_betas = betas[-min(max(1, cfg["restore_refine_betas"]), len(betas)):]
            refine_iters = max(1, min(2, cfg["iters"]))
            tree = restore_slices_with_refinement_plain(
                tree,
                base_tensor_network,
                cfg["sc_target"],
                cfg["alpha"],
                refine_betas,
                refine_iters,
                rng,
                cfg["restore_refine_rounds"],
            )

    tc, sc, mc = tree.tree_complexity()
    return {
        "seed": seed,
        "strategy": strategy,
        "greedy_alpha": greedy_alpha,
        "elapsed_s": time.perf_counter() - trial_start,
        "order_len": len(tree.tree_to_order()),
        "slices": len(tree.tn.slicing_bonds),
        "tc": tc,
        "sc": sc,
        "mc": mc,
        "objective": (len(tree.tn.slicing_bonds), score_fn(tc, sc, mc, cfg["sc_target"], cfg["alpha"]), tc, mc),
        "greedy_tc": greedy_tc,
        "greedy_sc": greedy_sc,
    }


def main():
    parser = argparse.ArgumentParser(description="Run batched multi-trial QEC optimization with the new design.")
    parser.add_argument("equation", type=Path)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--sc-target", type=float, default=33.0)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--beta-start", type=float, default=0.1)
    parser.add_argument("--beta-stop", type=float, default=10.0)
    parser.add_argument("--beta-steps", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--greedy-alpha", type=float, default=0.1)
    parser.add_argument("--greedy-strategy", default="paper_skewed")
    parser.add_argument("--greedy-alphas", default=None)
    parser.add_argument("--greedy-strategies", default=None)
    parser.add_argument("--peak-rebuild-patience", type=int, default=3)
    parser.add_argument("--peak-rebuild-size", type=int, default=5)
    parser.add_argument("--peak-rebuild-vertex-limit", type=int, default=3)
    parser.add_argument("--peak-rebuild-min-sc-delta", type=float, default=8.0)
    parser.add_argument("--post-target-betas", type=int, default=4)
    parser.add_argument("--post-target-rounds", type=int, default=6)
    parser.add_argument("--slice-replace-rounds", type=int, default=8)
    parser.add_argument("--slice-candidate-limit", type=int, default=8)
    parser.add_argument("--replace-candidate-limit", type=int, default=8)
    parser.add_argument("--restore-refine-rounds", type=int, default=6)
    parser.add_argument("--restore-refine-betas", type=int, default=4)
    parser.add_argument("--result-out", type=Path, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    log_event(
        "start",
        equation=str(args.equation),
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        top_k=args.top_k,
        max_workers=args.max_workers,
        sc_target=args.sc_target,
        iters=args.iters,
        greedy_alpha=args.greedy_alpha,
        greedy_strategy=args.greedy_strategy,
        greedy_alphas=args.greedy_alphas,
        greedy_strategies=args.greedy_strategies,
    )

    tensor_bonds, bond_dims, output = parse_qec_equation(args.equation)
    compressed_tensor_bonds, compressed_bond_dims, int_to_bond = compress_bond_labels(
        deepcopy(tensor_bonds),
        deepcopy(bond_dims),
    )
    bond_to_int = {bond: idx for idx, bond in int_to_bond.items()}
    network_data = (compressed_tensor_bonds, compressed_bond_dims)
    config = {
        "sc_target": args.sc_target,
        "iters": args.iters,
        "beta_start": args.beta_start,
        "beta_stop": args.beta_stop,
        "beta_steps": args.beta_steps,
        "alpha": args.alpha,
        "greedy_alpha": args.greedy_alpha,
        "greedy_strategy": args.greedy_strategy,
        "peak_rebuild_patience": args.peak_rebuild_patience,
        "peak_rebuild_size": args.peak_rebuild_size,
        "peak_rebuild_vertex_limit": args.peak_rebuild_vertex_limit,
        "peak_rebuild_min_sc_delta": args.peak_rebuild_min_sc_delta,
        "post_target_betas": args.post_target_betas,
        "post_target_rounds": args.post_target_rounds,
        "slice_replace_rounds": args.slice_replace_rounds,
        "slice_candidate_limit": args.slice_candidate_limit,
        "replace_candidate_limit": args.replace_candidate_limit,
        "restore_refine_rounds": args.restore_refine_rounds,
        "restore_refine_betas": args.restore_refine_betas,
    }

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    strategies = parse_csv_list(args.greedy_strategies) if args.greedy_strategies else [args.greedy_strategy]
    greedy_alphas = parse_csv_list(args.greedy_alphas, float) if args.greedy_alphas else [args.greedy_alpha]
    candidates = [(seed, strategy, greedy_alpha) for strategy in strategies for greedy_alpha in greedy_alphas for seed in seeds]
    prescreen_results = []
    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=init_worker,
        initargs=(network_data, bond_to_int[output], config),
    ) as pool:
        prescreen_futures = {pool.submit(prescreen_seed, candidate): candidate for candidate in candidates}
        for future in as_completed(prescreen_futures):
            result = future.result()
            prescreen_results.append(result)
            log_event("prescreen_done", **result)

    selected = select_prescreen_candidates(
        prescreen_results,
        min(args.top_k, len(prescreen_results)),
        args.sc_target,
        args.alpha,
    )
    selected_candidates = [(item["seed"], item["strategy"], item["greedy_alpha"]) for item in selected]
    log_event(
        "prescreen_selected",
        candidates=[
            {"seed": seed, "strategy": strategy, "greedy_alpha": greedy_alpha}
            for seed, strategy, greedy_alpha in selected_candidates
        ],
    )

    best_result = None
    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=init_worker,
        initargs=(network_data, bond_to_int[output], config),
    ) as pool:
        trial_futures = {}
        for candidate in selected_candidates:
            log_event(
                "trial_started",
                seed=candidate[0],
                strategy=candidate[1],
                greedy_alpha=candidate[2],
            )
            trial_futures[pool.submit(run_full_trial, candidate)] = candidate
        for future in as_completed(trial_futures):
            candidate = trial_futures[future]
            try:
                result = future.result()
            except Exception as exc:
                log_event(
                    "trial_failed",
                    seed=candidate[0],
                    strategy=candidate[1],
                    greedy_alpha=candidate[2],
                    error=repr(exc),
                    traceback=traceback.format_exc(),
                )
                continue
            log_event("trial_done", **{k: v for k, v in result.items() if k != "objective"})
            if best_result is None or result["objective"] < best_result["objective"]:
                best_result = result
                log_event(
                    "best_updated",
                    seed=result["seed"],
                    slices=result["slices"],
                    tc=result["tc"],
                    sc=result["sc"],
                    mc=result["mc"],
                )

    if best_result is None:
        raise RuntimeError("All full-trial candidates failed; see trial_failed events in the log.")

    final_result = {
        "equation": str(args.equation),
        "output_label": output,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "top_k": args.top_k,
        "selected_candidates": [
            {"seed": seed, "strategy": strategy, "greedy_alpha": greedy_alpha}
            for seed, strategy, greedy_alpha in selected_candidates
        ],
        "time_s": time.perf_counter() - started,
        "best_seed": best_result["seed"],
        "best_strategy": best_result["strategy"],
        "best_greedy_alpha": best_result["greedy_alpha"],
        "order_len": best_result["order_len"],
        "slices": best_result["slices"],
        "tc": best_result["tc"],
        "sc": best_result["sc"],
        "mc": best_result["mc"],
        "greedy_alpha": args.greedy_alpha,
        "greedy_strategy": args.greedy_strategy,
    }
    if args.result_out is not None:
        args.result_out.write_text(json.dumps(final_result, indent=2), encoding="utf-8")
    print(json.dumps(final_result, ensure_ascii=False))


if __name__ == "__main__":
    main()
