import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artensor.order_finder import find_order


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

    tensor_bonds, bond_dims, output = parse_qec_equation(args.equation)
    betas = np.linspace(args.beta_start, args.beta_stop, args.beta_steps)

    start = time.perf_counter()
    order, slicing_bonds, ctree = find_order(
        tensor_bonds,
        bond_dims,
        sc_target=args.sc_target,
        open_bonds=[output],
        trials=1,
        iters=args.iters,
        betas=betas,
        slicing_repeat=args.slicing_repeat,
        start_seed=args.seed,
        alpha=args.alpha,
        greedy_alpha=args.greedy_alpha,
        update_mode="optimized",
    )
    elapsed = time.perf_counter() - start
    tc, sc, mc = ctree.tree_complexity()

    result = {
        "equation": str(args.equation),
        "output_label": output,
        "seed": args.seed,
        "time_s": elapsed,
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
