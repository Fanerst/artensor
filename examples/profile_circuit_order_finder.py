import argparse
import cProfile
import json
import io
import pstats
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TORCH_DEPS = Path("/tmp/artensor-deps")
if TORCH_DEPS.exists() and str(TORCH_DEPS) not in sys.path:
    sys.path.insert(0, str(TORCH_DEPS))
if str(ROOT / "artensor") not in sys.path:
    sys.path.insert(0, str(ROOT / "artensor"))
if str(ROOT / "artensor" / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "artensor" / "examples"))

from benchmark_circuit_bitstring import build_closed_network
from artensor.order_finder import find_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsim-path", default=str(ROOT / "circuit_n30_m14_s0_e0_pEFGH.qsim"))
    parser.add_argument("--sc-target", type=float, default=32.0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--beta-start", type=float, default=3.0)
    parser.add_argument("--beta-stop", type=float, default=21.0)
    parser.add_argument("--beta-steps", type=int, default=2)
    parser.add_argument("--slicing-repeat", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--greedy-alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-out", default=str(ROOT / "circuit_order_finder.prof"))
    parser.add_argument("--stats-out", default=str(ROOT / "circuit_order_finder_stats.txt"))
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    _, tensor_bonds, bond_dims = build_closed_network(Path(args.qsim_path))
    profiler = cProfile.Profile()
    profiler.enable()
    _, slicing_bonds, ctree = find_order(
        tensor_bonds,
        bond_dims,
        final_qubits=[],
        seed=args.seed,
        max_bitstrings=1,
        sc_target=args.sc_target,
        trials=args.trials,
        iters=args.iters,
        betas=np.linspace(args.beta_start, args.beta_stop, args.beta_steps),
        slicing_repeat=args.slicing_repeat,
        alpha=args.alpha,
        greedy_alpha=args.greedy_alpha,
    )
    profiler.disable()
    profiler.dump_stats(args.profile_out)

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(args.top_n)
    Path(args.stats_out).write_text(stream.getvalue())

    tc, sc, mc = ctree.tree_complexity()
    print(json.dumps({
        "qsim_path": args.qsim_path,
        "profile_out": args.profile_out,
        "stats_out": args.stats_out,
        "tc": tc,
        "sc": sc,
        "mc": mc,
        "slices": len(slicing_bonds),
    }, indent=2))


if __name__ == "__main__":
    main()
