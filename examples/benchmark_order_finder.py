import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artensor.order_finder import find_order


def closed_torus(m, n, dim=2):
    tensor_bonds = {}
    bond_dims = {}

    def tid(i, j):
        return i * n + j

    for i in range(m):
        for j in range(n):
            right = f"h-{i}-{j}"
            down = f"v-{i}-{j}"
            tensor_bonds[tid(i, j)] = [
                right,
                down,
                f"h-{i}-{(j - 1) % n}",
                f"v-{(i - 1) % m}-{j}",
            ]
            bond_dims[right] = dim
            bond_dims[down] = dim
    return tensor_bonds, bond_dims


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    tensor_bonds, bond_dims = closed_torus(size, size)
    params = dict(
        sc_target=18,
        trials=1,
        iters=60,
        betas=np.linspace(0.5, 8.0, 24),
        slicing_repeat=1,
        start_seed=0,
        alpha=32.0,
    )

    print(f"closed torus benchmark: {size}x{size}")
    for mode in ("legacy", "optimized"):
        start = time.perf_counter()
        order, slicing_bonds, ctree = find_order(
            tensor_bonds,
            bond_dims,
            update_mode=mode,
            **params,
        )
        elapsed = time.perf_counter() - start
        tc, sc, mc = ctree.tree_complexity()
        print(
            f"{mode:>10} time={elapsed:.3f}s tc={tc:.6f} sc={sc:.1f} mc={mc:.6f} "
            f"sliced={len(slicing_bonds)} order_len={len(order)}"
        )


if __name__ == "__main__":
    main()
