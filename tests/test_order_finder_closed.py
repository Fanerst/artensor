import numpy as np
import sys
from pathlib import Path

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


def test_closed_torus_order_similarity():
    tensor_bonds, bond_dims = closed_torus(4, 4)
    params = dict(
        sc_target=12,
        trials=1,
        iters=4,
        betas=np.linspace(0.5, 4.0, 6),
        slicing_repeat=1,
        start_seed=0,
        alpha=32.0,
    )

    legacy = find_order(tensor_bonds, bond_dims, update_mode="legacy", **params)
    optimized = find_order(tensor_bonds, bond_dims, update_mode="optimized", **params)

    legacy_ctree = legacy[2]
    optimized_ctree = optimized[2]
    legacy_tc, legacy_sc, _ = legacy_ctree.tree_complexity()
    optimized_tc, optimized_sc, _ = optimized_ctree.tree_complexity()

    assert len(legacy[0]) == len(tensor_bonds) - 1
    assert len(optimized[0]) == len(tensor_bonds) - 1
    assert legacy_ctree.tree[legacy_ctree.all_tensors].contain_bonds == set()
    assert optimized_ctree.tree[optimized_ctree.all_tensors].contain_bonds == set()
    assert abs(optimized_sc - legacy_sc) <= 2
    assert abs(optimized_tc - legacy_tc) <= 1.0
