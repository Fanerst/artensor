import numpy as np
import pytest

from artensor import (
    AbstractTensorNetwork,
    ContractionTree,
    GreedyOrderFinder,
    MultiCostGreedyOrderFinder,
)
from artensor.order_finder import score_fn
from artensor.order_finder import _sliced_result_key
from artensor.order_finder import simulate_annealing


def grid_network(side=3, *, final_qubits=(), max_bitstrings=1):
    tensor_bonds = {i: [] for i in range(side**3)}
    bond_dims = {}
    next_bond = 0

    def vertex(x, y, z):
        return (x * side + y) * side + z

    for x in range(side):
        for y in range(side):
            for z in range(side):
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    if x + dx < side and y + dy < side and z + dz < side:
                        left = vertex(x, y, z)
                        right = vertex(x + dx, y + dy, z + dz)
                        tensor_bonds[left].append(next_bond)
                        tensor_bonds[right].append(next_bond)
                        bond_dims[next_bond] = 2
                        next_bond += 1
    return AbstractTensorNetwork(
        tensor_bonds,
        bond_dims,
        final_qubits=final_qubits,
        max_bitstring=max_bitstrings,
    )


def assert_metrics_equal(expected, actual):
    assert np.allclose(expected, actual, rtol=1e-12, atol=1e-12)


def test_score_function_is_stable_for_large_complexities():
    assert np.isfinite(score_fn(500.0, 10.0, 499.0, alpha=32.0))
    assert score_fn(500.0, 10.0, 499.0, alpha=0.0) == 500.0


def test_sliced_result_selection_honors_readwrite_weight():
    low_memory = (4.0, 20.0, 2.0)
    low_flops = (3.9, 20.0, 3.0)
    slicing_bonds = {"slice": 2}

    assert _sliced_result_key(
        low_flops, slicing_bonds, 20.0, 0.0
    ) < _sliced_result_key(low_memory, slicing_bonds, 20.0, 0.0)
    assert _sliced_result_key(
        low_memory, slicing_bonds, 20.0, 64.0
    ) < _sliced_result_key(low_flops, slicing_bonds, 20.0, 64.0)


def test_native_greedy_metrics_match_contraction_tree():
    network = grid_network(3, final_qubits=(0, 13, 26), max_bitstrings=5)
    for strategy in ("min_dim", "max_reduce"):
        order, tc, sc = GreedyOrderFinder(network)(strategy, seed=4)
        assert len(order) == len(network.tensor_bonds) - 1
        tree = ContractionTree(network, order)
        actual_tc, actual_sc, _ = tree.tree_complexity()
        assert_metrics_equal((tc, sc), (actual_tc, actual_sc))


def test_native_anneal_metrics_match_public_tree():
    from artensor import _order_core

    network = grid_network(3)
    initial_order = GreedyOrderFinder(network)("min_dim", seed=3)[0]
    order, tc, sc, mc = _order_core.anneal_order(
        network,
        initial_order,
        np.linspace(0.1, 3.0, 20),
        3,
        3,
        12.0,
        32.0,
        2.0,
    )
    assert len(order) == len(network.tensor_bonds) - 1
    assert_metrics_equal((tc, sc, mc), ContractionTree(network, order).tree_complexity())


def test_multicost_greedy_is_reproducible_and_valid():
    network = grid_network(3)
    finder = MultiCostGreedyOrderFinder(network)
    result1 = finder(seed=11, max_repeats=16, minimize="size")
    result2 = finder(seed=11, max_repeats=16, minimize="size")
    assert result1.order == result2.order
    assert result1.cost_function_id == result2.cost_function_id
    assert result1.repeats == result2.repeats == 16
    assert_metrics_equal(
        (result1.tc, result1.sc),
        ContractionTree(network, result1.order).tree_complexity()[:2],
    )


@pytest.mark.parametrize("cost_function_id", range(8))
def test_each_multicost_function_returns_a_valid_order(cost_function_id):
    network = grid_network(2)
    result = MultiCostGreedyOrderFinder(network)(
        seed=5,
        minimize="flops",
        max_repeats=1,
        cost_function_id=cost_function_id,
    )
    assert result.cost_function_id == cost_function_id
    assert result.repeats == 1
    assert_metrics_equal(
        (result.tc, result.sc),
        ContractionTree(network, result.order).tree_complexity()[:2],
    )


def test_multicost_time_budget_always_completes_one_path():
    result = MultiCostGreedyOrderFinder(grid_network(3))(
        max_repeats=100,
        max_time=1e-12,
    )
    assert result.repeats == 1


def test_python_greedy_fallback_remains_available():
    network = grid_network(2)
    order, tc, sc = GreedyOrderFinder(network, use_compiled=False)(seed=2)
    assert len(order) == len(network.tensor_bonds) - 1
    assert_metrics_equal(
        (tc, sc),
        ContractionTree(network, order).tree_complexity()[:2],
    )


def test_python_treesa_fallback_remains_available():
    network = grid_network(2)
    order, slicing_bonds = simulate_annealing(
        network,
        sc_target=1.0e9,
        trials=1,
        iters=1,
        betas=[0.1],
        slicing_repeat=0,
        workers=1,
        use_compiled=False,
    )
    assert len(order) == len(network.tensor_bonds) - 1
    assert slicing_bonds == {}
    assert all(np.isfinite(ContractionTree(network, order).tree_complexity()))


def test_hyperedge_is_retained_until_all_incident_tensors_are_merged():
    network = AbstractTensorNetwork(
        {0: ["h", "a"], 1: ["h", "b"], 2: ["h", "c"]},
        {"h": 2, "a": 2, "b": 2, "c": 2},
        output_bonds=("a", "b", "c"),
    )
    network.contract(0, 1)
    assert "h" in network.tensor_bonds[0]
    assert network.bond_tensors["h"] == {0, 2}
    network.contract(0, 2)
    assert "h" not in network.tensor_bonds[0]
