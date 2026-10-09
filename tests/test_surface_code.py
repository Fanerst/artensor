"""
Order finding on the tensor network of a distance-5, 5-round rotated surface code.

``surface_code_d5r5.dem`` is the detector error model of the stim circuit

    stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=5, rounds=5,
        after_clifford_depolarization=0.001,
        before_measure_flip_probability=0.001,
        after_reset_flip_probability=0.001,
    ).detector_error_model(decompose_errors=False, flatten_loops=True)

The network gives the probability of a batch of syndromes. Every detector is a
hyperedge shared by the tensors of the error mechanisms that flip it and by its
syndrome vector, which also carries the batch index. The batch index is an output bond.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from artensor import AbstractTensorNetwork, ContractionTree, GreedyOrderFinder
from artensor.order_finder import score_fn, simulate_annealing

# The pure-Python annealer is too slow for a network of this size.
pytest.importorskip("artensor._order_core")

ALPHA = 4.0
SHORT_ANNEALING = dict(
    iters=5,
    betas=np.linspace(3.0, 21.0, 21),
    alpha=ALPHA,
    workers=1,
)


def surface_code_network(batch=50):
    dem_file = Path(__file__).with_name("surface_code_d5r5.dem")
    errors = [
        [int(target[1:]) for target in line.split()[1:] if target.startswith("D")]
        for line in dem_file.read_text().splitlines()
        if line.startswith("error")
    ]
    detectors = sorted(set().union(*errors))
    tensor_bonds = errors + [["batch", detector] for detector in detectors]
    bond_dims = {detector: 2 for detector in detectors}
    bond_dims["batch"] = batch
    return AbstractTensorNetwork(tensor_bonds, bond_dims, output_bonds=["batch"])


def test_surface_code_network_structure():
    network = surface_code_network()
    assert len(network.tensor_bonds) == 1677 + 120
    assert len(network.bond_dims) == 120 + 1
    assert len(network.bond_tensors["batch"]) == 120
    degrees = [
        len(tensors) for bond, tensors in network.bond_tensors.items() if bond != "batch"
    ]
    assert (min(degrees), max(degrees)) == (4, 81)


def test_surface_code_annealing_without_slicing():
    network = surface_code_network()
    sc_target, trials = 30.0, 2
    greedy = GreedyOrderFinder(network)
    greedy_scores = []
    for seed in range(trials):
        tree = ContractionTree(network, greedy("min_dim", seed)[0])
        greedy_scores.append(score_fn(*tree.tree_complexity(), sc_target, ALPHA))

    reported = []
    order, slicing_bonds = simulate_annealing(
        network,
        sc_target=sc_target,
        trials=trials,
        slicing=False,
        trial_callback=lambda *result: reported.append(result),
        **SHORT_ANNEALING,
    )
    assert slicing_bonds == {}
    assert len(order) == len(network.tensor_bonds) - 1
    tree = ContractionTree(network, order)
    assert tree.tree[tree.all_tensors].contain_bonds == {"batch"}

    scores = [score_fn(*metrics, sc_target, ALPHA) for _, *metrics in reported]
    assert [trial for trial, *_ in reported] == list(range(trials))
    assert np.isclose(score_fn(*tree.tree_complexity(), sc_target, ALPHA), min(scores))
    # Every trial improves on the greedy order it starts from.
    assert all(score < start for score, start in zip(scores, greedy_scores))


def test_surface_code_slicing_keeps_the_batch_bond():
    network = surface_code_network()
    # Far below what the short annealing reaches (above 70), so the trial has to slice.
    sc_target = 40.0
    order, slicing_bonds = simulate_annealing(
        network, sc_target=sc_target, trials=1, slicing_repeat=0, **SHORT_ANNEALING
    )
    assert len(slicing_bonds) > 0
    assert "batch" not in slicing_bonds
    for bond in slicing_bonds:
        network.slicing(bond)
    tree = ContractionTree(network, order)
    assert tree.tree_complexity()[1] <= sc_target
    assert tree.tree[tree.all_tensors].contain_bonds == {"batch"}


@pytest.mark.skipif(
    os.environ.get("ARTENSOR_SLOW_TESTS") != "1",
    reason="full search, about half a minute; set ARTENSOR_SLOW_TESTS=1 to run it",
)
def test_surface_code_full_search_reaches_sc_target():
    # With a batch of 50 the best orders found have sc = log2(50) + 23 = 28.64.
    network = surface_code_network(batch=50)
    sc_target = 30.0
    order, slicing_bonds = simulate_annealing(
        network,
        sc_target=sc_target,
        trials=2,
        iters=300,
        betas=np.linspace(3.0, 21.0, 61),
        alpha=ALPHA,
        workers=1,
        slicing=False,
    )
    assert slicing_bonds == {}
    assert ContractionTree(network, order).tree_complexity()[1] <= sc_target
