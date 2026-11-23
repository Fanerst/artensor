"""backend='tenferro' against the torch path.

The tenferro cases run when the optional artensor_tenferro extension is installed (CUDA cases
also need a CUDA build and device); the torch-only cases always run.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from artensor import (
    NumericalTensorNetwork,
    TensorNetworkCircuit,
    contraction_scheme_sparse,
    find_order,
    quantum_circuit_simulation,
    tensor_contraction,
    tensor_contraction_sparse,
)
from artensor.contraction import _einsum_operation
from artensor.simulation import TensorNetworkSimulation

try:
    import artensor_tenferro
except ImportError:
    artensor_tenferro = None

CIRCUIT = Path(__file__).with_name("circuit_n12_m14_s0_e0_pEFGH.qsim")

# The amplitudes test_circuits.py checks.
CORRECTNESS = {
    "100001000001": 0.0198028199 + 1j * 0.0106442748,
    "000101111011": 0.00497586094 + 1j * -0.0245072283,
    "011000101100": -0.00853562169 + 1j * -0.00701293815,
    "111001100001": -0.0100137182 + 1j * 0.0147468708,
    "001110110000": 0.00681955926 + 1j * 0.0106616206,
}

# Devices for the tenferro cases; a test parametrized over an empty list is skipped.
DEVICES = []
if artensor_tenferro is not None:
    DEVICES.append("cpu")
    if artensor_tenferro.cuda_available() and torch.cuda.is_available():
        DEVICES.append("cuda")

TOLERANCE = {
    torch.complex64: dict(rtol=1e-4, atol=1e-6),
    torch.complex128: dict(rtol=1e-10, atol=1e-13),
    torch.float64: dict(rtol=1e-10, atol=1e-13),
}


def assert_close(actual, expected, dtype):
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected.detach().cpu().numpy(), **TOLERANCE[dtype]
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.complex128, torch.float64])
def test_dense_scheme_with_a_hyperedge_matches_torch(device, dtype):
    generator = torch.Generator().manual_seed(0)
    tensors = {
        0: torch.randn(2, 3, dtype=dtype, generator=generator),
        1: torch.randn(2, 4, dtype=dtype, generator=generator),
        2: torch.randn(2, 3, 4, 5, dtype=dtype, generator=generator),
    }
    # Bond "a" joins all three tensors, so the first step keeps it.
    scheme = [((0, 1), "ab,ac->abc"), ((0, 2), "abc,abcd->d")]
    expected = tensor_contraction(dict(tensors), scheme)
    moved = {k: v.to(device) for k, v in tensors.items()}
    result = tensor_contraction(moved, scheme, backend="tenferro")
    assert result.device.type == torch.device(device).type
    assert result.dtype == dtype
    assert result.shape == expected.shape
    assert_close(result, expected, dtype)
    assert torch.equal(moved[0], tensors[0].to(device)), "the tenferro path changed its inputs"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_sparse_circuit_amplitudes_match_torch(device, dtype):
    bitstrings = list(CORRECTNESS)
    expected, order = quantum_circuit_simulation(
        CIRCUIT, bitstrings, 30, trial_num=1, dtype=dtype, device="cpu"
    )
    result, order_tenferro = quantum_circuit_simulation(
        CIRCUIT, bitstrings, 30, trial_num=1, dtype=dtype, device=device, backend="tenferro"
    )
    assert list(order_tenferro) == list(order)
    assert result.device.type == torch.device(device).type
    assert result.dtype == dtype
    assert_close(result, expected, dtype)
    for bitstring, amplitude in zip(order_tenferro, result.cpu()):
        assert np.allclose(amplitude.item(), CORRECTNESS[bitstring], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_full_state_amplitudes_match_torch(device):
    expected, _ = quantum_circuit_simulation(
        CIRCUIT, [], 30, trial_num=1, dtype=torch.complex128, device="cpu"
    )
    result, _ = quantum_circuit_simulation(
        CIRCUIT, [], 30, trial_num=1, dtype=torch.complex128, device=device, backend="tenferro"
    )
    assert result.shape == expected.shape == (2,) * 12
    assert_close(result, expected, torch.complex128)
    amplitudes = result.reshape(-1).cpu().numpy()
    for bitstring, amplitude in CORRECTNESS.items():
        assert np.allclose(amplitudes[int(bitstring, 2)], amplitude, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_sliced_sparse_amplitudes_match_torch(device):
    # Sparse-state, because the simulation functions do not protect the open output bonds of a
    # full-state network from slicing.
    simulation = TensorNetworkSimulation.from_circuit_file(CIRCUIT, list(CORRECTNESS))
    simulation.prepare_contraction(sc_target=10, trials=1, iters=10)
    assert simulation.slicing_indices, "sc_target=10 should force slicing"
    expected = simulation.contraction(dtype=torch.complex128)
    result = simulation.contraction(dtype=torch.complex128, device=device, backend="tenferro")
    assert result.device.type == torch.device(device).type
    assert_close(result, expected, torch.complex128)
    for bitstring, amplitude in zip(simulation.bitstrings_sorted, result.cpu()):
        assert np.allclose(amplitude.item(), CORRECTNESS[bitstring], rtol=1e-4, atol=1e-6)


def _sparse_scheme(bitstrings):
    circuit = TensorNetworkCircuit(CIRCUIT)
    tensors, tensor_bonds, bond_dims, final_qubits = circuit.to_numerical_tn()
    network = NumericalTensorNetwork(tensors, tensor_bonds, bond_dims, final_qubits)
    bonds, final_qubit_inds = network._simplify("sparse")
    _, slicing_bonds, ctree = find_order(
        bonds, network.bond_dims, final_qubit_inds, 0, len(bitstrings),
        sc_target=30, trials=1, iters=10, slicing_repeat=1,
        betas=np.linspace(3.0, 21.0, 61),
    )
    assert not slicing_bonds
    scheme, _, order = contraction_scheme_sparse(ctree, bitstrings, sc_target=30)
    reordered = {
        i: network.tensors[j].to(torch.complex128)
        for i, j in enumerate(network.tensors.keys())
    }
    return scheme, reordered, order


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_scheme_with_scientific_notation_matches_torch(device):
    scheme, tensors, _ = _sparse_scheme(list(CORRECTNESS))
    expected_factor, expected = tensor_contraction_sparse(
        dict(tensors), scheme, scientific_notation=True
    )
    moved = {k: v.to(device) for k, v in tensors.items()}
    factor, result = tensor_contraction_sparse(
        moved, scheme, scientific_notation=True, backend="tenferro"
    )
    assert factor.dtype == torch.complex128
    assert factor.device.type == torch.device(device).type
    assert_close(factor, expected_factor, torch.complex128)
    assert_close(result, expected, torch.complex128)


@pytest.mark.parametrize("device", DEVICES)
def test_named_binary_operation_matches_torch(device):
    # 54 labels: more than torch.einsum's alphabet here, so artensor uses its named fallback.
    left_labels = list(range(30))
    right_labels = list(range(24, 54))
    output_labels = list(range(24)) + list(range(30, 54))
    operation = _einsum_operation((left_labels, right_labels), output_labels)
    assert isinstance(operation, dict)

    generator = torch.Generator().manual_seed(2)
    left_shape = [2 if label in (0, 1, 24, 25) else 1 for label in left_labels]
    right_shape = [2 if label in (24, 25, 52, 53) else 1 for label in right_labels]
    tensors = {
        0: torch.randn(left_shape, dtype=torch.complex128, generator=generator),
        1: torch.randn(right_shape, dtype=torch.complex128, generator=generator),
    }
    scheme = [((0, 1), operation, [[torch.tensor([0])], [torch.tensor([0])]])]
    expected = tensor_contraction_sparse(dict(tensors), scheme)
    moved = {k: v.to(device) for k, v in tensors.items()}
    result = tensor_contraction_sparse(moved, scheme, backend="tenferro")
    assert result.shape == expected.shape
    assert_close(result, expected, torch.complex128)


@pytest.mark.skipif(
    artensor_tenferro is None or not hasattr(os, "sched_getaffinity"),
    reason="needs artensor_tenferro and per-thread CPU affinity (Linux)",
)
def test_tenferro_leaves_the_callers_cpu_set_alone():
    # tenferro 0.12 pins the thread that builds or enters its CPU pool to one CPU; threads that
    # torch starts later would inherit it (tensor4all/tenferro-rs#2597).
    from artensor import tenferro_backend

    before = os.sched_getaffinity(0)
    tenferro_backend.get_backend("cpu", threads=3)  # a fresh engine, built in this test
    assert os.sched_getaffinity(0) == before
    tensors = {0: torch.randn(4, 5, dtype=torch.float64), 1: torch.randn(5, 6, dtype=torch.float64)}
    result = tensor_contraction(tensors, [((0, 1), "ab,bc->ac")], backend="tenferro")
    assert torch.allclose(result, tensors[0] @ tensors[1])
    assert os.sched_getaffinity(0) == before


def test_unknown_backend_is_rejected():
    tensors = {0: torch.ones(2), 1: torch.ones(2)}
    with pytest.raises(ValueError, match="unknown backend"):
        tensor_contraction(tensors, [((0, 1), "a,a->")], backend="jax")
