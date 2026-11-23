"""Slicing several bonds of one tensor, with every available backend."""

import pytest
import torch

from artensor import tensor_contraction
from artensor.simulation import TensorNetworkSimulation

try:
    import artensor_tenferro
except ImportError:
    artensor_tenferro = None

BACKENDS = [("torch", "cpu")]
if artensor_tenferro is not None:
    BACKENDS.append(("tenferro", "cpu"))
    if artensor_tenferro.cuda_available() and torch.cuda.is_available():
        BACKENDS.append(("tenferro", "cuda"))


@pytest.mark.parametrize("backend, device", BACKENDS)
def test_two_sliced_bonds_on_one_tensor(backend, device):
    generator = torch.Generator().manual_seed(1)
    t0 = torch.randn(2, 3, 2, dtype=torch.complex128, generator=generator)  # bonds p q r
    t1 = torch.randn(2, 4, dtype=torch.complex128, generator=generator)  # bonds p s
    t2 = torch.randn(3, 4, 2, dtype=torch.complex128, generator=generator)  # bonds q s r
    expected = torch.einsum("pqr,ps,qsr->", t0, t1, t2)

    simulation = TensorNetworkSimulation(
        {0: t0, 1: t1, 2: t2},
        {0: ["p", "q", "r"], 1: ["p", "s"], 2: ["q", "s", "r"]},
        {"p": 2, "q": 3, "r": 2, "s": 4},
        [], [], "normal", 1,
    )
    # p and r are sliced. Both sit on tensor 0, and each site's axis counts the unsliced tensor.
    simulation.slicing_indices = {"p": [(0, 0), (1, 0)], "r": [(0, 2), (2, 2)]}
    simulation.scheme = [((0, 1), "a,b->ab"), ((0, 2), "ab,ab->")]
    simulation.output_bonds = []
    simulation.tensor_contraction_func = tensor_contraction

    result = simulation.contraction(dtype=torch.complex128, device=device, backend=backend)
    assert torch.allclose(result.cpu(), expected)
