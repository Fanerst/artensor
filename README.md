# artensor
Generating contraction orders and perform numerical contractions for arbitrary tensor networks

## Installation

Since this package has not been upload to pypi, thus you need to install it manually. Firstly you need clone this repository by
```
git clone https://github.com/Fanerst/artensor.git
```
Before installing this package, you may need to refer the dependence requirements listed in `requirements.txt`.
Then go to the main directory and
```
pip install .
```
If you want to make it editable while using it as a package, you may install it as editable model
```
pip install -e .
```

## Running examples

Please refer the `examples/sycamore.ipynb` for a detailed example. This example shows how to use this package to do full-amplitude and sparse-state simulation of the Sycamore circuit with 30 qubits and 14 cycles.

## Order finding

The default greedy and TreeSA paths use the compiled C++ search core installed
by `pip install .` or `pip install -e .`. The ordinary greedy optimizer retains
a pure-Python fallback; the 2024 multi-cost portfolio requires the extension.

```python
import numpy as np
from artensor import (
    AbstractTensorNetwork,
    GreedyOrderFinder,
    MultiCostGreedyOrderFinder,
)
from artensor.order_finder import simulate_annealing

tn = AbstractTensorNetwork(tensor_bonds, bond_dims)

order, tc, sc = GreedyOrderFinder(tn)("min_dim", seed=0)

portfolio = MultiCostGreedyOrderFinder(tn)(
    seed=0,
    minimize="flops",       # or "size"
    max_repeats=128,         # max_time can impose a wall-time budget
)
print(portfolio.cost_function_name, portfolio.tc, portfolio.sc)

order, sliced_bonds = simulate_annealing(
    tn,
    sc_target=30,
    trials=8,
    iters=50,
    betas=np.linspace(3.0, 21.0, 61),
    workers=1,
)
```

`tc` follows Artensor's existing log10 convention and `sc` is log2. Set
`greedy_strategy="multi_cost"` on `simulate_annealing` to initialize TreeSA
from the portfolio optimizer.

### Reading an OME order

OMEinsumContractionOrders JSON trees can be used directly without a separate
translation script:

```python
from artensor import load_ome_order

order, sliced_bonds, contraction_tree = load_ome_order(
    "order.json",
    structure="tn_structure.json",  # restores dimensions after OME relabeling
)
```

`read_ome_order` performs parsing only. `load_ome_order` additionally builds
the Artensor tensor network and contraction tree, preserves explicit OME output
labels (including shared batch labels), and applies OME slices.

Sparse-state accounting remains opt-in: pass `final_qubits` and
`max_bitstrings` to `load_ome_order` only when loading a sparse-state problem.
They default to an ordinary tensor-network contraction.

## Citations
Please kindly cite the following paper if you use this package as part of you research.
1. Feng Pan, and Pan Zhang, *"Simulation of Quantum Circuits Using the Big-Batch Tensor Network Method."* [Phys. Rev. Lett. **128**, 030501 (2022)](https://doi.org/10.1103/PhysRevLett.128.030501).
2. Gleb Kalachev, Pavel Panteleev, Man-Hong Yung, *"Multi-Tensor Contraction for XEB Verification of Quantum Circuits."* [arxiv:2108.05665](https://arxiv.org/abs/2108.05665).
3. Feng Pan, Keyang Chen, and Pan Zhang, *"Solving the Sampling Problem of the Sycamore Quantum Circuits."* [Phys. Rev. Lett. **129**, 090502](https://doi.org/10.1103/PhysRevLett.129.090502).
4. Feng Pan, Hanfeng Gu, Lvlin Kuang, Bing Liu, Pan Zhang, *"Efficient Quantum Circuit Simulation by
Tensor Network Methods on Modern GPUs."* [arxiv:2310.03978](https://arxiv.org/abs/2310.03978).
5. Oliver Orgler and Albert Blacher, *"A Multi-Cost Approach to Efficient Tensor
Network Contraction Path Finding."* [arXiv:2405.09644](https://arxiv.org/abs/2405.09644).
