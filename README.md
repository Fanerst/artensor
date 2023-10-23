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

## Citations
Please kindly cite the following paper if you use this package as part of you research.
1. Feng Pan, and Pan Zhang, *"Simulation of Quantum Circuits Using the Big-Batch Tensor Network Method."* [Phys. Rev. Lett. **128**, 030501 (2022)](https://doi.org/10.1103/PhysRevLett.128.030501).
2. Gleb Kalachev, Pavel Panteleev, Man-Hong Yung, *"Multi-Tensor Contraction for XEB Verification of Quantum Circuits."* [arxiv:2108.05665](https://arxiv.org/abs/2108.05665).
3. Feng Pan, Keyang Chen, and Pan Zhang, *"Solving the Sampling Problem of the Sycamore Quantum Circuits."* [Phys. Rev. Lett. **129**, 090502](https://doi.org/10.1103/PhysRevLett.129.090502).
4. Feng Pan, Hanfeng Gu, Lvlin Kuang, Bing Liu, Pan Zhang, *"Efficient Quantum Circuit Simulation by
Tensor Network Methods on Modern GPUs."* [arxiv:2310.03978](https://arxiv.org/abs/2310.03978).

