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