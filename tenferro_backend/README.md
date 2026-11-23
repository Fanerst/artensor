# artensor-tenferro

An optional contraction backend for artensor built on
[tenferro](https://github.com/tensor4all/tenferro-rs), a Rust tensor library, running on the
CPU or on CUDA GPUs. With it installed, artensor's contraction functions accept
`backend="tenferro"`:

```python
import torch
from artensor import quantum_circuit_simulation

amplitudes, bitstrings = quantum_circuit_simulation(
    "circuit.qsim", bitstrings, sc_target=30, device="cuda", backend="tenferro"
)
```

The same argument works on `tensor_contraction`, `tensor_contraction_sparse`,
`tensor_network_contraction` and `TensorNetworkSimulation.contraction`. Inputs and results stay
torch tensors, and the result lands on `device`. The torch backend remains the default.

## Install

You need Rust 1.96 or newer (see [rustup.rs](https://rustup.rs)). From the artensor
repository:

```bash
pip install ./tenferro_backend
```

For CUDA, enable the `cuda` feature at build time:

```bash
MATURIN_PEP517_ARGS="--features cuda" pip install ./tenferro_backend
```

At run time a CUDA build needs an NVIDIA driver, the CUDA 12 runtime libraries (cuBLAS) and
cuTENSOR. If tenferro cannot find a library, point it there with `TENFERRO_CUBLAS_PATH` or
`TENFERRO_CUTENSOR_PATH`. Check the build with:

```python
import artensor_tenferro
artensor_tenferro.CUDA_BUILD, artensor_tenferro.cuda_available()
```

For development, `maturin develop --release` builds the extension into the active environment.

## How it works

`artensor/tenferro_backend.py` compiles a contraction scheme into a short program over numbered
slots. The program has binary einsums, row selections, reshapes and concatenations, and follows
`tensor_contraction` or `tensor_contraction_sparse` step by step. The extension runs it on one
tenferro engine, and the loop over sliced bonds runs inside the extension too. So a sliced
contraction is one call, and its tensors stay on the GPU between slices. Each array crosses
between NumPy and tenferro once, without reordering. A row-major array is used as the
column-major tensor with its axes reversed, and the program's axes and labels are mapped to
match.

The backend keeps one engine per device (`artensor.tenferro_backend.get_backend`). A CPU engine
uses `torch.get_num_threads()` threads, so `torch.set_num_threads` sets both backends.

tenferro 0.12's CPU thread pool (threadteam 0.5) pins the thread that builds or enters it to a
single CPU (tensor4all/tenferro-rs#2597). Any thread started from it later inherits that CPU,
including torch's and MKL's workers. artensor therefore gives the calling thread back its CPU
set after every call into tenferro. Call the extension through artensor, not directly. In a
direct test on a 96-CPU host, torch's next four 3072 × 3072 matmuls took 76 s instead of 0.44 s.

## Differences from the torch backend

- **No autograd:** results do not track gradients.
- **Inputs are left unchanged:** the torch functions overwrite entries of the `tensors` they are
  given; the tenferro path does not.
- **Scientific notation:** `tensor_contraction_sparse(..., scientific_notation=True)` is
  supported, but not inside a sliced contraction. The simulation functions never combine them.
- **Dtypes:** complex64, complex128, float32 and float64.
- **GPUs:** CUDA contractions go through cuTENSOR. On a V100S (sm_70), the cuTENSOR 2 library
  tested here refuses to create a handle (`CUTENSOR_STATUS_NOT_SUPPORTED`), so use newer GPUs.
  The tests ran on an A100.

## Performance

The test is a 30-qubit, 14-cycle random circuit with 1,000 sparse-state amplitudes, on one
artensor scheme (10^11.7 flops, largest tensor 2^30). Each timed call starts from host tensors
and ends with the result on the device. Both backends ran alternately in one process, with
tenferro 0.12 and torch 2.5.1, on 2026-11-23:

| device | dtype | torch | tenferro |
|---|---|---|---|
| A100 80GB PCIe | complex64 | 0.85–1.06 s | 9.7–10.0 s |
| A100 80GB PCIe | complex128 | 1.3–2.2 s | 19.8–20.5 s |
| 16 threads of 2× Xeon Platinum 8268 | complex64 | 40.7–40.8 s | 114–133 s |
| 16 threads of 2× Xeon Platinum 8268 | complex128 | 75–78 s | 215–229 s |

The amplitudes agree to 1e-6 (complex64) and 2e-15 (complex128). The tenferro backend is about
3× slower on the CPU and 10× slower on the A100 here. On CUDA, the known costs are inside
tenferro:

- every cuTENSOR plan evicted from its 32-entry cache drains the stream (tensor4all/tenferro-rs#2706);
- the 4 GiB pool cap described below.

The backend already merges axes before every gather and concatenation, which avoids
tensor4all/tenferro-rs#2708.

## Versions

The crate builds against tenferro 0.12 from crates.io, the latest release when it was written.
Two later speedups are on tenferro's `main` and not yet released:

- tensor4all/tenferro-rs#2716 drops a one-thread copy from CPU einsum with large results;
- tensor4all/tenferro-rs#2721 reads strided operands in place on CUDA.

Moving to the next release is a version bump in `Cargo.toml`.

On CUDA, tenferro 0.12 keeps at most 4 GiB of freed buffers for reuse, even when the
`TENFERRO_CUDA_POOL_RELEASE_MIB` environment variable asks for more. Above that, large
intermediates are mapped again at every synchronization. tensor4all/tenferro-rs#2717 lets an
explicit value raise the budget.
