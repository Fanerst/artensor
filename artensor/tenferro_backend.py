"""Run artensor's contraction schemes on tenferro.

This module is the Python half of ``backend="tenferro"``. The Rust half is the optional
``artensor_tenferro`` extension in ``tenferro_backend/`` of this repository, which runs on one
tenferro engine (CPU or CUDA).

Each scheme, dense or sparse-state, is compiled here into a short program of binary einsums,
row selections, reshapes and concatenations over numbered slots. The program mirrors
``tensor_contraction`` or ``tensor_contraction_sparse`` step by step. The extension runs it,
including the loop over sliced bonds, so a sliced contraction crosses into Rust once and its
tensors stay on the device.
"""

import contextlib
import functools
import os

import numpy as np
import torch

__all__ = [
    "available",
    "get_backend",
    "tensor_contraction",
    "tensor_contraction_sparse",
    "sliced_contraction",
]

_NUMPY_DTYPES = {
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
    torch.float32: np.float32,
    torch.float64: np.float64,
}


def _extension():
    try:
        import artensor_tenferro
    except ImportError as exc:
        raise ImportError(
            "backend='tenferro' needs the optional artensor-tenferro extension: run "
            "`pip install ./tenferro_backend` in the artensor repository "
            "(see tenferro_backend/README.md)"
        ) from exc
    return artensor_tenferro


def available():
    """Whether the tenferro backend is installed."""
    try:
        _extension()
    except ImportError:
        return False
    return True


def _device_spec(device):
    device = torch.device(device)
    if device.type == "cpu":
        return "cpu"
    if device.type == "cuda":
        return f"cuda:{0 if device.index is None else device.index}"
    raise ValueError(f"the tenferro backend runs on cpu or cuda, not {device}")


@contextlib.contextmanager
def _keep_cpu_set():
    """Give the calling thread back its CPU set when the block ends.

    tenferro 0.12's CPU thread pool (threadteam 0.5) pins the thread that builds or enters it
    to a single CPU. Threads started from it later inherit that one CPU: torch's and MKL's
    workers, or the workers of a second engine (tensor4all/tenferro-rs#2597, fixed on
    tenferro's main by threadteam 0.6).
    """
    if not hasattr(os, "sched_getaffinity"):
        yield
        return
    saved = os.sched_getaffinity(0)
    try:
        yield
    finally:
        if os.sched_getaffinity(0) != saved:
            os.sched_setaffinity(0, saved)


@functools.lru_cache(maxsize=None)
def _backend(spec, threads):
    with _keep_cpu_set():
        return _extension().Backend(spec, threads)


def get_backend(device="cpu", threads=None):
    """The tenferro backend for ``device``, created once and then reused.

    A CPU backend uses ``threads`` worker threads, ``torch.get_num_threads()`` by default.
    Reusing a backend keeps tenferro's plan cache and buffer pool, which its engine owns.
    """
    spec = _device_spec(device)
    if spec != "cpu":
        threads = 1
    elif threads is None:
        threads = torch.get_num_threads()
    return _backend(spec, int(threads))


def _labels(operation):
    """The label lists of a binary operation: an einsum string or a named contraction."""
    if isinstance(operation, str):
        inputs, output = operation.split("->")
        operands = inputs.split(",")
        if len(operands) != 2:
            raise ValueError(f"expected a binary einsum, got {operation!r}")
        return list(operands[0]), list(operands[1]), list(output)
    if isinstance(operation, dict) and operation.get("op") == "binary_named_contract":
        return (
            list(operation["left_labels"]),
            list(operation["right_labels"]),
            list(operation["output_labels"]),
        )
    raise ValueError(f"unsupported contraction operation: {operation!r}")


def _numbered(operation):
    """The operation's labels renumbered 0, 1, ... in order of first appearance."""
    ids = {}

    def number(labels):
        return [ids.setdefault(label, len(ids)) for label in labels]

    left, right, output = _labels(operation)
    return number(left), number(right), number(output)


def _positions(index):
    return [int(p) for p in torch.as_tensor(index).reshape(-1).tolist()]


class _Program:
    """Instructions over numbered slots; the first ``n_inputs`` slots hold the inputs."""

    def __init__(self, n_inputs):
        self.instrs = []
        self.n_slots = n_inputs

    def _new(self):
        slot = self.n_slots
        self.n_slots += 1
        return slot

    def einsum(self, lhs, rhs, operation):
        out = self._new()
        self.instrs.append(("einsum", out, lhs, rhs, *_numbered(operation)))
        return out

    def index_select(self, src, axis, positions):
        out = self._new()
        self.instrs.append(("index_select", out, src, axis, _positions(positions)))
        return out

    def reshape(self, src, shape):
        out = self._new()
        self.instrs.append(("reshape", out, src, [int(d) for d in shape]))
        return out

    def concat(self, srcs, axis):
        out = self._new()
        self.instrs.append(("concat", out, list(srcs), axis))
        return out

    def free(self, *slots):
        self.instrs.extend(("free", slot) for slot in slots)

    def normalize(self, slot):
        self.instrs.append(("normalize", slot))


def _compile_dense(scheme, ids):
    """``tensor_contraction`` as a program: one einsum per step, the result in slot ``i``."""
    slot = {tid: k for k, tid in enumerate(ids)}
    program = _Program(len(ids))
    result = slot[ids[0]] if len(ids) == 1 else None
    for (i, j), operation in scheme:
        out = program.einsum(slot[i], slot[j], operation)
        program.free(slot[i], slot[j])
        slot[i], slot[j] = out, None
        result = out
    if result is None:
        raise ValueError("an empty scheme cannot contract more than one tensor")
    return program, result


def _compile_sparse(scheme, ids, scientific_notation=False):
    """``tensor_contraction_sparse`` as a program, with the same four kinds of step."""
    slot = {tid: k for k, tid in enumerate(ids)}
    program = _Program(len(ids))
    result = slot[ids[0]] if len(ids) == 1 else None
    for step in scheme:
        i, j = step[0]
        operation = step[1]
        batch_i, batch_j = step[2]
        shape = step[3] if len(step) > 3 else None
        if len(batch_i) > 1:
            # Rows in chunks: contract the rows of each chunk, then stack the chunks.
            parts = []
            for rows_i, rows_j in zip(batch_i, batch_j):
                left = program.index_select(slot[i], 0, rows_i)
                right = program.index_select(slot[j], 0, rows_j)
                part = program.einsum(left, right, operation)
                program.free(left, right)
                if shape:
                    reshaped = program.reshape(part, shape)
                    program.free(part)
                    part = reshaped
                parts.append(part)
            out = program.concat(parts, 0)
            program.free(*parts)
        elif len(step) > 3 and len(batch_i) == len(batch_j) == 1:
            # One chunk: pair the selected rows of both sides.
            left = program.index_select(slot[i], 0, batch_i[0])
            right = program.index_select(slot[j], 0, batch_j[0])
            out = program.einsum(left, right, operation)
            program.free(left, right)
        elif len(step) > 3:
            # Every pair of rows, then the rows to keep.
            product = program.einsum(slot[i], slot[j], operation)
            out = program.reshape(product, shape)
            program.free(product)
            if len(batch_i) == 1:
                rows = program.index_select(out, 0, batch_i[0])
                program.free(out)
                out = rows
        else:
            out = program.einsum(slot[i], slot[j], operation)
        program.free(slot[i], slot[j])
        slot[i], slot[j] = out, None
        if scientific_notation:
            program.normalize(out)
        result = out
    if result is None:
        raise ValueError("an empty scheme cannot contract more than one tensor")
    return program, result


def _ids(tensors):
    return list(tensors.keys()) if isinstance(tensors, dict) else list(range(len(tensors)))


def _run(program, result, tensors, ids, device, dtype=None, slicing=()):
    """Run ``program`` on ``tensors``; returns (result on ``device``, log10 factor, dtype)."""
    values = [torch.as_tensor(tensors[tid]) for tid in ids]
    if dtype is None:
        dtype = values[0].dtype
        for value in values[1:]:
            dtype = torch.promote_types(dtype, value.dtype)
    if dtype not in _NUMPY_DTYPES:
        raise TypeError(
            f"the tenferro backend supports complex64, complex128, float32 and float64, not {dtype}"
        )
    arrays = [
        np.ascontiguousarray(
            value.detach().to(device="cpu", dtype=dtype).resolve_conj().resolve_neg().numpy()
        )
        for value in values
    ]
    backend = get_backend(device)
    with _keep_cpu_set():
        factor, array = backend.run(
            arrays, program.instrs, result, program.n_slots, list(slicing)
        )
    return torch.from_numpy(array).to(device), factor, dtype


def tensor_contraction(tensors, scheme):
    """``artensor.tensor_contraction`` on tenferro, on the device of the tensors.

    Unlike the torch path, ``tensors`` is left unchanged.
    """
    ids = _ids(tensors)
    device = torch.as_tensor(tensors[ids[0]]).device
    program, result = _compile_dense(scheme, ids)
    out, _, _ = _run(program, result, tensors, ids, device)
    return out


def tensor_contraction_sparse(tensors, contraction_scheme, scientific_notation=False):
    """``artensor.tensor_contraction_sparse`` on tenferro, on the device of the tensors.

    With ``scientific_notation`` it returns ``(log10 factor, tensor)`` like the torch path, the
    factor as a tensor of the inputs' dtype. ``tensors`` is left unchanged.
    """
    ids = _ids(tensors)
    device = torch.as_tensor(tensors[ids[0]]).device
    program, result = _compile_sparse(contraction_scheme, ids, scientific_notation)
    out, factor, dtype = _run(program, result, tensors, ids, device)
    if scientific_notation:
        return torch.tensor(factor, dtype=dtype, device=device), out
    return out


def sliced_contraction(tensors, scheme, slicing_indices, sparse=False, dtype=None, device="cpu"):
    """The sum of ``scheme`` over every value of the sliced bonds, as one tenferro call.

    ``slicing_indices`` maps each sliced bond to its ``(tensor id, axis)`` sites, each axis counted
    in the unsliced tensor (``TensorNetworkSimulation.slicing_indices``). Slices are summed in the
    order of artensor's own loop. The result lands on ``device`` as a torch tensor.
    """
    ids = _ids(tensors)
    slot = {tid: k for k, tid in enumerate(ids)}
    if sparse:
        program, result = _compile_sparse(scheme, ids)
    else:
        program, result = _compile_dense(scheme, ids)
    slicing = []
    for sites in slicing_indices.values():
        sites = [(slot[tid], int(axis)) for tid, axis in sites]
        first_slot, first_axis = sites[0]
        dim = torch.as_tensor(tensors[ids[first_slot]]).shape[first_axis]
        slicing.append((int(dim), sites))
    out, _, _ = _run(program, result, tensors, ids, device, dtype, slicing)
    return out
