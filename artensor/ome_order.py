"""Read OMEinsumContractionOrders JSON plans into Artensor.

OMEinsumContractionOrders serializes a binary ``NestedEinsum`` tree whose leaf
``tensorindex`` values use Julia's one-based indexing.  Artensor uses pairwise
orders over zero-based tensor IDs.  This module converts between those two
representations without trying to identify tensors from their bond lists (which
is ambiguous when an einsum contains duplicate inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from .contraction_tree import ContractionTree
from .tensor_network import AbstractTensorNetwork


JSONSource = str | PathLike[str] | Mapping[str, Any] | TextIO


def _sorted_labels(labels):
    labels = list(labels)
    try:
        return sorted(labels)
    except TypeError:
        return sorted(labels, key=lambda value: (type(value).__name__, repr(value)))


@dataclass(frozen=True)
class OMEOrder:
    """An OME order translated to Artensor's zero-based representation."""

    tensor_bonds: dict[int, list[Any]]
    output_bonds: tuple[Any, ...]
    order: tuple[tuple[int, int], ...]
    slicing_bonds: tuple[Any, ...]
    label_type: str | None = None


def _read_json(source: JSONSource) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    if hasattr(source, "read"):
        data = json.load(source)
    else:
        with Path(source).expanduser().open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("an OME order must be encoded as a JSON object")
    return data


def _tree_to_order(tree: Mapping[str, Any], tensor_count: int) -> tuple[tuple[int, int], ...]:
    order: list[tuple[int, int]] = []
    leaves: list[int] = []
    locations: dict[int, int] = {}
    stack: list[tuple[Mapping[str, Any], bool]] = [(tree, False)]

    while stack:
        node, expanded = stack.pop()
        if node.get("isleaf"):
            try:
                tensor_id = int(node["tensorindex"]) - 1
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid OME leaf tensorindex") from exc
            if not 0 <= tensor_id < tensor_count:
                raise ValueError(
                    f"OME leaf tensorindex {tensor_id + 1} is outside 1:{tensor_count}"
                )
            leaves.append(tensor_id)
            locations[id(node)] = tensor_id
            continue

        args = node.get("args")
        if (
            not isinstance(args, list)
            or len(args) != 2
            or not all(isinstance(child, Mapping) for child in args)
        ):
            raise ValueError("Artensor requires a binary OME contraction tree")
        if not expanded:
            stack.append((node, True))
            stack.append((args[1], False))
            stack.append((args[0], False))
            continue

        left_rep = locations[id(args[0])]
        right_rep = locations[id(args[1])]
        if left_rep == right_rep:
            raise ValueError("an OME internal node contains the same subtree twice")
        order.append((left_rep, right_rep))
        locations[id(node)] = left_rep

    if len(leaves) != tensor_count or set(leaves) != set(range(tensor_count)):
        raise ValueError("OME tree leaves must contain every input tensor exactly once")
    return tuple(order)


def read_ome_order(source: JSONSource) -> OMEOrder:
    """Parse an OME JSON order without constructing a tensor network.

    Parameters
    ----------
    source:
        A filename, open text stream, or already-decoded JSON mapping produced
        by ``OMEinsumContractionOrders.writejson``.
    """

    data = _read_json(source)
    inputs = data.get("inputs")
    tree = data.get("tree")
    if not isinstance(inputs, list) or not isinstance(tree, Mapping):
        raise ValueError("OME order JSON must contain 'inputs' and 'tree'")
    tensor_bonds = {i: list(bonds) for i, bonds in enumerate(inputs)}
    order = _tree_to_order(tree, len(tensor_bonds))
    return OMEOrder(
        tensor_bonds=tensor_bonds,
        output_bonds=tuple(data.get("output", ())),
        order=order,
        slicing_bonds=tuple(data.get("slices", ())),
        label_type=data.get("label-type"),
    )


def load_ome_order(
    source: JSONSource,
    bond_dims: Mapping[Any, int | float] | None = None,
    final_qubits: Sequence[int] = (),
    max_bitstrings: int = 1,
    *,
    default_bond_dim: int | float = 2,
    structure: JSONSource | None = None,
) -> tuple[list[tuple[int, int]], list[Any], ContractionTree]:
    """Load an OME JSON plan as an Artensor contraction tree.

    The return value matches :func:`artensor.find_order`: pairwise order,
    sliced bonds, and :class:`~artensor.ContractionTree`.  OME JSON does not
    store bond dimensions, so callers may supply them; otherwise every bond is
    assigned ``default_bond_dim``.  ``structure`` accepts the original network
    JSON when OME relabeled non-integer bonds to sorted one-based integers.
    """

    ome_order = read_ome_order(source)
    if structure is not None:
        structure_data = _read_json(structure)
        structure_inputs = structure_data.get("inputs")
        if not isinstance(structure_inputs, list):
            raise ValueError("structure JSON must contain an 'inputs' list")
        original_labels = _sorted_labels(
            {label for bonds in structure_inputs for label in bonds}
        )
        label_map = {
            label: index for index, label in enumerate(original_labels, start=1)
        }
        mapped_inputs = [
            [label_map[label] for label in bonds]
            for bonds in structure_inputs
        ]
        if mapped_inputs != list(ome_order.tensor_bonds.values()):
            raise ValueError(
                "the structure labels do not map to the inputs stored in the OME order"
            )
        structure_dims = structure_data.get("bond_dims")
        if not isinstance(structure_dims, Mapping):
            raise ValueError("structure JSON must contain a 'bond_dims' mapping")
        mapped_dims = {}
        for label in original_labels:
            if label in structure_dims:
                dimension = structure_dims[label]
            elif str(label) in structure_dims:
                dimension = structure_dims[str(label)]
            else:
                raise ValueError(f"structure bond_dims is missing label {label!r}")
            mapped_dims[label_map[label]] = dimension
        if bond_dims is not None and dict(bond_dims) != mapped_dims:
            raise ValueError("bond_dims conflicts with dimensions in structure JSON")
        bond_dims = mapped_dims
    all_bonds = {
        bond for bonds in ome_order.tensor_bonds.values() for bond in bonds
    }
    if bond_dims is None:
        resolved_dims = {bond: default_bond_dim for bond in all_bonds}
    else:
        missing = all_bonds.difference(bond_dims)
        if missing:
            preview = sorted(map(repr, missing))[:5]
            raise ValueError(f"bond_dims is missing OME labels: {preview}")
        resolved_dims = {bond: bond_dims[bond] for bond in all_bonds}

    tensor_network = AbstractTensorNetwork(
        {i: list(bonds) for i, bonds in ome_order.tensor_bonds.items()},
        resolved_dims,
        list(final_qubits),
        max_bitstrings,
        ome_order.output_bonds,
    )
    for bond in ome_order.slicing_bonds:
        if bond not in tensor_network.bond_dims:
            raise ValueError(f"OME slice label {bond!r} is not present in the inputs")
        tensor_network.slicing(bond)

    order = list(ome_order.order)
    slicing_bonds = list(ome_order.slicing_bonds)
    tree = ContractionTree(tensor_network, order)
    return order, slicing_bonds, tree


__all__ = ["OMEOrder", "read_ome_order", "load_ome_order"]
