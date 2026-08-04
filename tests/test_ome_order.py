import json
import math

import pytest
import torch

from artensor import load_ome_order, read_ome_order
from artensor.contraction import contraction_scheme, tensor_contraction


def ome_plan(*, slices=()):
    # (tensor 1 * tensor 2) * tensor 3, with duplicate-shaped inputs allowed.
    return {
        "label-type": "Char",
        "inputs": [["a", "b"], ["b", "c"], ["c", "a"]],
        "output": [],
        "slices": list(slices),
        "tree": {
            "isleaf": False,
            "eins": {"ixs": [["a", "c"], ["c", "a"]], "iy": []},
            "args": [
                {
                    "isleaf": False,
                    "eins": {
                        "ixs": [["a", "b"], ["b", "c"]],
                        "iy": ["a", "c"],
                    },
                    "args": [
                        {"isleaf": True, "tensorindex": 1},
                        {"isleaf": True, "tensorindex": 2},
                    ],
                },
                {"isleaf": True, "tensorindex": 3},
            ],
        },
    }


def test_read_ome_order_uses_leaf_ids_not_bond_matching(tmp_path):
    plan = ome_plan(slices=("b",))
    filename = tmp_path / "order.json"
    filename.write_text(json.dumps(plan), encoding="utf-8")
    parsed = read_ome_order(filename)
    assert parsed.order == ((0, 1), (0, 2))
    assert parsed.slicing_bonds == ("b",)
    order, slices, tree = load_ome_order(filename)
    assert order == [(0, 1), (0, 2)]
    assert slices == ["b"]
    assert "b" not in tree.tn.bond_dims


def test_structure_relabeling_matches_qec_translation_logic():
    structure = {
        "inputs": [["z", "a"], ["a", "m"], ["m", "z"]],
        "bond_dims": {"a": 2, "m": 3, "z": 5},
    }
    plan = ome_plan()
    plan["label-type"] = "Int64"
    plan["inputs"] = [[3, 1], [1, 2], [2, 3]]
    plan["tree"]["eins"] = {"ixs": [[3, 2], [2, 3]], "iy": []}
    plan["tree"]["args"][0]["eins"] = {
        "ixs": [[3, 1], [1, 2]],
        "iy": [3, 2],
    }
    _, _, tree = load_ome_order(plan, structure=structure)
    assert tree.tn.bond_dims == {1: 2, 2: 3, 3: 5}


def test_structure_relabeling_sorts_numeric_labels_numerically():
    structure = {
        "inputs": [[10, 1], [1, 2], [2, 10]],
        # JSON object keys become strings even when the tensor labels are ints.
        "bond_dims": {"1": 2, "2": 3, "10": 5},
    }
    plan = ome_plan()
    plan["label-type"] = "Int64"
    plan["inputs"] = [[3, 1], [1, 2], [2, 3]]
    plan["tree"]["eins"] = {"ixs": [[3, 2], [2, 3]], "iy": []}
    plan["tree"]["args"][0]["eins"] = {
        "ixs": [[3, 1], [1, 2]],
        "iy": [3, 2],
    }
    _, _, tree = load_ome_order(plan, structure=structure)
    assert tree.tn.bond_dims == {1: 2, 2: 3, 3: 5}


def test_loaded_ome_tree_contracts_numerically():
    plan = ome_plan()
    _, _, tree = load_ome_order(plan, {"a": 2, "b": 3, "c": 4})
    tensors = {
        0: torch.randn(2, 3, dtype=torch.float64),
        1: torch.randn(3, 4, dtype=torch.float64),
        2: torch.randn(4, 2, dtype=torch.float64),
    }
    expected = torch.einsum("ab,bc,ca->", tensors[0], tensors[1], tensors[2])
    scheme, output_bonds = contraction_scheme(tree)
    actual = tensor_contraction(dict(tensors), scheme)
    assert output_bonds == []
    assert torch.allclose(actual, expected)


def test_shared_ome_output_label_is_not_contracted():
    plan = {
        "label-type": "Int64",
        "inputs": [[1, 2], [1, 3]],
        "output": [1, 2, 3],
        "tree": {
            "isleaf": False,
            "eins": {"ixs": [[1, 2], [1, 3]], "iy": [1, 2, 3]},
            "args": [
                {"isleaf": True, "tensorindex": 1},
                {"isleaf": True, "tensorindex": 2},
            ],
        },
    }
    _, _, tree = load_ome_order(plan, {1: 2, 2: 3, 3: 4})
    root = tree.tree[tree.all_tensors]
    assert root.contain_bonds == {1, 2, 3}
    assert root.sc == pytest.approx(3.0 + math.log2(3.0))

    tensors = {
        0: torch.randn(2, 3, dtype=torch.float64),
        1: torch.randn(2, 4, dtype=torch.float64),
    }
    scheme, output_bonds = contraction_scheme(tree)
    actual = tensor_contraction(dict(tensors), scheme)
    expected = torch.einsum("ab,ac->abc", tensors[0], tensors[1])
    actual = actual.permute(*(output_bonds.index(label) for label in (1, 2, 3)))
    assert set(output_bonds) == {1, 2, 3}
    assert torch.allclose(actual, expected)


def test_ome_reader_rejects_nonbinary_or_incomplete_trees():
    plan = ome_plan()
    plan["tree"]["args"] = plan["tree"]["args"][:1]
    with pytest.raises(ValueError, match="binary"):
        read_ome_order(plan)


def test_deep_ome_tree_does_not_depend_on_python_recursion_limit():
    tensor_count = 1500
    tree = {"isleaf": True, "tensorindex": 1}
    for tensor_index in range(2, tensor_count + 1):
        tree = {
            "isleaf": False,
            "args": [
                tree,
                {"isleaf": True, "tensorindex": tensor_index},
            ],
        }
    parsed = read_ome_order(
        {
            "inputs": [[tensor] for tensor in range(tensor_count)],
            "output": [],
            "tree": tree,
        }
    )
    assert len(parsed.order) == tensor_count - 1
    assert parsed.order[-1] == (0, tensor_count - 1)
