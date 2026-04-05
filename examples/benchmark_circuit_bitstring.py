import argparse
import ast
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TORCH_DEPS = Path("/tmp/artensor-deps")
if TORCH_DEPS.exists() and str(TORCH_DEPS) not in sys.path:
    sys.path.insert(0, str(TORCH_DEPS))
if str(ROOT / "artensor") not in sys.path:
    sys.path.insert(0, str(ROOT / "artensor"))

from artensor.circuit import TensorNetworkCircuit
from artensor.order_finder import find_order


def load_cirq_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def eval_ast_number(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_ast_number(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return eval_ast_number(node.operand)
    if isinstance(node, ast.BinOp):
        left = eval_ast_number(node.left)
        right = eval_ast_number(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "np" and node.attr == "pi":
            return np.pi
    raise ValueError(f"Unsupported numeric AST: {ast.dump(node)}")


def parse_qubit_call(node):
    if not isinstance(node, ast.Call):
        raise ValueError(f"Unsupported qubit node: {ast.dump(node)}")
    func = node.func
    if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "cirq" and func.attr == "GridQubit"):
        raise ValueError(f"Unsupported qubit call: {ast.dump(node)}")
    return tuple(int(eval_ast_number(arg)) for arg in node.args)


def parse_gate_expr(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "on":
        gate_name, params = parse_gate_expr(node.func.value)
        qubits = [parse_qubit_call(arg) for arg in node.args]
        return gate_name, params, qubits
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "cirq":
            if node.func.attr == "PhasedXPowGate":
                phase = next(eval_ast_number(kw.value) for kw in node.keywords if kw.arg == "phase_exponent")
                exponent = next(eval_ast_number(kw.value) for kw in node.keywords if kw.arg == "exponent")
                if abs(phase - 0.25) < 1e-12 and abs(exponent - 0.5) < 1e-12:
                    return "hz_1_2", ()
            if node.func.attr == "Rz":
                return "rz", (float(eval_ast_number(node.args[0])),)
            if node.func.attr == "FSimGate":
                theta = next(eval_ast_number(kw.value) for kw in node.keywords if kw.arg == "theta")
                phi = next(eval_ast_number(kw.value) for kw in node.keywords if kw.arg == "phi")
                return "fsim", (float(theta), float(phi))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        if isinstance(node.left, ast.Attribute) and isinstance(node.left.value, ast.Name) and node.left.value.id == "cirq":
            exponent = eval_ast_number(node.right)
            if abs(exponent - 0.5) < 1e-12:
                if node.left.attr == "X":
                    return "x_1_2", ()
                if node.left.attr == "Y":
                    return "y_1_2", ()
    raise ValueError(f"Unsupported gate AST: {ast.dump(node)}")


def write_qsim_from_ast_module(module_path: Path, out_path: Path):
    module_ast = ast.parse(module_path.read_text())
    qubit_order = None
    circuit_moments = None
    for node in module_ast.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "QUBIT_ORDER" for target in node.targets):
            qubit_order = [parse_qubit_call(elt) for elt in node.value.elts]
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CIRCUIT" for target in node.targets):
            moments_kw = next(kw for kw in node.value.keywords if kw.arg == "moments")
            circuit_moments = moments_kw.value.elts
    if qubit_order is None or circuit_moments is None:
        raise ValueError(f"Could not parse circuit module {module_path}")
    qubit_to_id = {qubit: idx for idx, qubit in enumerate(qubit_order)}
    lines = [str(len(qubit_order))]
    for layer_idx, moment in enumerate(circuit_moments):
        ops_kw = next(kw for kw in moment.keywords if kw.arg == "operations")
        for op in ops_kw.value.elts:
            name, params, qubits = parse_gate_expr(op)
            qubit_ids = [qubit_to_id[qubit] for qubit in qubits]
            parts = [str(layer_idx), name, *[str(qid) for qid in qubit_ids]]
            parts.extend(repr(value) for value in params)
            lines.append(" ".join(parts))
    out_path.write_text("\n".join(lines) + "\n")


def gate_to_qsim_name(gate):
    gate_name = gate.__class__.__name__
    if gate_name == "PhasedXPowGate":
        if abs(gate.phase_exponent - 0.25) < 1e-12 and abs(gate.exponent - 0.5) < 1e-12:
            return "hz_1_2", ()
    if gate_name == "_PauliXPowerGate" and abs(gate.exponent - 0.5) < 1e-12:
        return "x_1_2", ()
    if gate_name == "_PauliYPowerGate" and abs(gate.exponent - 0.5) < 1e-12:
        return "y_1_2", ()
    if gate_name == "Rz":
        return "rz", (float(gate._rads),)
    if gate_name == "FSimGate":
        return "fsim", (float(gate.theta), float(gate.phi))
    raise ValueError(f"Unsupported gate: {gate!r}")


def write_qsim_from_cirq_module(module_path: Path, out_path: Path):
    try:
        module = load_cirq_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name != "cirq":
            raise
        write_qsim_from_ast_module(module_path, out_path)
        return
    qubit_to_id = {(qubit.row, qubit.col): idx for idx, qubit in enumerate(module.QUBIT_ORDER)}
    lines = [str(len(module.QUBIT_ORDER))]
    for layer_idx, moment in enumerate(module.CIRCUIT):
        for op in moment.operations:
            name, params = gate_to_qsim_name(op.gate)
            qubit_ids = [qubit_to_id[(qubit.row, qubit.col)] for qubit in op.qubits]
            parts = [str(layer_idx), name, *[str(qid) for qid in qubit_ids]]
            parts.extend(repr(value) for value in params)
            lines.append(" ".join(parts))
    out_path.write_text("\n".join(lines) + "\n")


def filtered_bond_dims(tensor_bonds, bond_dims):
    active_bonds = sorted({bond for bonds in tensor_bonds.values() for bond in bonds})
    return {bond: bond_dims[bond] for bond in active_bonds}


def build_closed_network(qsim_path: Path):
    nqubits = int(qsim_path.read_text().splitlines()[0])
    circuit = TensorNetworkCircuit(str(qsim_path), final_state="0" * nqubits)
    _, tensor_bonds, bond_dims, _ = circuit.to_numerical_tn()
    bond_dims = filtered_bond_dims(tensor_bonds, bond_dims)
    return nqubits, tensor_bonds, bond_dims


def export_einsum_json(nqubits, tensor_bonds, bond_dims, out_path: Path):
    bond_ids = sorted(bond_dims)
    bond_to_label = {bond: idx + 1 for idx, bond in enumerate(bond_ids)}
    payload = {
        "einsum": {
            "ixs": [[bond_to_label[bond] for bond in tensor_bonds[tensor_id]] for tensor_id in sorted(tensor_bonds)],
            "iy": [],
        },
        "size": {str(bond_to_label[bond]): int(bond_dims[bond]) for bond in bond_ids},
        "meta": {
            "num_tensors": len(tensor_bonds),
            "num_labels": len(bond_ids),
            "closed": True,
            "bitstring": "0" * nqubits,
        },
    }
    out_path.write_text(json.dumps(payload))
    return payload


def run_artensor(tensor_bonds, bond_dims, args):
    started = time.perf_counter()
    _, slicing_bonds, ctree = find_order(
        tensor_bonds,
        bond_dims,
        final_qubits=[],
        seed=args.seed,
        max_bitstrings=1,
        sc_target=args.sc_target,
        trials=args.trials,
        iters=args.iters,
        betas=np.linspace(args.beta_start, args.beta_stop, args.beta_steps),
        slicing_repeat=args.slicing_repeat,
        alpha=args.alpha,
        greedy_alpha=args.greedy_alpha,
    )
    elapsed = time.perf_counter() - started
    tc, sc, mc = ctree.tree_complexity()
    return {
        "engine": "artensor",
        "time_s": elapsed,
        "tc": tc,
        "sc": sc,
        "mc": mc,
        "slices": len(slicing_bonds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default=str(ROOT / "artensor" / "examples" / "circuit_n30_m14_s0_e0_pEFGH.py"))
    parser.add_argument("--qsim-path", default=str(ROOT / "artensor" / "examples" / "circuits" / "circuit_n53_m20_s0_e0_pABCDCDAB.qsim"))
    parser.add_argument("--qsim-out", default=str(ROOT / "circuit_n30_m14_s0_e0_pEFGH.qsim"))
    parser.add_argument("--einsum-out", default=str(ROOT / "circuit_n30_m14_s0_e0_pEFGH_one_bitstring_einsum.json"))
    parser.add_argument("--result-out", default=str(ROOT / "circuit_n30_m14_s0_e0_pEFGH_artensor_result.json"))
    parser.add_argument("--sc-target", type=float, default=32.0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--beta-start", type=float, default=3.0)
    parser.add_argument("--beta-stop", type=float, default=21.0)
    parser.add_argument("--beta-steps", type=int, default=61)
    parser.add_argument("--slicing-repeat", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--greedy-alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    module_path = Path(args.module) if args.module else None
    qsim_path = Path(args.qsim_path)
    qsim_out = Path(args.qsim_out)
    einsum_out = Path(args.einsum_out)
    result_out = Path(args.result_out)

    if module_path is not None:
        write_qsim_from_cirq_module(module_path, qsim_out)
        benchmark_qsim = qsim_out
    else:
        benchmark_qsim = qsim_path
    nqubits, tensor_bonds, bond_dims = build_closed_network(benchmark_qsim)
    einsum_payload = export_einsum_json(nqubits, tensor_bonds, bond_dims, einsum_out)
    if args.export_only:
        payload = {
            "qsim_path": str(benchmark_qsim),
            "einsum_path": str(einsum_out),
            "num_tensors": einsum_payload["meta"]["num_tensors"],
            "num_labels": einsum_payload["meta"]["num_labels"],
        }
        result_out.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return
    result = run_artensor(tensor_bonds, bond_dims, args)
    result["inputs"] = {
        "qsim_path": str(benchmark_qsim),
        "einsum_path": str(einsum_out),
        "num_tensors": einsum_payload["meta"]["num_tensors"],
        "num_labels": einsum_payload["meta"]["num_labels"],
        "sc_target": args.sc_target,
        "trials": args.trials,
        "iters": args.iters,
        "betas": [args.beta_start, args.beta_stop, args.beta_steps],
        "slicing_repeat": args.slicing_repeat,
        "alpha": args.alpha,
        "greedy_alpha": args.greedy_alpha,
        "seed": args.seed,
    }
    result_out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
