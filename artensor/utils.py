import numpy as np

LOG10_2 = np.log10(2.0)


def log2_accum_dims(bond_dims, bonds):
    """
    Return log2 of production of bond dimensions of given bonds
    """
    return sum(np.log2(bond_dims[bond]) for bond in bonds)


def log2_accum_cached(log2_bond_dims, bonds):
    """
    Return the sum of cached log2 bond dimensions for the given bonds.
    """
    return sum(log2_bond_dims[bond] for bond in bonds)

def final_qubits_num(num_fq, contain_tensors):
    """
    Calculate contained final qubits in a node set
    """
    return sum(num_fq[i] for i in contain_tensors)

def log10sumexp2(s):
    values = tuple(s)
    if not values:
        return 0
    ms = max(values)
    return np.log10(sum(np.exp2(value - ms) for value in values)) + ms * LOG10_2


def log10sumexp2_pair(a, b):
    ms = max(a, b)
    return np.log10(np.exp2(a - ms) + np.exp2(b - ms)) + ms * LOG10_2

def log2sumexp2(s):
    values = tuple(s)
    if not values:
        return 0
    ms = max(values)
    return np.log2(sum(np.exp2(value - ms) for value in values)) + ms
