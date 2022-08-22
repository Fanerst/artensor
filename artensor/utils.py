import numpy as np

def log2_accum_dims(bond_dims, bonds):
    """
    Return log2 of production of bond dimensions of given bonds
    """
    return np.sum(np.log2([bond_dims[bond] for bond in bonds]))

def final_qubits_num(num_fq, contain_tensors):
    """
    Calculate contained final qubits in a node set
    """
    return sum([num_fq[i] for i in contain_tensors])

def log10sumexp2(s):
    s = np.array(s)
    if len(s) == 0:
        return 0
    else:
        ms = max(s)
        return np.log10(sum(np.exp2(s - ms))) + ms * np.log10(2)

def log2sumexp2(s):
    s = np.array(s)
    if len(s) == 0:
        return 0
    else:
        ms = max(s)
        return np.log2(sum(np.exp2(s - ms))) + ms