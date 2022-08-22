import time
import pytest
from artensor import (
    ContractionTree, 
    AbstractTensorNetwork,
    GreedyOrderFinder
)

# ContractionTree, ContractionVertex, AbstractTensorNetwork class tests 
def test_hyper_tn():
    eq, shapes = 'ab,ac,ad,bc,be,cf,de,df,ef->', [(2,2)]*9
    order = [(0, 1), (3, 5), (0, 3), (4, 8), (0, 4), (6, 7), (0, 6), (0, 2)]
    eq_sep = eq.strip('->').split(',')
    tensor_bonds = [list(e) for e in eq_sep]
    bond_dims = {}
    for i in range(len(tensor_bonds)):
        for j in range(len(tensor_bonds[i])):
            bond_dims[tensor_bonds[i][j]] = shapes[i][j]
    tensor_network = AbstractTensorNetwork(tensor_bonds, bond_dims)
    ctree = ContractionTree(tensor_network, order, 0)
    print('tree results', ctree.tree_complexity())

def test_oridinary_tn():
    eq, shapes = 'abc,ade,cdf,bgh,egi,fhi->', [(2,2,2)] * 6
    order = [(0,1), (3,4), (0,3), (2,6), (0,2), (5,7), (0,5), (0,8)]
    eq_sep = eq.strip('->').split(',')
    tensor_bonds = [list(e) for e in eq_sep]
    bond_dims = {}
    for i in range(len(tensor_bonds)):
        for j in range(len(tensor_bonds[i])):
            bond_dims[tensor_bonds[i][j]] = shapes[i][j]
    tensor_network = AbstractTensorNetwork(tensor_bonds, bond_dims)
    ctree = ContractionTree(tensor_network, order, 0)
    print('tree results', ctree.tree_complexity())

def test_slicing_add():
    eq, shapes = 'abc,ade,cdf,bgh,egi,fhi->', [(2,2,2)] * 6
    order = [(0,1), (3,4), (0,3), (2,6), (0,2), (5,7), (0,5), (0,8)]
    eq_sep = eq.strip('->').split(',')
    tensor_bonds = [list(e) for e in eq_sep]
    bond_dims = {}
    for i in range(len(tensor_bonds)):
        for j in range(len(tensor_bonds[i])):
            bond_dims[tensor_bonds[i][j]] = shapes[i][j]
    tensor_network = AbstractTensorNetwork(tensor_bonds, bond_dims)
    ctree = ContractionTree(tensor_network, order, 0)

    t0 = time.time()
    ctree1 = ctree.copy()
    print('copy time', time.time() - t0)
    t0 = time.time()
    print(ctree.slicing_tree_complexity('a'), time.time() - t0)

    t0 = time.time()
    print(ctree.slicing_tree_complexity_new('a'), time.time() - t0)

    ctree.slicing('a')
    print(ctree.tree_complexity())
    print(ctree1.tree_complexity())

    ctree.add_bond('a')#, {0, 1, 2}, 2)
    print(ctree.tree_complexity())
    print(ctree1.tree_complexity())


def test_multi_configuration():
    print('-'*10, 'multi-bitstring test', '-'*10)
    eq, shapes = 'ab,ac,ad,bc,be,cf,de,df,ef,a,b,c->', [(2,2)]*9 + [(2,)]*3
    order = [(0,1), (3,4), (0,3), (2,6), (0,2), (5,7), (0,5), (0,8), (0,9), (0,10), (0,11)]
    eq_sep = eq.strip('->').split(',')
    tensor_bonds = [list(e) for e in eq_sep]
    bond_dims = {}
    for i in range(len(tensor_bonds)):
        for j in range(len(tensor_bonds[i])):
            bond_dims[tensor_bonds[i][j]] = shapes[i][j]

    tensor_network = AbstractTensorNetwork(tensor_bonds, bond_dims, final_qubits=[9,10,11], max_bitstring=7)
    ctree = ContractionTree(tensor_network, order, 0)
    print('tree results', ctree.tree_complexity())
    for v in ctree.tree.values():
        print(v.contain_tensors, v.contain_bonds, v.tc, v.sc, v.mc)

    greedy_finder = GreedyOrderFinder(tensor_network)
    order_new, _, _ = greedy_finder('min_dim', 0)
    ctree_new = ContractionTree(tensor_network, order_new, 0)
    print(eq_sep)
    print(order_new)
    print(ctree_new.tree_complexity())