import sys
from traceback import print_exc
from math import log10, log2, ceil
from copy import deepcopy
import numpy as np
import multiprocessing as mp
import time

from .utils import(
    log2_accum_dims,
    final_qubits_num,
    log10sumexp2,
    log2sumexp2
)


class AbstractTensorNetwork:
    def __init__(self, tensor_bonds, bond_dims, final_qubits=[], max_bitstring=1) -> None:
        """
        Class of abstract tensor network
        Parameters:
        -----------
        tensor_bonds: list of sets
            represent bonds in each individual tensor represented by the index of this list
        bond_dims: dict
            key is the bond, and value is the bond dimension
        final_qubits: list of ints
            tensor id of final qubits, used for multi-bitstring contraction, the final qubit tensor
            will have additional data dimension, the complexity will be calculated by the factor
        max_bitstring: int
            maximum number of bitstrings to calculate during contraction
        -----------
        """
        self.tensor_bonds = tensor_bonds
        self.bond_dims = bond_dims
        self.bond_tensors = {bond: set() for bond in self.bond_dims.keys()} # determine tensors corresponding to each bond
        for i in range(len(tensor_bonds)):
            for j in tensor_bonds[i]:
                self.bond_tensors[j].add(i)
        self.final_qubits = final_qubits
        if final_qubits:
            self.num_fq = [1 if i in final_qubits else 0 for i in range(len(tensor_bonds))]
        else:
            self.num_fq = [0 for i in range(len(tensor_bonds))]
        self.max_bitstring = max_bitstring
        self.log2_max_bitstring = log2(max_bitstring)
        self.slicing_bonds = {}
        self.slicing_bond_tensors = {}
        pass
    
    def slicing(self, bond):
        """
        slicing a bond in the tensor network
        """
        assert bond in self.bond_dims.keys()
        assert bond not in self.slicing_bonds.keys()
        dim = self.bond_dims.pop(bond)
        tensors = self.bond_tensors.pop(bond)
        for tensor_id in tensors:
            self.tensor_bonds[tensor_id].remove(bond)
        self.slicing_bonds[bond] = dim
        self.slicing_bond_tensors[bond] = tensors
    
    def add_bond(self, bond):
        """
        adding a bond that already been sliced back to the tensor network
        """
        assert bond not in self.bond_dims.keys()
        assert bond in self.slicing_bonds.keys()
        dim = self.slicing_bonds.pop(bond)
        tensors = self.slicing_bond_tensors.pop(bond)
        self.bond_dims[bond] = dim
        self.bond_tensors[bond] = tensors
        for tensor_id in tensors:
            self.tensor_bonds[tensor_id].append(bond)
        return tensors

    def sub_tensor_network(self, tensor_lists, tensor_bonds=None):
        sub_num_fq = [final_qubits_num(self.num_fq, tensor_list) for tensor_list in tensor_lists]
        sub_tn = AbstractTensorNetwork(tensor_bonds, self.bond_dims, [], self.max_bitstring)
        sub_tn.num_fq = sub_num_fq
        return sub_tn

class ContractionVertex:
    def __init__(self, contain_tensors, tn, left, right) -> None:
        """
        Class of contraction vertex
        Parameters:
        -----------
        contain_tensors: set
            determine which tensors in this contraction vertex
        tn: AbstractTensorNetwork class
            the underlying tensor network
        left: ContractionVertex class or None
            left leaf, None for childless vertex
        right: ContractionVertex class or None
            right leaf, None for childless vertex
        -----------
        """
        self.update_info(contain_tensors, tn, left, right)

    def update_info(self, contain_tensors, tn, left, right):
        """
        Calculate and store contraction information
        """
        self.contain_tensors = contain_tensors
        # self.involved_bonds = left.involved_bonds | right.involved_bonds if left and right else set().union(*[tn.tensor_bonds[tensor_id] for tensor_id in contain_tensors])
        self.left = left
        self.right = right
        if left and right:
            self.tc, self.sc, self.contain_bonds, self.mc, self.contract_bonds, self.all_bonds = get_tc_sc_contraction(tn, left, right)
        else:
            self.tc, self.sc, self.contain_bonds, self.mc = get_tc_sc_inner(tn, contain_tensors)
            self.all_bonds = self.contain_bonds

    def is_leaf(self):
        if self.left and self.right:
            return False
        else:
            return True


def bonds_out(tensor_bonds, part):
    """
    Calculating resulting bonds after contracting tensors in part
    """
    all_bonds = set().union(*[tensor_bonds[tensor_id] for tensor_id in part])
    other_bonds = set().union(*[tensor_bonds[tensor_id] for tensor_id in range(len(tensor_bonds)) if tensor_id not in part])
    out_bonds = all_bonds & other_bonds
    return out_bonds


def get_tc_sc_inner(tn:AbstractTensorNetwork, part):
    """
    Calculating complexity of specific tensors in part
    return tc, sc, resulting_bonds and mc
    """
    assert len(part) == 1
    bonds1 = set(tn.tensor_bonds[list(part)[0]])# bonds_out(tn.tensor_bonds, part)
    multiconfig_factor = min(tn.log2_max_bitstring, final_qubits_num(tn.num_fq, part))
    return 0.0, log2_accum_dims(tn.bond_dims, bonds1) + multiconfig_factor, bonds1, 0.0


def get_tc_sc_contraction(tn:AbstractTensorNetwork, left:ContractionVertex, right:ContractionVertex):
    """
    Calculating complexity of contracting tensors in left and right
    return tc, sc, resulting_bonds and mc
    """
    contracted_tensors = left.contain_tensors | right.contain_tensors
    all_bonds = left.contain_bonds | right.contain_bonds
    common_bonds = left.contain_bonds & right.contain_bonds
    contract_bonds = set([bond for bond in common_bonds if tn.bond_tensors[bond].issubset(contracted_tensors)])
    result_bonds = all_bonds - contract_bonds

    l_num_fq = final_qubits_num(tn.num_fq, left.contain_tensors)
    r_num_fq = final_qubits_num(tn.num_fq, right.contain_tensors)
    num_fq = l_num_fq + r_num_fq
    # assert final_qubits_num(tn.num_fq, contracted_tensors) == num_fq
    multiconfig_factor = min(tn.log2_max_bitstring, num_fq)
    batch_contraction_penalty = 0.0
    if l_num_fq < tn.log2_max_bitstring and r_num_fq < tn.log2_max_bitstring and multiconfig_factor < num_fq:
        batch_contraction_penalty = num_fq - ceil(tn.log2_max_bitstring)
    elif max(l_num_fq, r_num_fq) >= tn.log2_max_bitstring:
        batch_contraction_penalty = max(0, min(l_num_fq, r_num_fq) - len(contract_bonds))

    tc = log2_accum_dims(tn.bond_dims, all_bonds) if contract_bonds else log2_accum_dims(tn.bond_dims, all_bonds) - 1 # not -1, fix later
    sc = log2_accum_dims(tn.bond_dims, result_bonds)
    tc += multiconfig_factor + batch_contraction_penalty
    sc += multiconfig_factor
    mc = log2sumexp2([left.sc, right.sc, sc])
    return tc, sc, result_bonds, mc, contract_bonds, all_bonds


def score_fn(tc, sc, mc, sc_target=30.0, sc_weight=2.0, arithmetic_intensity=32.0):
    """
    Score function for finding order
    """
    return log10(arithmetic_intensity * 10 ** mc + 10 ** tc) + sc_weight * log10(2) * max(0, sc - sc_target)

class ContractionTree:
    def __init__(self, tn:AbstractTensorNetwork, order, seed=0) -> None:
        """
        Class of contraction tree
        Parameters:
        -----------
        tn: AbstractTensorNetwork class
            the underlying tensor network
        order: list of set
            contraction order to construct the contraction tree
        seed: int
            seem useless currently, remove in the future update
        -----------
        """
        self.order = order
        self.tn = tn
        self.all_tensors = frozenset(range(len(self.tn.tensor_bonds)))
        self.tree = self.construct_contractiontree(order)

        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def construct_contractiontree(self, order):
        """
        Construct the contraction tree according to the order
        """
        self.order = order
        tree = {}
        current_branch = {}
        for edge in order:
            m, n = edge
            if m not in current_branch.keys():
                left = ContractionVertex(frozenset([m]), self.tn, None, None)
                tree[left.contain_tensors] = left
            else:
                left = current_branch[m]
            if n not in current_branch.keys():
                right = ContractionVertex(frozenset([n]), self.tn, None, None)
                tree[right.contain_tensors] = right
            else:
                right = current_branch[n]
            merged_contain_tensors = left.contain_tensors | right.contain_tensors
            parent = ContractionVertex(merged_contain_tensors, self.tn, left, right)
            tree[merged_contain_tensors] = parent
            current_branch[m] = parent
            # left.parent = parent
            # right.parent = parent
        return tree

    def vertex_list(self, sequence='leaves-root'):
        """
        Enumerate the vertices as a list from leaves to root or from root to leaves
        """
        vertex_list = [self.tree[self.all_tensors]]
        queue = [self.tree[self.all_tensors]]

        while len(queue):
            vertex = queue.pop(0)
            if vertex.left and vertex.right:
                next_vertices = [vertex.left, vertex.right]
                vertex_list += next_vertices
                queue += next_vertices
        
        if sequence == 'leaves-root':
            vertex_list.reverse()
        else:
            assert sequence == 'root-leaves'

        return vertex_list

    def select_slicing_bonds(self):
        """
        Select the set of bonds which make up the biggest intermediate tensors to slice
        """
        _, sc, _ = self.tree_complexity()
        slicing_bonds_pool = set().union(*[vertex.contain_bonds for vertex in self.tree.values() if vertex.sc == sc])
        # if len(slicing_bonds_pool) == 0:
        #     for vertex in self.tree.values():
        #         if vertex.sc == sc:
        #             print(vertex.contain_tensors)
        #             print(vertex.contain_bonds)
        #             print(vertex.contract_bonds)
        #             print(vertex.tc, vertex.sc)
        assert len(slicing_bonds_pool) > 0
        return slicing_bonds_pool

    def slicing(self, bond):
        """
        Slice a bond and update the contraction information of all involved contraction vertex 
        """
        self.tn.slicing(bond)
        # print(bond, len(self.tree.values()))
        for vertex in self.vertex_list():# self.tree.values():
            if bond in vertex.all_bonds:
            # if bond in vertex.contain_bonds or (vertex.left and bond in vertex.left.contain_bonds) or (vertex.right and bond in vertex.right.contain_bonds):
                # print('-'*20)
                # print(vertex.contain_tensors, vertex.involved_bonds)
                # if vertex.left and vertex.right:
                #     print(vertex.left.contain_tensors, vertex.left.contain_bonds)
                #     print(vertex.right.contain_tensors, vertex.right.contain_bonds)
                vertex.update_info(vertex.contain_tensors, self.tn, vertex.left, vertex.right)
                # print(vertex.contain_tensors, vertex.contain_bonds, vertex.tc, vertex.sc)
    
    def add_bond(self, bond):
        """
        Add a bond already been sliced and update the contraction information of all involved contraction vertex 
        """
        tensors = self.tn.add_bond(bond)
        for vertex in self.vertex_list():# self.tree.values():
            if tensors & vertex.contain_tensors:
                vertex.update_info(vertex.contain_tensors, self.tn, vertex.left, vertex.right)

    def slicing_tree_complexity(self, bond):
        """
        Calculate the complexity after slicing a bond
        TODO:
        1. find a smarter way to do this
        """
        slicing_tree = self.copy()
        slicing_tree.slicing(bond)
        result = slicing_tree.tree_complexity()
        del slicing_tree
        return result

    def slicing_tree_complexity_new(self, bond):
        tcs, mcs, scs = [], [], []
        for vertex in self.tree.values():
            # print(vertex.contain_tensors, vertex.contain_bonds, vertex.tc, vertex.sc, vertex.mc)
            if bond in vertex.all_bonds:
                sc = vertex.sc - log2(self.tn.bond_dims[bond]) if bond in vertex.contain_bonds else vertex.sc
                if vertex.left and vertex.right:
                    tc = vertex.tc - log2(self.tn.bond_dims[bond])
                    if bond in vertex.contract_bonds and len(vertex.contract_bonds) == 1:
                        tc -= 1
                    sc_left = vertex.left.sc - log2(self.tn.bond_dims[bond]) if bond in vertex.left.all_bonds else vertex.left.sc
                    sc_right = vertex.right.sc - log2(self.tn.bond_dims[bond]) if bond in vertex.right.all_bonds else vertex.right.sc
                    mc = log2sumexp2([sc_left, sc_right, sc])
                    tcs.append(tc)
                    scs.append(sc)
                    mcs.append(mc)
                else:
                    tc = 0.0
                    mc = 0.0
                    scs.append(sc)
            else:
                tc, sc, mc = vertex.tc, vertex.sc, vertex.mc
                if vertex.left and vertex.right:
                    tcs.append(tc)
                    scs.append(sc)
                    mcs.append(mc)
                else:
                    scs.append(sc)
            # print(tc, sc, mc)
        # print(tcs, scs, mcs)
        tc = log10sumexp2(tcs)
        sc = max(scs)
        mc = log10sumexp2(mcs)

        return tc, sc, mc
    
    def add_bond_complexity(self, bond):
        """
        Calculate the complexity after adding a bond
        TODO:
        1. find a smarter way to do this
        """
        add_bond_tree = self.copy()
        add_bond_tree.add_bond(bond)
        result = add_bond_tree.tree_complexity()
        del add_bond_tree
        return result
    
    # def check_contractiontree(self):
    #     root = self.tree[-1]
    #     self.tree = [root]
    #     current_vertices = [root]
    #     while current_vertices:
    #         next_vertices = []
    #         for vertex in current_vertices:
    #             if vertex.left and vertex.right:
    #                 next_vertices += [vertex.left, vertex.right]
    #                 self.tree += [vertex.left, vertex.right]
    #         current_vertices = next_vertices
    #     self.tree.reverse()
    
    def tree_to_order(self):
        """
        Return a contractin order in breadth-first search way according to the contraction tree
        """
        # tree = self.tree
        current_vertices = [self.tree[self.all_tensors]]
        order = []
        while current_vertices:
            next_vertices = []
            for vertex in current_vertices:
                if vertex.left and vertex.right:
                    next_vertices += [vertex.left, vertex.right]
                    rep_tensors = [min(vertex.left.contain_tensors), min(vertex.right.contain_tensors)]
                    order.append((min(rep_tensors), max(rep_tensors)))
            current_vertices = next_vertices
        order.reverse()
        return order

    def tree_order_dfs(self):
        """
        Return a contractin order in depth-first search way according to the contraction tree
        """
        stack = [self.tree[self.all_tensors]]
        order = []
        while len(stack):
            vertex = stack.pop()
            if vertex.left and vertex.right:
                rep_tensors = [min(vertex.left.contain_tensors), min(vertex.right.contain_tensors)]
                order.append((min(rep_tensors), max(rep_tensors)))
                if vertex.left.sc > vertex.right.sc:
                    stack += [vertex.left, vertex.right]
                else:
                    stack += [vertex.right, vertex.left]
        order.reverse()
        return order
    
    def spanning_tree(self, root, size=8):
        """
        Find a subtree and its leaves of size corresponding to the root/subroot
        """
        stack = [root]
        leaves = []
        tree_vertices = []

        while len(stack) + len(leaves) < size and len(stack):
            vertex = stack.pop(0)
            tree_vertices.append(vertex)

            if len(vertex.contain_tensors) == 1:
                leaves.append(vertex)
            else:
                stack.append(vertex.left)
                stack.append(vertex.right)

        tree_leaves = stack + leaves
        tree_vertices += stack
        tree_vertices.reverse()

        return tree_leaves, tree_vertices

    def apply_order(self, order, tree_leaves, local_tree, root):
        """
        apply a new contraction order into the tree/subtree
        """
        for vertex in local_tree[:-1]:
            if vertex not in tree_leaves:
                self.tree.pop(vertex.contain_tensors)

        for i, j in order:
            left, right = tree_leaves[i], tree_leaves[j]
            if (i, j) != order[-1]:
                merged_contain_tensors = left.contain_tensors | right.contain_tensors
                parent = ContractionVertex(merged_contain_tensors, self.tn, left, right)
                self.tree[merged_contain_tensors] = parent
            else:
                try:
                    assert left.contain_tensors | right.contain_tensors == root.contain_tensors
                except:
                    print(root.contain_tensors)
                    print(i, j, order, len(tree_leaves))
                    for leave in tree_leaves:
                        print(leave.contain_tensors)
                    print_exc()
                    sys.exit(0)
                parent = root
                root.update_info(root.contain_tensors, self.tn, left, right)
                # root.left = left
                # root.right = right
            # left.parent = parent
            # right.parent = parent
            tree_leaves[i] = parent

    def tree_complexity(self, tree=None, root=None):
        """
        Calculate the contractin complexity of the contraction tree
        """
        if tree is None:
            tree = self.tree.values()
        if root is None:
            root = self.tree[self.all_tensors]
        # assert root == tree[-1]
        current_vertices = [root]
        tcs, scs = [], []
        mcs = []
        while current_vertices:
            next_vertices = []
            for vertex in current_vertices:
                left, right = vertex.left, vertex.right
                if left in tree and right in tree:
                    # factor = min(log2(self.tn.max_bitstring), final_qubits_num(self.tn.num_fq, vertex.contain_tensors))
                    if left and right:
                        next_vertices += [left, right]
                        scs.append(vertex.sc)
                        tcs.append(vertex.tc)
                        mcs.append(vertex.mc)
                        # print(vertex.contain_tensors, vertex.tc, vertex.sc)
                    else:
                        scs.append(vertex.sc)
                        # print(vertex.contain_tensors, vertex.sc)
                else:
                    # factor = min(log2(self.tn.max_bitstring), final_qubits_num(self.tn.num_fq, vertex.contain_tensors))
                    scs.append(vertex.sc)
            current_vertices = next_vertices
        # print('tree complexity:', tcs, scs)
        tc = log10sumexp2(tcs)
        sc = max(scs)
        mc = log10sumexp2(mcs)
        # score = score_fn(tc, sc, mc)
        # print('contraction tree result:', tc, sc, score)
        return tc, sc, mc

    def tree_complexity_new_order(self, tree_leaves, new_order):
        """
        Calculate the contraction complexity in a new contraction order of the subtree
        """
        tmp_tree, tcs, scs = [], [], []
        mcs = []
        current_branch = {}
        for m, n in new_order:
            if m not in current_branch.keys():
                left = tree_leaves[m]
            else:
                left = current_branch[m]
            if n not in current_branch.keys():
                right = tree_leaves[n]
            else:
                right = current_branch[n]
            parent = ContractionVertex(left.contain_tensors | right.contain_tensors, self.tn, left, right)
            tmp_tree.append(parent)
            current_branch[m] = parent
            # factor = min(log2(self.tn.max_bitstring), final_qubits_num(self.tn.num_fq, parent.contain_tensors))
            tc_step, sc_step = parent.tc, parent.sc
            scs.append(sc_step)
            tcs.append(tc_step)
            mcs.append(parent.mc)
        for leaf in tree_leaves:
            scs.append(leaf.sc)   
        tc = log10sumexp2(tcs)
        sc = max(scs)
        mc = log10sumexp2(mcs)
        return tc, sc, mc
    
    def copy(self):
        """
        return a copy of current contraction tree
        TODO:
        1. try to remove the deepcopy
        """
        # ctree = object.__new__(ContractionTree)
        # ctree.seed, ctree.rng, ctree.max_bitstring = self.seed, self.rng, self.max_bitstring
        # properties = ['tensor_bonds', 'bond_dims', 'slicing_bonds', 'bond_tensors', 'slicing_bond_tensors',
        #               'order', 'all_tensors', 'tree', 'num_fq']
        # if self.final_qubits is not None:
        #     properties.append('final_qubits')
        # else:
        #     ctree.final_qubits = None
        # for attr in properties:
        #     setattr(ctree, attr, deepcopy(getattr(self, attr)))
        
        # ctree = ContractionTree(deepcopy(self.tensor_bonds), deepcopy(self.bond_dims), deepcopy(self.order), self.seed, self.final_qubits, self.max_bitstring)
        ctree = deepcopy(self)
        return ctree


def determine_old_order(vertex, local_tree_leaves):
    """
    Given subroot and subtree, determine the order of it, only useful when the subtree size is 3
    """
    if vertex.left not in local_tree_leaves:
        branch = vertex.left
    elif vertex.right not in local_tree_leaves:
        branch = vertex.right
    else:
        print(vertex.left, vertex.right, local_tree_leaves)
        raise ValueError('something wrong with the local tree')
    first_contract = sorted((local_tree_leaves.index(branch.left), local_tree_leaves.index(branch.right)))
    if first_contract == [0, 2]:
        return [(0,2), (0,1)]
    elif first_contract == [0, 1]:
        return [(0,1), (0,2)]
    else:
        assert first_contract == [1, 2]
        return [(1,2), (0,1)]


class GreedyOrderFinderNew:
    def __init__(self, tensor_network:AbstractTensorNetwork) -> None:
        """
        Class of greedy order finder
        Parameters:
        -----------
        tensor_network: AbstractTensorNetwork class
            the underlying tensor network
        -----------
        """
        self.tn = tensor_network

    def _construct_pair_info(self):
        """
        Construct the pair contraction info
        """
        self.pair_info = {}.fromkeys(self.potential_contraction_pair)
        for pair in self.pair_info.keys():
            self.pair_info[pair] = self._update_pair_info(pair)

    def _update_pair_info(self, pair):
        """
        Update the pair contraction info
        """
        i, j = pair
        contracted_tensors = self.current_branch[i].contain_tensors | self.current_branch[j].contain_tensors
        all_bonds = self.current_branch[i].contain_bonds | self.current_branch[j].contain_bonds
        common_bonds = self.current_branch[i].contain_bonds & self.current_branch[i].contain_bonds
        contract_bonds = set([bond for bond in common_bonds if self.tn.bond_tensors[bond].issubset(contracted_tensors)])
        result_bonds = all_bonds - contract_bonds
        multiconfig_factor = min(self.tn.log2_max_bitstring, final_qubits_num(self.tn.num_fq, contracted_tensors))
        sc = log2_accum_dims(self.tn.bond_dims, result_bonds)
        sc += multiconfig_factor
        if 'min_dim' in self.strategy:
            value = sc
        elif 'max_reduce' in self.strategy:
            value = sc - (log2_accum_dims(self.tn.bond_dims, self.current_branch[i].contain_bonds) + log2_accum_dims(self.tn.bond_dims, self.current_branch[j].contain_bonds))
        else:
            value = 1.0
        return value

    def _pair_select(self, rng):
        """
        select a pair for contraction
        """
        min_value = min(self.pair_info.values())
        min_pairs = [pair for pair in self.pair_info.keys() if self.pair_info[pair] == min_value]
        pair = min_pairs[rng.choice(range(len(min_pairs)))]
        return pair
    
    def contract(self, pair):
        """
        Contract a pair
        """
        i, j = pair
        left = self.current_branch[i]
        right = self.current_branch[j]
        merged_contain_tensors = left.contain_tensors | right.contain_tensors
        parent = ContractionVertex(merged_contain_tensors, self.tn, left, right)
        self.tree[merged_contain_tensors] = parent
        self.current_branch[i] = parent
        
        pairs_add = []
        for neigh in self.tensor_neighbors[j]:
            pair_eliminate = (min(j, neigh), max(j, neigh))
            self.pair_info.pop(pair_eliminate)
            if neigh != i and neigh not in self.tensor_neighbors[i]:
                pairs_add.append((min(i, neigh), max(i, neigh)))
        pairs_add += [(min(i, m), max(i, m)) for m in self.tensor_neighbors[i] if m != j]
        pairs_add = set(pairs_add)

        self.tensor_neighbors[i] = self.tensor_neighbors[i] | self.tensor_neighbors[j]
        self.tensor_neighbors[i].discard(i)
        self.tensor_neighbors[i].discard(j)

        for tensor_id in self.tensor_neighbors[j]:
            if tensor_id != i:
                self.tensor_neighbors[tensor_id].discard(j)
                self.tensor_neighbors[tensor_id].add(i)

        for pair_update in pairs_add:
            self.pair_info[pair_update] = self._update_pair_info(pair_update)
        
        return parent.tc, parent.sc, parent.mc


    def greedy_order(self, seed):
        """
        Return a greedy order according to specific greedy strategy
        """
        tcs, scs, mcs, order = [], [np.log2(np.prod([self.tn.bond_dims[bond] for bond in self.tensor_bonds[i]])) for i in range(len(self.tensor_bonds))], [], []
        rng = np.random.RandomState(seed)
        uncontract = True
        while uncontract:
            if len(self.pair_info) > 0:
                pair = self._pair_select(rng)
                tc, sc, mc = self.contract(pair)
                order.append(pair)
                tcs.append(tc)
                scs.append(sc)
                mcs.append(mc)
            else:
                involved_nodes = set()
                for pair in order:
                    involved_nodes.add(pair[1])
                source_node = order[-1][0]
                uninvolved_nodes = set(list(range(len(self.tensor_bonds)))) - involved_nodes
                for node in uninvolved_nodes:
                    if node == source_node:
                        continue
                    pair = (source_node, node)
                    tc, sc, mc = self.contract(pair)
                    order.append(pair)
                    tcs.append(tc)
                    scs.append(sc)
                    mcs.append(mc)
                uncontract = False

        tc = log10sumexp2(tcs)
        sc = max(scs)
        mc = log10sumexp2(mcs)
        return order, tc, sc, mc

    def __call__(self, strategy='min_dim', seed=0, tree_leaves=None):
        """
        Call the class
        """
        self.tensor_neighbors = []
        if tree_leaves is None:
            self.tensor_bonds = self.tn.tensor_bonds
            bond_tensors = self.tn.bond_tensors
        else:
            self.tensor_bonds = [leaf.contain_bonds for leaf in tree_leaves]
            all_bonds = list(set().union(*self.tensor_bonds))
            bond_tensors = {bond: set() for bond in all_bonds} # determine tensors corresponding to each bond
            for i in range(len(self.tensor_bonds)):
                for j in self.tensor_bonds[i]:
                    bond_tensors[j].add(i)
        for i in range(len(self.tensor_bonds)):
            self.tensor_neighbors.append(set())
            for bond in self.tensor_bonds[i]:
                self.tensor_neighbors[i] = self.tensor_neighbors[i] | bond_tensors[bond]
            self.tensor_neighbors[i].discard(i)
        self.potential_contraction_pair = [(i, j) for i in range(len(self.tensor_bonds)) for j in self.tensor_neighbors[i] if i < j] 
        self.strategy = strategy
        self.current_branch = {i: tree_leaves[i] for i in range(len(self.tensor_bonds))}
        self.tree = {tree_leaves[i].contain_tensors : self.current_branch[i] for i in range(len(self.tensor_bonds))}
        self._construct_pair_info()
        order, tc, sc, mc = self.greedy_order(seed)
        return order, self.tree, tc, sc, mc
    

def simulate_annealing(tree, sc_target=-1, trials=10, iters=50, betas=np.linspace(0.1, 10, 100), slicing_repeat=4, start_seed=0):
    init_result = tree.tree_complexity()
    t0 = time.time()
    args = [(tree.copy(), sc_target, init_result, iters, betas, start_seed + i, slicing_repeat) for i in range(trials)]
    p = mp.Pool(trials)
    results = p.starmap(sa_trial, args)
    p.close()
    t1 = time.time()
    results_slicing = [(result[0][1] + len(result[1].tn.slicing_bonds) * log10(2), result[1]) for result in results]
    # print([r[0] for r in results_slicing])
    best_result, best_tree = sorted(results_slicing, key=lambda info:info[0])[0]
    # print(init_result)
    # print(t1 - t0)
    # print(best_result)
    # print(len(best_tree.tn.slicing_bonds))

    return best_tree.tree_to_order(), best_tree.tn.slicing_bonds

def sa_trial(tree, sc_target, init_result, iters, betas, seed, slicing_repeat=4):
    init_tc, init_sc, init_mc = init_result
    init_score = score_fn(init_tc, init_sc, init_mc)
    best_result = [(init_score, init_tc, init_sc, init_mc), tree.copy()]
    sub_root = tree.tree[tree.all_tensors]
    rng = np.random.RandomState(seed)
    for beta in betas:
        for iter in range(iters):
            # t0 = time.time()
            tree_update(sub_root, tree, 3, beta, init_sc, rng, sc_target=sc_target)
            tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
            result = (score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target), tc_tmp, sc_tmp, mc_tmp)
            if result[0] < best_result[0][0]:
                best_result = [result, tree.copy()]
            # t1 = time.time()
            # print(beta, iter, init_result[2], result, best_result[0][0], t1 - t0)
    
    result = best_result[1].tree_complexity()
    optimized_sc = result[1]
    # print('after optimization, results are', result)
    slicing_loop = 0
    while slicing_loop < slicing_repeat * abs(optimized_sc - sc_target) or best_result[0][2] > sc_target:
        tree = best_result[1]
        # t0 = time.time()
        current_tc, current_sc, current_mc = tree.tree_complexity()
        # print('slicing init', current_score, current_tc, current_sc)
        if current_sc > sc_target:# or rng.rand() > 0.5:
            scores_slicing = []
            for bond in tree.select_slicing_bonds():
                tc_slicing, sc_slicing, mc_slicing = tree.slicing_tree_complexity_new(bond)
                scores_slicing.append((bond, score_fn(tc_slicing, sc_slicing, mc_slicing, sc_target), tc_slicing, sc_slicing, mc_slicing))
            slicing_bond = sorted(scores_slicing, key=lambda info:info[1])[0][0]
            # print(slicing_bond, scores_slicing)
            tree.slicing(slicing_bond)
        elif len(tree.tn.slicing_bonds) > 0:
            bond_add = rng.choice(list(tree.tn.slicing_bonds.keys()))
            # scores_adding = []
            # for bond in tree.tn.slicing_bonds.keys():
            #     scores_adding.append((bond, tree.add_bond_complexity(bond)[0]))
            # idx = rng.choice(max(len(scores_adding) // 4, 2))
            # bond_add = sorted(scores_adding, key=lambda info:info[1])[idx][0]
            # print(bond_add, scores_adding)
            tree.add_bond(bond_add)
        tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
        result = (score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target), tc_tmp, sc_tmp, mc_tmp)
        best_result = (result, tree.copy())
        # tt = time.time() - t0
        # if current_sc > sc_target:
        #     print(f'slicing time: {tt} slicing bond {slicing_bond} after slicing {result}')
        # else:
        #     print(f'add bond time: {tt} add bond {bond_add} after adding {result}' )
        for beta in betas[-10:]:
            for iter in range(iters):
                sub_root = tree.tree[tree.all_tensors]
                # t0 = time.time()
                tree_update(sub_root, tree, 3, beta, sc_target, rng, check_detail=False, sc_target=sc_target)
                tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
                result = (score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target), tc_tmp, sc_tmp, mc_tmp)
                if result[0] < best_result[0][0]:
                    best_result = (result, tree.copy())
                # t1 = time.time()
                # print(slicing_loop, beta, iter, sc_target, result, best_result[0][0], t1-t0, len(tree.tn.slicing_bonds), list(tree.tn.slicing_bonds.keys()))
        slicing_loop += 1
    return best_result


def tree_update(vertex, tree, size, beta, initial_sc, rng, check_detail=False, sc_target=30.0):
    """
    Local update of the contraction tree in a recursive way.
    For each step, get the size 3 subtree of current contraction vertex and find out the possible
    alternative 2 other contraction orders to update and randomly choose one, the update probability
    is calculated according to their score ratio
    """
    local_tree_leaves, local_tree = tree.spanning_tree(vertex, size)
    if len(local_tree_leaves) > 2:
        tc_tree, sc_tree, mc_tree = tree.tree_complexity(local_tree, vertex)
        reference_score = score_fn(tc_tree, sc_tree, mc_tree, sc_target)
        # reference_score, tc_tree, sc_tree = tree.tree_complexity(local_tree, vertex)
        # if sc_tree <= initial_sc:
        #     reference_score = tc_tree
        order_old = determine_old_order(vertex, local_tree_leaves)
        order_pool = [[(0,2),(0,1)], [(0,1),(0,2)], [(1,2),(0,1)]]

        if not check_detail:
            order_pool.remove(order_old)
            order_new = order_pool[rng.choice(2)]
            tc_new, sc_new, mc_new = tree.tree_complexity_new_order(local_tree_leaves, order_new)
            score_new = score_fn(tc_new, sc_new, mc_new, sc_target)
            # if sc_new <= initial_sc:
            #     score_new = tc_new
            # else:
            #     score_new = score_fn(tc_new, sc_new, mc_new)
        else:
            results = []
            for order_new in order_pool:
                tc_new, sc_new, mc_new = tree.tree_complexity_new_order(local_tree_leaves, order_new)
                if sc_new <= initial_sc:
                    score_new = tc_new
                else:
                    score_new = score_fn(tc_new, sc_new, mc_new, sc_target)
                results.append((score_new, order_new, tc_new, sc_new))
            score_new, order_new, tc_new, sc_new = sorted(results, key=lambda r:r[0])[0]
            print(order_old, tc_tree, sc_tree)
            for v in local_tree:
                print(local_tree.index(v), v.contain_tensors, v.contain_bonds, v.sc, v.tc)
            print(results)
            print('-'*50)

        # print('-'*20)
        # print(fqs_local_tree)
        # print(score_new, reference_score, tc_tree, sc_tree, order_old, np.equal(score_new, reference_score), np.exp(-beta * (score_new-reference_score)), results)
        # for i in range(len(tn_local_tree.neighbors)):
        #     print(tn_local_tree.neighbors[i], [log2(tn_local_tree.shapes[i][j]) for j in range(len(tn_local_tree.shapes[i]))])
        if rng.rand() < np.exp(-beta * (score_new-reference_score)): # score_new < reference_score: #
            # original_score, original_tc, original_sc = tree.tree_complexity()
            tree.apply_order(order_new, local_tree_leaves, local_tree, vertex)
            # tree.check_contractiontree()
            # print('*'*5, 'change order', tree.tree_complexity(), order_new, order_old)
            # after_score, after_tc, after_sc = tree.tree_complexity()
            # print(original_score, original_tc, original_sc)
            # print(after_score, after_tc, after_sc)
            # print(score_new, tc_new, sc_new, reference_score, tc_tree, sc_tree)
            # if after_score > original_score:
            #     print('-'*20)
            #     print(original_score, original_tc, original_sc)
            #     print(after_score, after_tc, after_sc)
            #     print(score_new, tc_new, sc_new, reference_score, tc_tree, sc_tree)
            #     print(np.exp(-beta * (score_new-reference_score)))
            #     # print(results[0][2:], tc_tn, sc_tn)
            #     print('-'*20)

        for next_vertex in [vertex.left, vertex.right]:
            tree_update(next_vertex, tree, size, beta, initial_sc, rng, check_detail, sc_target)


def find_order(tensor_bonds, bond_dims, seed=0, final_qubits=[], max_bitstrings=1, **vargs):
    """
    Function wrapper for finding the contraction order of a given tensor network
    """
    from .greedy import GreedyOrderFinder
    tensor_network = AbstractTensorNetwork(
        deepcopy(tensor_bonds), 
        deepcopy(bond_dims),
        final_qubits,
        max_bitstrings)
    greedy_order = GreedyOrderFinder(tensor_network)
    order, tc, sc = greedy_order('min_dim', seed)
    ctree = ContractionTree(deepcopy(tensor_network), order, seed)

    order_slicing, slicing_bonds = simulate_annealing(ctree, **vargs)

    for bond in slicing_bonds:
        tensor_network.slicing(bond)

    ctree_new = ContractionTree(tensor_network, order_slicing, seed)

    return order_slicing, slicing_bonds, ctree_new


# functions below are on developing
def random_tree_update(tree:ContractionTree, iters, size, beta, rng:np.random.mtrand.RandomState, sc_target=30):
    for _ in range(iters):
        vertex = tree.tree[rng.choice(list(tree.tree.keys()))]
        local_tree_leaves, local_tree = tree.spanning_tree(vertex, size)
        if len(local_tree_leaves) > 2:
            tc_tree, sc_tree, mc_tree = tree.tree_complexity(local_tree, vertex)
            reference_score = score_fn(tc_tree, sc_tree, mc_tree, sc_target)
            order_old = determine_old_order(vertex, local_tree_leaves)
            order_pool = [[(0,2),(0,1)], [(0,1),(0,2)], [(1,2),(0,1)]]
            order_pool.remove(order_old)
            order_new = order_pool[rng.choice(2)]
            tc_new, sc_new, mc_new = tree.tree_complexity_new_order(local_tree_leaves, order_new)
            score_new = score_fn(tc_new, sc_new, mc_new, sc_target)
            if rng.rand() < np.exp(-beta * (score_new-reference_score)):
                tree.apply_order(order_new, local_tree_leaves, local_tree, vertex)

def random_tree_sa(tree:ContractionTree, sc_target=-1, iters=100, betas=np.linspace(0.1, 10, 100), seed=0):
    init_tc, init_sc, init_mc = tree.tree_complexity()
    init_score = score_fn(init_tc, init_sc, init_mc, sc_target)

    best_result = [(init_score, init_tc, init_sc, init_mc), tree.copy()]
    rng = np.random.RandomState(seed)
    for beta in betas:
        t0 = time.time()
        random_tree_update(tree, iters, 3, beta, rng, sc_target)
        tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
        result = (score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target), tc_tmp, sc_tmp, mc_tmp)
        if result[0] < best_result[0][0]:
            best_result = [result, tree.copy()]
        t1 = time.time()
        print(seed, beta, result, best_result[0][0], t1 - t0)
        
    result = best_result[1].tree_complexity()
    print('after optimization, results are', result)
    optimized_sc = result[1]
    slicing_loop = 0
    while slicing_loop < 4 * abs(optimized_sc - sc_target) or best_result[0][2] > sc_target:
        tree = best_result[1]
        t0 = time.time()
        current_tc, current_sc, current_mc = tree.tree_complexity()
        current_score = score_fn(current_tc, current_sc, current_mc, sc_target)
        print('slicing init', current_score, current_tc, current_sc, current_mc)
        if current_sc > sc_target:
            scores_slicing = []
            for bond in tree.tn.bond_dims.keys():
                tc_slicing, sc_slicing, mc_slicing = tree.slicing_tree_complexity(bond)
                scores_slicing.append((bond, score_fn(tc_slicing, sc_slicing, mc_slicing, sc_target), tc_slicing, sc_slicing, mc_slicing))
            slicing_bond = sorted(scores_slicing, key=lambda info:info[1])[0][0]
            print(slicing_bond, scores_slicing)
            tree.slicing(slicing_bond)
        else:
            # bond_add = rng.choice(list(trees[i].tn.slicing_bonds.keys()))
            scores_adding = []
            for bond in tree.tn.slicing_bonds.keys():
                tc_add_bond, sc_add_bond, mc_add_bond = tree.add_bond_complexity(bond)
                scores_adding.append((bond, score_fn(tc_add_bond, sc_add_bond, mc_add_bond, sc_target), tc_add_bond, sc_add_bond, mc_add_bond))
            idx = rng.choice(max(len(scores_adding) // 4, 2))
            bond_add = sorted(scores_adding, key=lambda info:info[1])[idx][0]
            print(bond_add, scores_adding)
            tree.add_bond(bond_add)
        tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
        result = (score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target), tc_tmp, sc_tmp, mc_tmp)
        best_result = (result, tree.copy())
        tt = time.time() - t0
        if current_sc > sc_target:
            print(f'slicing time: {tt} slicing bond {slicing_bond} after slicing {result}')
        else:
            print(f'add bond time: {tt} add bond {bond_add} after adding {result}' )
        for beta in betas[23:]:# np.linspace(12.0, 20.0, 9):
            t0 = time.time()
            random_tree_update(tree, iters, 3, beta, rng, sc_target)
            tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
            result = (score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target), tc_tmp, sc_tmp, mc_tmp)
            if result[0] < best_result[0][0]:
                best_result = (result, tree.copy())
            t1 = time.time()
            print(seed, slicing_loop, beta, result, best_result[0][0], t1-t0, len(tree.tn.slicing_bonds), list(tree.tn.slicing_bonds.keys()))
        slicing_loop += 1
    tree = best_result[1]
    score = tree.tree_complexity()
    print(init_score, init_tc, init_sc)
    final_tc, final_sc, final_mc = tree.tree_complexity()
    print(f'results: sliced tc {final_tc} sliced sc {final_sc} sliced mc {final_mc} sliced bonds number {len(tree.tn.slicing_bonds)} overall tc {final_tc + log10(2**len(tree.tn.slicing_bonds))}')

    return tree.tree_to_order(), tree.tn.slicing_bonds


def tree_optimize(tree, sc_target=-1, trials=10, iters=50, subtree_size=3, slicing_repeat=4, start_seed=0):
    init_result = tree.tree_complexity()
    t0 = time.time()
    args = [(tree.copy(), sc_target, init_result, iters, subtree_size, start_seed + i, slicing_repeat) for i in range(trials)]
    p = mp.Pool(trials)
    results = p.starmap(tree_optimize_subroutine, args)
    p.close()
    t1 = time.time()
    results_slicing = [(result[0][1] + len(result[1].tn.slicing_bonds) * log10(2), result[1]) for result in results]
    print(results_slicing)
    best_result, best_tree = sorted(results_slicing, key=lambda info:info[0])[0]
    # print(init_result)
    # print(t1 - t0)
    print(best_result)
    print(len(best_tree.tn.slicing_bonds))

    return best_tree.tree_to_order(), best_tree.tn.slicing_bonds

def tree_optimize_subroutine(tree, sc_target, init_result, iters, subtree_size,seed, slicing_repeat=4):
    init_tc, init_sc, init_mc = init_result
    init_score = score_fn(init_tc, init_sc, init_mc)
    best_result = [(init_score, init_tc, init_sc, init_mc), tree.copy()]
    sub_root = tree.tree[tree.all_tensors]
    rng = np.random.RandomState(seed)
    for iter in range(iters):
        t0 = time.time()
        vertex = tree.tree[rng.choice(list(tree.tree.keys())[-3:])]
        subtree_update(vertex, tree, subtree_size, rng, sc_target=sc_target)
        tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
        result = (score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target), tc_tmp, sc_tmp, mc_tmp)
        if result[0] < best_result[0][0]:
            best_result = [result, tree.copy()]
        t1 = time.time()
        print(iter, init_result[2], result, best_result[0][0], t1 - t0)
    
    result = best_result[1].tree_complexity()
    optimized_sc = result[1]
    print('after optimization, results are', result)
    slicing_loop = 0
    # while slicing_loop < slicing_repeat * abs(optimized_sc - sc_target) or best_result[0][2] > sc_target:
    #     tree = best_result[1]
    #     t0 = time.time()
    #     current_tc, current_sc, current_mc = tree.tree_complexity()
    #     print('slicing init', current_tc, current_sc)
    #     if current_sc > sc_target:# or rng.rand() > 0.5:
    #         scores_slicing = []
    #         for bond in tree.select_slicing_bonds():
    #             tc_slicing, sc_slicing, mc_slicing = tree.slicing_tree_complexity_new(bond)
    #             scores_slicing.append((bond, score_fn(tc_slicing, sc_slicing, mc_slicing, sc_target), tc_slicing, sc_slicing, mc_slicing))
    #         slicing_bond = sorted(scores_slicing, key=lambda info:info[1])[0][0]
    #         print(slicing_bond, scores_slicing)
    #         tree.slicing(slicing_bond)
    #     elif len(tree.tn.slicing_bonds) > 0:
    #         bond_add = rng.choice(list(tree.tn.slicing_bonds.keys()))
    #         # scores_adding = []
    #         # for bond in tree.tn.slicing_bonds.keys():
    #         #     scores_adding.append((bond, tree.add_bond_complexity(bond)[0]))
    #         # idx = rng.choice(max(len(scores_adding) // 4, 2))
    #         # bond_add = sorted(scores_adding, key=lambda info:info[1])[idx][0]
    #         # print(bond_add, scores_adding)
    #         tree.add_bond(bond_add)
    #     tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
    #     result = (score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target), tc_tmp, sc_tmp, mc_tmp)
    #     best_result = (result, tree.copy())
    #     tt = time.time() - t0
    #     if current_sc > sc_target:
    #         print(f'slicing time: {tt} slicing bond {slicing_bond} after slicing {result}')
    #     else:
    #         print(f'add bond time: {tt} add bond {bond_add} after adding {result}' )
    #     for iter in range(iters):
    #         vertex = tree.tree[rng.choice(list(tree.tree.keys()))]
    #         t0 = time.time()
    #         subtree_update(vertex, tree, subtree_size, rng, sc_target=sc_target)
    #         tc_tmp, sc_tmp, mc_tmp = tree.tree_complexity()
    #         result = (score_fn(tc_tmp, sc_tmp, mc_tmp, sc_target), tc_tmp, sc_tmp, mc_tmp)
    #         if result[0] < best_result[0][0]:
    #             best_result = (result, tree.copy())
    #         t1 = time.time()
    #         print(slicing_loop, iter, sc_target, result, best_result[0][0], t1-t0, len(tree.tn.slicing_bonds), list(tree.tn.slicing_bonds.keys()))
    #     slicing_loop += 1
    return best_result

def subtree_update(vertex:ContractionVertex, tree:ContractionTree, size, rng, sc_target):
    """
    Single step of local search in contraction tree
    """
    local_tree_leaves, local_tree = tree.spanning_tree(vertex, size)
    if len(local_tree_leaves) > 2:
        tc_tree, sc_tree, mc_tree = tree.tree_complexity(local_tree, vertex)
        score_reference = score_fn(tc_tree, sc_tree, mc_tree, sc_target)
        # for leaf in local_tree_leaves:
        #     print(leaf.contain_tensors)
        #     print(leaf.contain_bonds)
        # print(local_tree)
        # sub_tensor_bonds = [leaf.contain_bonds for leaf in local_tree_leaves]
        # sub_tensor_lists = [leaf.contain_tensors for leaf in local_tree_leaves]
        # sub_tn = tree.tn.sub_tensor_network(sub_tensor_lists, sub_tensor_bonds)
        # print(sub_tn.tensor_bonds, sub_tn.num_fq)
        greedy_finder = GreedyOrderFinderNew(tree.tn)
        result_new = []
        for i in range(20):
            order_new, tree_new, tc_new, sc_new, mc_new = greedy_finder(
                'max_reduce', 
                rng.randint(len(local_tree_leaves)), 
                local_tree_leaves
            )
            score_new = score_fn(tc_new, sc_new, mc_new, sc_target)
            result_new.append((order_new, score_new))
        print([result[1] for result in result_new])
        order_new, score_new = sorted(result_new, key=lambda r:r[1])[0]
        # print(order_new, tree_new)
        print('scores', score_reference, score_new)
        if np.exp(0.1 * (score_new - score_reference)) < rng.rand():
            print('updated!', vertex.contain_tensors)
            # print('-'*20)
            # print('leaves:')
            # for leaf in local_tree_leaves:
            #     print(leaf.contain_tensors)
            #     print(leaf.contain_bonds)
            # print('tree:')
            # for v in local_tree:
            #     print(v.contain_tensors, v.contain_bonds)
            print(f'before all tree complexity {tree.tree_complexity()}')
            tree.apply_order(order_new, local_tree_leaves, local_tree, vertex)
            # print(order_new)
            print(f'before tc {tc_tree} sc {sc_tree} mc {mc_tree} score {score_reference}') 
            print(f'after tc {tc_new} sc {sc_new} mc {mc_new} score {score_new}')
            print(f'after all tree complexity {tree.tree_complexity()}')

        # for next_vertex in local_tree_leaves:
        #     subtree_update(next_vertex, tree, size, rng, sc_target)
        