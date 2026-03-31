from .tensor_network import AbstractTensorNetwork
from .utils import final_qubits_num, log2_accum_cached, log2sumexp2, log10sumexp2, log10sumexp2_pair
from math import log2, ceil
from copy import deepcopy
import numpy as np
import sys
from traceback import print_exc
from collections import deque
from collections import Counter
from types import SimpleNamespace


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
        self.rep_tensor = -1
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
            self.tc, self.sc, self.multiconfig_factor, self.contain_bonds, self.mc, self.contract_bonds, self.all_bonds = \
                get_tc_sc_contraction(tn, left, right)
        else:
            self.tc, self.sc, self.multiconfig_factor, self.contain_bonds, self.mc = \
                get_tc_sc_inner(tn, contain_tensors)
            self.all_bonds = self.contain_bonds
            self.contract_bonds = set()

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
    other_bonds = set().union(*[tensor_bonds[tensor_id] for tensor_id in tensor_bonds.keys() if tensor_id not in part])
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
    return 0.0, log2_accum_cached(tn.log2_bond_dims, bonds1) + multiconfig_factor, multiconfig_factor, bonds1, 0.0


def get_tc_sc_contraction(tn:AbstractTensorNetwork, left:ContractionVertex, right:ContractionVertex):
    """
    Calculating complexity of contracting tensors in left and right
    return tc, sc, resulting_bonds and mc
    """
    _, tc, sc, multiconfig_factor, result_bonds, mc, contract_bonds, all_bonds = merge_vertex_info(tn, left, right)
    return tc, sc, multiconfig_factor, result_bonds, mc, contract_bonds, all_bonds


def merge_vertex_info(tn:AbstractTensorNetwork, left, right):
    contracted_tensors = left.contain_tensors | right.contain_tensors
    all_bonds = left.contain_bonds | right.contain_bonds
    common_bonds = left.contain_bonds & right.contain_bonds
    contract_bonds = {bond for bond in common_bonds if tn.bond_tensors[bond].issubset(contracted_tensors)}
    result_bonds = all_bonds - contract_bonds
    combined_multiconfig_factor = left.multiconfig_factor + right.multiconfig_factor
    multiconfig_factor = min(tn.log2_max_bitstring, combined_multiconfig_factor)

    all_bonds_cost = log2_accum_cached(tn.log2_bond_dims, all_bonds)
    tc = all_bonds_cost if contract_bonds else all_bonds_cost - 1
    sc = log2_accum_cached(tn.log2_bond_dims, result_bonds)
    tc += multiconfig_factor
    sc += multiconfig_factor
    if combined_multiconfig_factor > tn.log2_max_bitstring:
        mc = log2sumexp2([
            left.sc - left.multiconfig_factor + multiconfig_factor,
            right.sc - right.multiconfig_factor + multiconfig_factor,
            sc
        ])
    else:
        mc = log2sumexp2([left.sc, right.sc, sc])
    return contracted_tensors, tc, sc, multiconfig_factor, result_bonds, mc, contract_bonds, all_bonds


def local_tree_score(branch, root, leaves):
    tc = log10sumexp2_pair(branch.tc, root.tc)
    sc = max(*(leaf.sc for leaf in leaves), branch.sc, root.sc)
    mc = log10sumexp2_pair(branch.mc, root.mc)
    return tc, sc, mc


def candidate_local_tree_score(tn, first, second, third):
    mid_tensors, mid_tc, mid_sc, mid_multiconfig_factor, mid_bonds, mid_mc, mid_contract_bonds, mid_all_bonds = \
        merge_vertex_info(tn, first, second)
    mid = SimpleNamespace(
        contain_tensors=mid_tensors,
        contain_bonds=mid_bonds,
        multiconfig_factor=mid_multiconfig_factor,
        sc=mid_sc,
        all_bonds=mid_all_bonds,
        contract_bonds=mid_contract_bonds
    )
    _, root_tc, root_sc, _, _, root_mc, _, _ = merge_vertex_info(tn, mid, third)
    tc = log10sumexp2_pair(mid_tc, root_tc)
    sc = max(first.sc, second.sc, third.sc, mid_sc, root_sc)
    mc = log10sumexp2_pair(mid_mc, root_mc)
    return tc, sc, mc


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
        self.all_tensors = frozenset(self.tn.tensor_bonds.keys())
        self.tree = self.construct_contractiontree(order)

        # self.seed = seed
        # self.rng = np.random.RandomState(seed)

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
        queue = deque([self.tree[self.all_tensors]])

        while len(queue):
            vertex = queue.popleft()
            if vertex.left and vertex.right:
                next_vertices = [vertex.left, vertex.right]
                vertex_list += next_vertices
                queue.extend(next_vertices)
        
        if sequence == 'leaves-root':
            vertex_list.reverse()
        else:
            assert sequence == 'root-leaves'

        return vertex_list

    def select_local_update(self, root):
        left, right = root.left, root.right
        if not (left and right):
            return None
        if left.left and left.right:
            return "left", left, right, left.left, left.right
        if right.left and right.right:
            return "right", right, left, right.left, right.right
        return None

    def apply_local_update(self, root, side, branch, outer, first, second, choice):
        old_key = branch.contain_tensors
        self.tree.pop(old_key)
        if side == "left":
            if choice == 0:
                branch.update_info(first.contain_tensors | outer.contain_tensors, self.tn, first, outer)
                root.right = second
            else:
                branch.update_info(second.contain_tensors | outer.contain_tensors, self.tn, second, outer)
                root.right = first
            root.left = branch
        else:
            if choice == 0:
                branch.update_info(outer.contain_tensors | second.contain_tensors, self.tn, outer, second)
                root.left = first
            else:
                branch.update_info(outer.contain_tensors | first.contain_tensors, self.tn, outer, first)
                root.left = second
            root.right = branch
        self.tree[branch.contain_tensors] = branch
        root.update_info(root.contain_tensors, self.tn, root.left, root.right)

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

    def select_slicing_bond_heuristic(self, target_sc=None):
        """
        Pick a slicing bond from the peak-space intermediates using a cheap frequency heuristic.
        """
        if target_sc is None:
            _, target_sc, _ = self.tree_complexity()
        bond_counter = Counter()
        for vertex in self.tree.values():
            if vertex.sc == target_sc:
                bond_counter.update(vertex.contain_bonds)
        assert bond_counter
        return max(
            bond_counter,
            key=lambda bond: (bond_counter[bond], self.tn.log2_bond_dims[bond], str(bond))
        )

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
                # print(vertex.contain_tensors, vertex.contain_bonds, vertex.tc, vertex.sc)
                # if vertex.left and vertex.right:
                #     print(vertex.left.contain_tensors, vertex.left.contain_bonds)
                #     print(vertex.right.contain_tensors, vertex.right.contain_bonds)
                # if bond in vertex.contract_bonds:
                #     print('*'*20)
                vertex.update_info(vertex.contain_tensors, self.tn, vertex.left, vertex.right)
                # print(vertex.tc, vertex.sc)
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

    def mark_rep_tensor(self):
        vertex_list = self.vertex_list('leaves-root')
        for vertex in vertex_list:
            if vertex.left and vertex.right:
                if vertex.left.sc > vertex.right.sc:
                    vertex.rep_tensor = vertex.left.rep_tensor
                else:
                    vertex.rep_tensor = vertex.right.rep_tensor
            else:
                vertex.rep_tensor = min(vertex.contain_tensors)
    
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
        self.mark_rep_tensor()
        stack = [self.tree[self.all_tensors]]
        order = []
        while len(stack):
            vertex = stack.pop()
            if vertex.left and vertex.right:
                # rep_tensors = [min(vertex.left.contain_tensors), min(vertex.right.contain_tensors)]
                # order.append((min(rep_tensors), max(rep_tensors)))
                if vertex.rep_tensor == vertex.left.rep_tensor:
                    order.append((vertex.left.rep_tensor, vertex.right.rep_tensor))
                elif vertex.rep_tensor == vertex.right.rep_tensor:
                    order.append((vertex.right.rep_tensor, vertex.left.rep_tensor))
                else:
                    raise ValueError('Incorrect rep tensor mark process.')
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
