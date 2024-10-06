import numpy as np
from numpy.typing import NDArray
from functools import partial
from typing import Set, Optional


class ClusterTreeNode:
    def __init__(self, name: str, members: Set[int], centroid: NDArray[np.float64]) -> None:
        """
        Initialize a ClusterTreeNode instance.

        Args:
            name (str): Name of the cluster node.
            members (Set[int]): Set of member indices belonging to this cluster.
            centroid (NDArray[np.float64]): Centroid of the cluster.
        """
        self.name = name
        self.members = members
        self.centroid = centroid
        self.parent: Optional[ClusterTreeNode] = None
        self.children: Set['ClusterTreeNode'] = set()
        self.depth: int = 0

    def add_child(self, child: 'ClusterTreeNode') -> None:
        """
        Adds a child node to the current cluster node.

        Args:
            child (ClusterTreeNode): Child node to be added.

        Raises:
            AssertionError: If the child members are not a subset of the parent members.
        """
        assert child.members.issubset(self.members), 'ClusterTree is not hierarchical.'
        child.parent = self
        child.depth = self.depth + 1
        self.children.add(child)

    def merge_cluster(self, other: 'ClusterTreeNode') -> None:
        """
        Merges another ClusterTreeNode into the current node.

        Args:
            other (ClusterTreeNode): Another node to merge with this one.

        Raises:
            AssertionError: If the two clusters being merged are not on the same depth level.
        """
        assert other.depth == self.depth, 'Only clusters on the same level can be merged.'

        self.name = self.name + '_' + other.name
        self.members.update(other.members)
        for child in other.children:
            child.parent = self
            self.children.add(child)

        _n_self = len(self.members)
        _n_other = len(other.members)
        self.centroid = ((_n_self * self.centroid) + (_n_other * other.centroid)) / (_n_self + _n_other)

        other.parent.children.remove(other)
        other.members.clear()
        other.children.clear()
        other.parent = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the ClusterTreeNode.
        """
        parent_name = self.parent.name if self.parent else "None"
        children_names = ', '.join([child.name for child in self.children]) or "None"
        
        return (f"ClusterTreeNode(name='{self.name}', level={self.depth}, "
                f"num_members={len(self.members)}, parent='{parent_name}', "
                f"children=[{children_names}])")