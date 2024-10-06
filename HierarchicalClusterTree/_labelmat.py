from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from functools import partial
from ._cluster_tree import ClusterTreeNode
from ._utils import calculate_centroid
from typing import Callable, Dict

# Convert from label matrix to tree

def recurse_build_tree(labelmat: np.ndarray, depth: int, parent: ClusterTreeNode, 
                       centroid_func: Callable[[set[int]], NDArray[np.float64]]) -> None:
    """
    Recursively builds the cluster tree from the label matrix.

    Args:
        labelmat (np.ndarray): The label matrix where each row represents a sample and each column a level.
        depth (int): The current depth in the tree.
        parent (ClusterTreeNode): The parent node in the tree.
        centroid_func (Callable[[set[int]], NDArray[np.float64]]): Function to calculate the centroid of a cluster.
    """
    level = labelmat[:, depth]
    parent_subset = level[list(parent.members)]

    for label in np.unique(parent_subset):
        members = set(np.where(level == label)[0])
        centroid = centroid_func(members)
        node = ClusterTreeNode(str(label), members, centroid)
        parent.add_child(node)
        
        if depth + 1 < labelmat.shape[1]:
            recurse_build_tree(labelmat, depth + 1, node, centroid_func)


def labelmat_to_tree(labelmat: np.ndarray, embedding: NDArray[np.float64]) -> ClusterTreeNode:
    """
    Converts a label matrix to a cluster tree.

    Args:
        labelmat (np.ndarray): Label matrix where each row represents a sample.
        embedding (NDArray[np.float64]): Embedding matrix representing data points in a high-dimensional space.

    Returns:
        ClusterTreeNode: The root node of the generated cluster tree.
    """
    centroid_func = partial(calculate_centroid, embedding=embedding)
    members = set(np.arange(labelmat.shape[0]))
    centroid = centroid_func(members)
    
    # Create the root node
    tree = ClusterTreeNode('root', members, centroid)
    
    # Build the tree recursively
    recurse_build_tree(labelmat, 0, tree, centroid_func)
    
    return tree

# Convert from tree to label matrix

def recurse_build_labeldict(cluster: ClusterTreeNode, labeldict: Dict[int, NDArray]) -> None:
    """
    Recursively builds a dictionary mapping each depth level to a label matrix.

    Args:
        cluster (ClusterTreeNode): Current cluster node.
        labeldict (Dict[int, NDArray[object]]): Dictionary where keys are depth levels and values are arrays of cluster names.
    """
    labeldict[cluster.depth][list(cluster.members)] = cluster.name

    for child in cluster.children:
        recurse_build_labeldict(child, labeldict)


def tree_to_labelmat(tree: ClusterTreeNode) -> np.ndarray:
    """
    Converts a cluster tree back to a label matrix.

    Args:
        tree (ClusterTreeNode): The root of the cluster tree.

    Returns:
        np.ndarray: Label matrix where each row represents a sample and each column a tree level.
    """
    n_members = len(tree.members)
    labeldict = defaultdict(lambda: np.empty(shape=n_members, dtype=object))
    
    # Recursively populate the labeldict from the tree
    recurse_build_labeldict(tree, labeldict)
    
    # Convert the dictionary of levels to a label matrix
    labelmat = np.vstack([level.reshape(1, -1) for level in labeldict.values()]).T
    
    return labelmat
