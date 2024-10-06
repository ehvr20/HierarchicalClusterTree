import numpy as np
from itertools import combinations
from typing import Callable
from ._cluster_tree import ClusterTreeNode


def prune_tree(tree: ClusterTreeNode, cond_func: Callable) -> None:
    """
    Prunes a cluster tree based on a condition function.

    Args:
        tree (ClusterTreeNode): The root of the cluster tree.
        cond_func (Callable): A function that evaluates whether two clusters should be merged.
    """
    if not tree.children:
        return

    previous_children = set()

    while previous_children != tree.children:
        previous_children = tree.children
        comparisons = combinations(tree.children, 2)
        comparisons = sorted(comparisons, key=lambda x: np.linalg.norm(x[0].centroid - x[1].centroid))

        for clusterA, clusterB in comparisons:
            if cond_func(clusterA, clusterB):
                continue
            else:
                clusterA.merge_cluster(clusterB)
                break

    for child in tree.children:
        prune_tree(child, cond_func)
