"""
Basic tests for the tree module. Tests if we can build trees, and recover the original data.
"""

import numpy as np

from imagetree.tree import Node, QuadTree, TreeConfiguration


def test_tree_build():
    config = TreeConfiguration(
        base_grid_size=64,
        refinement_levels=3,
        dtype=np.float32,
    )

    tree = QuadTree(config)
    tree.initialize_from_array(np.zeros((256, 256), dtype=np.float32))
    tree.walk_tree_and_populate()

    assert tree.initialized


def test_tree_recover():
    config = TreeConfiguration(
        base_grid_size=64,
        refinement_levels=2,
        dtype=np.float64,
    )

    tree = QuadTree(config)
    base_array = np.random.rand(256, 256)
    tree.initialize_from_array(base_array)
    tree.walk_tree_and_populate()

    recovered = tree.nodes[0][0].children[0][0].children[0][0].data

    assert recovered.shape == (64, 64)
    assert recovered.dtype == np.float64
    assert (recovered == base_array[0:64, 0:64]).all()
