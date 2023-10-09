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
