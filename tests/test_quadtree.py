"""
Basic tests for the tree module. Tests if we can build trees, and recover the original data.
"""

import numpy as np
import pytest

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


def test_illegal_tree_configuration():
    with pytest.raises(ValueError):
        config = TreeConfiguration(
            base_grid_size=31,
            refinement_levels=222,
            dtype=np.float32,
        )


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

    recovered = tree.nodes[-1][-1].children[-1][-1].children[-1][-1].data

    assert recovered.shape == (64, 64)
    assert recovered.dtype == np.float64
    assert (recovered == base_array[-64:, -64:]).all()


def test_tree_recover_arbritary():
    config = TreeConfiguration(
        base_grid_size=64,
        refinement_levels=2,
        dtype=np.float64,
    )

    tree = QuadTree(config)
    base_array = np.random.rand(256, 256)
    tree.initialize_from_array(base_array)
    tree.walk_tree_and_populate()

    # These are prime on purpose...
    x = 73
    y = 67
    height = 29
    width = 31

    recovered = tree.extract_pixels(x=x, y=y, height=height, width=width)

    assert recovered.shape == (width, height)
    assert recovered.dtype == np.float64
    assert (recovered == base_array[x : x + width, y : y + height]).all()

    # Now try to recover the entire array.
    recovered = tree.extract_pixels(x=0, y=0, height=256, width=256)

    assert recovered.shape == (256, 256)
    assert recovered.dtype == np.float64
    assert (recovered == base_array).all()

    # We should behave well and return something sensible if they ask for a region
    # larger than the entire grid, or for instance one that overlaps the edges.

    x = -5
    y = -7
    height = 317
    width = 307

    recovered = tree.extract_pixels(x=x, y=y, height=height, width=width)

    assert recovered.shape == (width, height)
    assert recovered.dtype == np.float64
    assert (recovered[abs(x) : abs(x) + 256, abs(y) : abs(y) + 256] == base_array).all()


def test_non_square():
    config = TreeConfiguration(
        base_grid_size=64,
        refinement_levels=2,
        dtype=np.float64,
    )

    x_size = 1427
    y_size = 829

    tree = QuadTree(config)
    base_array = np.array([np.arange(x_size)] * y_size).T
    tree.initialize_from_array(base_array)
    tree.walk_tree_and_populate()

    recovered = tree.extract_pixels(x=0, y=0, height=y_size, width=x_size)

    import matplotlib.pyplot as plt

    from imagetree.visualisation import plot_grid

    # plt.imshow(recovered, cmap="Greys", vmin=0.0, vmax=x_size)
    # plt.savefig('recovered.png')
    # plt.clf()
    # plot_grid(grid=tree.nodes, fig=None, ax=None, vmin=0.0, vmax=x_size)
    # plt.xlim(0, x_size)
    # plt.ylim(0, y_size)
    # plt.savefig('base.png')

    assert recovered.shape == (x_size, y_size)
    assert recovered.dtype == np.float64
    assert (recovered == base_array).all()
