"""
Contains objects and functions required to generate the quad tree of images.
"""


from typing import Any, Optional

from itertools import chain

import numpy as np


class TreeConfiguration:
    def __init__(
        self,
        base_grid_size: int,
        refinement_levels: int,
        dtype=np.float32,
        base_grid_channels: int = 1,
    ):
        """
        Parameters
        ----------

        base_grid_size : int
            The size of the base grid cells (i.e. one
            dimensional resolution of images that we segment into).
        refinement_levels : int
            The number of refinement levels to use. This is
            the number of times we will subdivide the image. The top level cells
            will cover base_grid_size * 2 ** refinement_levels pixels.
        dtype : np.dtype
            Data type to use for derived grids.
        base_grid_channels: int
            Number of channels to use for the base grid. For raw data, this is 1,
            for RGBA images, this is 4.
        """

        self.base_grid_size = base_grid_size
        self.refinement_levels = refinement_levels
        self.dtype = dtype
        self.base_grid_channels = base_grid_channels

        if self.base_grid_size % 2 != 0:
            raise ValueError("Base grid size must be even.")

    @property
    def grid_specification(self) -> dict[str, Any]:
        """
        Kwargs specification of the grid for numpy constructors.

        Returns
        -------
        dict[str, Any]
            Dictionary containing both the shape and data type.
        """

        if self.base_grid_channels == 1:
            shape = (self.base_grid_size, self.base_grid_size)
        else:
            shape = (self.base_grid_size, self.base_grid_size, self.base_grid_channels)

        return dict(shape=shape, dtype=self.dtype)


class Node:
    __slots__ = ("x", "y", "data", "size", "level", "children")

    def __init__(
        self,
        x: int,
        y: int,
        size: int,
        level: int,
        data: np.ndarray,
    ):
        """
        Parameters
        ----------

        x : int
            The x coordinate of the node.

        y : int
            The y coordinate of the node.

        size : int
            The size of the node, in base original grid units.

        level : int
            The level of the node in the tree. The top level nodes are at level 0.

        data : np.ndarray
            The data contained in the node. This is a 2D array corresponding to
            the data at the bottom level of (size, size), but is in fact a fixed
            size buffer as defined by the TreeConfiguration.
        """
        self.x = x
        self.y = y
        self.data = data
        self.size = size
        self.level = level
        self.children: Optional[list[list[Node]]] = None

    @property
    def flat_children(self) -> chain["Node"]:
        """
        A flattened list of all children of this node. This is useful for iterating
        over all nodes when you do not care about order.
        """
        return chain.from_iterable(self.children)


class QuadTree:
    initialized: bool
    "Whether the tree has been initialized."
    configuration: TreeConfiguration
    "Configuration object for this tree."
    nodes: list[list[Node]]
    "Top level node grid"
    nx_top_level: int
    "Number of cells on top level in x direction (first index)."
    ny_top_level: int
    "Number of cells on top level in y direction (second index)."
    top_level_grid_size: int
    "Size of top level grid cells in base level grid units. Entirely determined by configuration."

    def __init__(self, configuration: TreeConfiguration):
        self.configuration = configuration
        self.initialized = False

        return

    def initialize_from_node_list(self, nodes: list[list[Node]]):
        """
        Initialize the tree from a list of nodes. This is not recommended in general as
        it may lead to a set of nodes that are not consistent with the configuration.

        Parameters
        ----------

        nodes : list[list[Node]]
            The list of nodes to use to initialize the tree. This should be a list of lists
            of nodes, where the first index corresponds to the x index of the node, and
            the second index corresponds to the y index  of the node.
        """
        self.nodes = nodes
        self.nx_top_level = len(nodes)
        self.ny_top_level = len(nodes[0])
        self.top_level_grid_size = (
            self.configuration.base_grid_size
            * 2**self.configuration.refinement_levels
        )

        return

    def initialize_from_array(self, data=None):
        self.top_level_grid_size = (
            self.configuration.base_grid_size
            * 2**self.configuration.refinement_levels
        )
        self.nx_top_level = int(np.ceil(data.shape[0] / self.top_level_grid_size))
        self.ny_top_level = int(np.ceil(data.shape[1] / self.top_level_grid_size))

        grid = [
            [None for _ in range(self.ny_top_level)] for _ in range(self.nx_top_level)
        ]

        def populate_children(node: Node):
            if node.level == self.configuration.refinement_levels:
                # Actually need to read the data from main grid.
                node.data = np.zeros(**self.configuration.grid_specification)

                # Find which area actually overlaps.
                x_min = min(node.x, data.shape[0])
                x_max = min(node.x + node.size, data.shape[0])
                y_min = min(node.y, data.shape[1])
                y_max = min(node.y + node.size, data.shape[1])

                dx = x_max - x_min
                dy = y_max - y_min

                if (dx == 0) and (dy == 0):
                    return

                input_selector = np.s_[x_min : x_min + dx, y_min : y_min + dy]
                output_selector = np.s_[0:dx, 0:dy]

                node.data[output_selector] = data[input_selector]

                return

            half_node_size = node.size // 2

            node.children = [
                [
                    Node(
                        x=node.x + half_node_size * x,
                        y=node.y + half_node_size * y,
                        size=node.size // 2,
                        level=node.level + 1,
                        data=None,
                    )
                    for y in range(2)
                ]
                for x in range(2)
            ]

            for child in node.flat_children:
                populate_children(child)

        for x in range(self.nx_top_level):
            for y in range(self.ny_top_level):
                node = Node(
                    x=x * self.top_level_grid_size,
                    y=y * self.top_level_grid_size,
                    size=self.top_level_grid_size,
                    level=0,
                    data=None,
                )

                populate_children(node)

                grid[x][y] = node

        self.nodes = grid

    def walk_tree_and_populate(self):
        """
        Walks the tree and populates the lower resolution nodes with data from the
        floor.
        """

        def walk_tree_and_populate_lower_resolution(node: Node):
            if node.level == self.configuration.refinement_levels:
                return

            for child in node.flat_children:
                walk_tree_and_populate_lower_resolution(child)

            # Now populate this node with the sum of its children.

            base_grid_size = self.configuration.base_grid_size

            # Average the children.
            node.data = np.empty(**self.configuration.grid_specification)

            def average_child(x, y):
                child_data = node.children[x][y].data

                if np.issubdtype(self.configuration.dtype, np.floating):
                    quarter_data = 0.25 * child_data
                else:
                    # We lose some precision doing it this way, but it prevents
                    # overflows.
                    quarter_data = child_data // 4

                return (
                    quarter_data[::2, ::2]
                    + quarter_data[1::2, ::2]
                    + quarter_data[::2, 1::2]
                    + quarter_data[1::2, 1::2]
                )

            node.data[0 : base_grid_size // 2, 0 : base_grid_size // 2] = average_child(
                0, 0
            )
            node.data[
                0 : base_grid_size // 2, base_grid_size // 2 : base_grid_size
            ] = average_child(0, 1)
            node.data[
                base_grid_size // 2 : base_grid_size, 0 : base_grid_size // 2
            ] = average_child(1, 0)
            node.data[
                base_grid_size // 2 : base_grid_size,
                base_grid_size // 2 : base_grid_size,
            ] = average_child(1, 1)

            return

        for x in range(self.nx_top_level):
            for y in range(self.ny_top_level):
                walk_tree_and_populate_lower_resolution(self.nodes[x][y])

        self.initialized = True

        return

    def walk_tree_to_level_and_index(self, x: int, y: int, level: int) -> Node:
        """
        Walks the tree down to a fixed level and x, y cell index. This is useful
        for leaflet indexing.

        Parameters
        ----------
        x : int
            Horizontal cell index at `level`
        y : int
            Vertical cell index at `level`
        level : int
            Level to recurse down to

        Returns
        -------
        Node
            Tree node at index `x` and `y` at level.
        """

        top_level_index_x = x // 2**level
        top_level_index_y = y // 2**level

        node = self.nodes[top_level_index_x][top_level_index_y]

        current_level = 0

        left_target_level = top_level_index_x * 2**level
        right_target_level = (top_level_index_x + 1) * 2**level
        bottom_target_level = top_level_index_y * 2**level
        top_target_level = (top_level_index_y + 1) * 2**level

        while current_level < level:
            x_center = (right_target_level + left_target_level) // 2
            y_center = (top_target_level + bottom_target_level) // 2

            if x < x_center:
                right_target_level = x_center
                if y < y_center:
                    node = node.children[0][0]
                    top_target_level = y_center
                else:
                    node = node.children[0][1]
                    bottom_target_level = y_center
            else:
                left_target_level = x_center
                if y < y_center:
                    node = node.children[1][0]
                    top_target_level = y_center
                else:
                    node = node.children[1][1]
                    bottom_target_level = y_center

            current_level += 1

        return node.data

    def extract_pixels(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Extracts requested pixels to a buffer. Top left is (x, y).

        Parameters
        ----------
        x : int
            Top left x coordinate
        y : int
            Top left y coordinate
        width : int
            Width of requested image in pixels
        height : int
            Height of requested image in pixels

        Returns
        -------
        np.ndarray
            Requested pixel buffer of size (width, height)

        Raises
        ------
        RuntimeError
            If the number of channels in the base grid is not 1, 3, or 4.
        """

        if self.configuration.base_grid_channels == 1:
            output_buffer = np.zeros((width, height), dtype=self.configuration.dtype)
        elif self.configuration.base_grid_channels == 3:
            output_buffer = np.zeros((width, height, 3), dtype=self.configuration.dtype)
        elif self.configuration.base_grid_channels == 4:
            output_buffer = np.zeros((width, height, 4), dtype=self.configuration.dtype)
            # Set all alpha channels to full.
            if np.issubdtype(self.configuration.dtype, np.floating):
                output_buffer[:, :, 3] = 1.0
            elif np.issubdtype(self.configuration.dtype, np.unit8):
                output_buffer[:, :, 3] = 255
            else:
                raise RuntimeError("Cannot understand data type for output buffer.")
        else:
            raise RuntimeError(
                "Cannot understand number of channels for output buffer "
                f"({self.configuration.base_grid_channels}, should be one "
                "of 1, 3, or 4)."
            )

        def node_overlaps(node: Node) -> bool:
            x_spans_left_edge = x <= node.x and x + width > node.x
            x_spans_right_edge = (
                x < node.x + node.size and x + width >= node.x + node.size
            )
            x_contains = x >= node.x and x + width <= node.x + node.size

            x_valid = x_spans_left_edge or x_spans_right_edge or x_contains

            print(x_spans_left_edge, x_spans_right_edge, x_contains, x_valid, "x")

            y_spans_bottom_edge = y <= node.y and y + height > node.y
            y_spans_top_edge = (
                y < node.y + node.size and y + height >= node.y + node.size
            )
            y_contains = y >= node.y and y + height <= node.y + node.size

            y_valid = y_spans_bottom_edge or y_spans_top_edge or y_contains

            print(y_spans_bottom_edge, y_spans_top_edge, y_contains, y_valid, "y")

            return x_valid and y_valid

        def recurse_tree(node: Node):
            if not node_overlaps(node):
                return

            if node.level == self.configuration.refinement_levels:
                # We're at the bottom... Figure out where we overlap with the buffer.
                input_selector = np.s_[
                    max(x - node.x, 0) : min(x - node.x + width, node.size),
                    max(y - node.y, 0) : min(y - node.y + height, node.size),
                ]

                output_selector = np.s_[
                    max(node.x - x, 0) : min(node.x - x + node.size, width),
                    max(node.y - y, 0) : min(node.y - y + node.size, height),
                ]

                print("input", input_selector)
                print("output", output_selector)

                output_buffer[output_selector] = node.data[input_selector]

                return

            for child in node.flat_children:
                recurse_tree(child)

        for node in chain.from_iterable(self.nodes):
            recurse_tree(node=node)

        return output_buffer
