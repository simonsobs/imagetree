"""
Contains objects and functions required to generate the quad tree of images.
"""


import numpy as np


class TreeConfiguration:
    def __init__(self, base_grid_size: int, refinement_levels: int, dtype=np.float32):
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
        """

        self.base_grid_size = base_grid_size
        self.refinement_levels = refinement_levels
        self.dtype = dtype


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
        self.children = None


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
                node.data = np.zeros(
                    (
                        self.configuration.base_grid_size,
                        self.configuration.base_grid_size,
                    ),
                    dtype=self.configuration.dtype,
                )

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
                Node(
                    x=node.x + half_node_size * x,
                    y=node.y + half_node_size * y,
                    size=node.size // 2,
                    level=node.level + 1,
                    data=None,
                )
                for x in range(2)
                for y in range(2)
            ]

            for child in node.children:
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

            for child in node.children:
                walk_tree_and_populate_lower_resolution(child)

            # Now populate this node with the sum of its children.

            base_grid_size = self.configuration.base_grid_size

            # Average the children.
            node.data = np.empty(
                (base_grid_size, base_grid_size), dtype=self.configuration.dtype
            )

            def average_child(index):
                child_data = node.children[index].data

                return 0.25 * (
                    child_data[::2, ::2]
                    + child_data[1::2, ::2]
                    + child_data[::2, 1::2]
                    + child_data[1::2, 1::2]
                )

            node.data[0 : base_grid_size // 2, 0 : base_grid_size // 2] = average_child(
                0
            )
            node.data[
                0 : base_grid_size // 2, base_grid_size // 2 : base_grid_size
            ] = average_child(1)
            node.data[
                base_grid_size // 2 : base_grid_size, 0 : base_grid_size // 2
            ] = average_child(2)
            node.data[
                base_grid_size // 2 : base_grid_size,
                base_grid_size // 2 : base_grid_size,
            ] = average_child(3)

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
                    node = node.children[0]
                    top_target_level = y_center
                else:
                    node = node.children[1]
                    bottom_target_level = y_center
            else:
                left_target_level = x_center
                if y < y_center:
                    node = node.children[2]
                    top_target_level = y_center
                else:
                    node = node.children[3]
                    bottom_target_level = y_center

            current_level += 1

        return node.data
