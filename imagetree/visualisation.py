"""
Tree visualisation functions, for use in debugging and exploring trees.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tree import Node


def plot_grid(grid: list[list[Node]], fig=None, ax=None) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the cell grid at the top level.

    Parameters
    ----------
    grid : list[list[Node]]
        Cell grid from tree.
    fig : plt.Figure, optional
        Matplotlib figure, will be generated if not given, by default None
    ax : plt.Axes, optional
        Matplotlib axes, will be generated if not given., by default None

    Returns
    -------
    plt.Figure
        Figure containing the plot.
    plt.Axes
        Axes containing the plot.
    """

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    for grid_row in grid:
        for node in grid_row:
            ax.imshow(
                node.data.T,
                extent=[
                    node.x,
                    node.x + node.size,
                    node.y,
                    node.y + node.size,
                ],
                origin="lower",
                vmin=-100,
                vmax=100,
                cmap="Grays",
            )

            ax.add_patch(
                Rectangle(
                    (node.x, node.y),
                    node.size,
                    node.size,
                    edgecolor="C0",
                    facecolor="none",
                )
            )

    return fig, ax


def plot_hierarchy_for_cell(node: Node, refinement_levels: int, fig=None, ax=None):
    """
    Cell hierarchy plotter, given a node.

    Parameters
    ----------
    node : Node
        Node to plot.
    refinement_levels : int
        Refinement levels from the tree configuration.
    fig : plt.Figure, optional
        Optional figure argument to plot on top of, by default None
    ax : plt.Axes, optional
        Axes to plot on top of, by default None
    """

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    ax.add_patch(
        Rectangle(
            (node.x, node.y),
            node.size,
            node.size,
            edgecolor=f"C{node.level}",
            facecolor="none",
            zorder=100 - node.level,
        )
    )

    if node.level == refinement_levels:
        ax.imshow(
            node.data.T,
            extent=[
                node.x,
                node.x + node.size,
                node.y,
                node.y + node.size,
            ],
            origin="lower",
            vmin=-100,
            vmax=100,
            cmap="Grays",
        )

        return
    else:
        # Display final child.
        child = node.children[-1][-1]

        ax.imshow(
            child.data.T,
            extent=[
                child.x,
                child.x + child.size,
                child.y,
                child.y + child.size,
            ],
            origin="lower",
            vmin=-100,
            vmax=100,
            cmap="Grays",
        )

    # Leave final cell at this 'level' unsplit.
    for child in node.flat_children[:-1]:
        plot_hierarchy_for_cell(
            child, refinement_levels=refinement_levels, fig=fig, ax=ax
        )
