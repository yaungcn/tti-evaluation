import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def parse_mesh(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    nodes = []
    elements = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("% Elements (triangles)"):
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("%"):
                    break
                parts = line.split()
                if len(parts) >= 3:
                    elements.append([int(p) for p in parts[:3]])
                i += 1
            break

        if line and not line.startswith("%") and not line.startswith("#"):
            try:
                coords = [float(x) for x in line.split()]
                if len(coords) >= 2:
                    nodes.append(coords[:2])
            except ValueError:
                pass

        i += 1

    nodes = np.array(nodes)
    elements = np.array(elements) - 1

    return nodes, elements


def visualize_mesh(
    filename,
    show_nodes=True,
    show_elements=True,
    node_color="red",
    node_size=10,
    edge_color="black",
    fill_color="lightblue",
    alpha=0.7,
    title=None,
    figsize=(10, 10),
    save_path=None,
    xlim=None,
    ylim=None,
    show_square=True,
    ax=None,
):
    nodes, elements = parse_mesh(filename)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if show_elements:
        polygons = nodes[elements]
        collection = PolyCollection(
            polygons,
            alpha=alpha,
            facecolors=fill_color,
            edgecolors=edge_color,
            linewidths=0.5,
        )
        ax.add_collection(collection)

    if show_nodes:
        ax.scatter(nodes[:, 0], nodes[:, 1], c=node_color, s=node_size, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel(r"X ($m$)")
    ax.set_ylabel(r"Y ($m$)")

    if title is None:
        title = f"Mesh: {len(nodes)} nodes, {len(elements)} elements"
    ax.set_title(title)

    ax.autoscale()

    if show_square:
        center_x = 0
        center_y = 0
        half_size = 5
        square = plt.Rectangle(
            (center_x - half_size, center_y - half_size),
            10,
            10,
            fill=False,
            edgecolor="k",
            linewidth=1.5,
            linestyle="-",
        )

        half_size2 = 10
        square2 = plt.Rectangle(
            (-half_size2, -half_size2),
            20,
            20,
            fill=False,
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(square2)

        x_scatter = [
            -5, -5, -5, -5, -5, 5, 5, 5, 5, 5,
        ]
        y_scatter = [
            -5, -2.5, 0, 2.5, 5, -5, -2.5, 0, 2.5, 5,
        ]
        sx = [0, 0, 0, 0, 0]
        sy = [-5, -2.5, 0, 2.5, 5]
        ax.add_patch(square)
        ax.scatter(sx, sy, marker="X", c="w", edgecolors="k", s=80, linewidths=1.2)
        ax.scatter(
            x_scatter,
            y_scatter,
            marker="o",
            c="w",
            edgecolors="k",
            s=80,
            linewidths=1.2,
        )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if ax is None and save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)

    return fig, ax
