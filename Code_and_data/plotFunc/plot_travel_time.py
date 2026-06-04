import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import Normalize, LogNorm, PowerNorm


def load_travel_time_data(filepath):
    data = np.load(filepath, allow_pickle=True).item()
    x = data["x"]
    y = data["y"]
    t_peak = data["t_peak"]
    simulate_time = data.get("simulate_time", None)
    print(f"走时数据已读取: {len(x)} 个节点")
    if simulate_time:
        print(f"模拟时间范围: {simulate_time}")
    return x, y, t_peak, simulate_time


def plot_travel_time_map(
    x,
    y,
    t_peak,
    xlim=None,
    ylim=None,
    title="Travel Time Map",
    save_path=None,
    save=True,
    show_square=True,
    vmin=None,
    vmax=None,
    levels=50,
    contour_levels=20,
    contour_ranges=None,
    norm_type="linear",
    norm_gamma=1.0,
    figsize=(12, 10),
    ax=None,
    cax=None,
):
    if xlim is None:
        xlim = (x.min(), x.max())
    if ylim is None:
        ylim = (y.min(), y.max())

    xi = np.linspace(xlim[0], xlim[1], 200)
    yi = np.linspace(ylim[0], ylim[1], 200)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((x, y), t_peak, (Xi, Yi), method="cubic")
    Zi = np.nan_to_num(Zi, nan=0)

    if vmin is None:
        vmin = Zi.min()
    if vmax is None:
        vmax = Zi.max()

    Zi_clipped = np.clip(Zi, vmin, vmax)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if contour_ranges is not None:
        contour_levels = []
        for r in contour_ranges:
            contour_levels.extend(np.arange(r[0], r[1] + r[2], r[2]).tolist())
        contour_levels = sorted(list(set(contour_levels)))
        levels = contour_levels
    else:
        if isinstance(levels, int):
            levels = np.linspace(vmin, vmax, levels)
        if isinstance(contour_levels, int):
            contour_levels = np.linspace(vmin, vmax, contour_levels)

    if norm_type == "log":
        if vmin <= 0:
            vmin = 1e-10
            print(f"Warning: LogNorm requires vmin > 0, 已自动调整为 {vmin}")
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif norm_type == "power":
        norm = PowerNorm(gamma=norm_gamma, vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    cf = ax.contourf(Xi, Yi, Zi_clipped, levels=levels, cmap="jet", norm=norm)
    cs = ax.contour(
        Xi, Yi, Zi_clipped, levels=contour_levels, colors="black", linewidths=0.5
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    half_size = 5
    square = plt.Rectangle(
        (-half_size, -half_size),
        10,
        10,
        fill=False,
        edgecolor="k",
        linewidth=1,
    )
    ax.add_patch(square)

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

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    if cax is not None:
        cbar = fig.colorbar(cf, cax=cax)
    elif ax is None:
        cbar = fig.colorbar(cf, ax=ax)
    else:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        _cax = divider.append_axes("right", size="7%", pad=0.1)
        cbar = fig.colorbar(cf, cax=_cax)
    cbar.set_label(r"Travel Time ($s$)")

    if show_square:
        x_scatter = [
            -5, -5, -5, -5, -5, 5, 5, 5, 5, 5,
        ]
        y_scatter = [
            -5, -2.5, 0, 2.5, 5, -5, -2.5, 0, 2.5, 5,
        ]
        sx = [0, 0, 0, 0, 0]
        sy = [-5, -2.5, 0, 2.5, 5]

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

    if ax is None:
        plt.tight_layout()
        if save:
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
                print(f"图片已保存至: {save_path}")
            else:
                plt.savefig(
                    "travel_time_map.png", dpi=300, bbox_inches="tight", transparent=True
                )
                print("图片已保存至: travel_time_map.png")
        plt.close()
    return fig, ax
