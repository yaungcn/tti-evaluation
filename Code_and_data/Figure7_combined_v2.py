import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from plotFunc.plot_mesh import visualize_mesh
from plotFunc.plot_travel_time import load_travel_time_data, plot_travel_time_map

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)
plt.rcParams["axes.linewidth"] = 1.0

data_sequences = [
    {
        "name": "case1",
        "surr_param": -13.8,
        "max_time": 200,
        "countor_range": [(0, 2.5, 0.5), (2.5, 22.5, 5), (20, 200, 40)],
    },
    {
        "name": "case2",
        "surr_param": -11.5,
        "max_time": 150,
        "countor_range": [(0, 10, 0.5), (10, 20, 5), (20, 150, 20)],
    },
    {
        "name": "case3",
        "surr_param": -9.2,
        "max_time": 25,
        "countor_range": [(0, 10, 0.5), (10, 25, 2.5)],
    },
    {
        "name": "case4",
        "surr_param": -6.9,
        "max_time": 7.5,
        "countor_range": [(0, 5.5, 0.5), (5.5, 7.5, 1)],
    },
    {
        "name": "case5",
        "surr_param": -4.6,
        "max_time": 5,
        "countor_range": [(0, 5, 0.5)],
    },
]
fig = plt.figure(figsize=(22, 13))
gs = GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.35)
plot_axes = [
    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])],
]

# ---- (a) Mesh ----
visualize_mesh(
    "./data/dataset_mesh/net-20x20-v2.txt",
    xlim=(-15, 15),
    ylim=(-15, 15),
    node_color="k",
    show_nodes=False,
    edge_color="black",
    fill_color="w",
    title="",
    ax=plot_axes[0][0],
)

# ---- (b)-(f) Travel time maps ----
for idx, config in enumerate(data_sequences):
    surr = config["surr_param"]

    # case1-2 → row 0 (col 1-2), case3-5 → row 1 (col 0-2)
    if idx < 2:
        row, col = 0, idx + 1
    else:
        row, col = 1, idx - 2

    ax = plot_axes[row][col]

    x, y, t_peak, simulate_time = load_travel_time_data(
        f"./data/dataset_scenarios/3a/mean_{surr}/model_export/homo_-9.2_lnKs/travel_time.npy"
    )

    dist = np.sqrt((x - 0) ** 2 + (y - (-5)) ** 2)
    idx_nearest = np.argmin(dist)
    t_at_point = t_peak[idx_nearest]
    t_peak_relative = t_peak - t_at_point

    # Colorbar within this subplot's area using inset_axes (right side)
    cax = ax.inset_axes([1.05, 0, 0.06, 1])

    plot_travel_time_map(
        x,
        y,
        t_peak_relative,
        xlim=(-15, 15),
        ylim=(-15, 15),
        title="",
        vmin=0,
        vmax=config["max_time"],
        contour_ranges=config["countor_range"],
        norm_type="power",
        norm_gamma=0.3,
        ax=ax,
        cax=cax,
    )

# ---- Subplot labels (outside axes, aligned by column) ----
for i in range(2):
    for j in range(3):
        ax = plot_axes[i][j]
        ax.text(
            -0.2,
            0.95,
            f"({chr(ord('a') + i * 3 + j)})",
            transform=ax.transAxes,
            fontsize=24,
            va="bottom",
            ha="left",
        )

import os
save_dir = "./figure_table_output/Figure/Figure7"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(
    f"{save_dir}/Figure7_combined_v2.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2,
    transparent=True,
)
plt.savefig(
    f"{save_dir}/Figure7_combined_v2.svg",
    bbox_inches="tight",
    pad_inches=0.2,
    transparent=True,
)
plt.show()
