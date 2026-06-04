import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score, mean_squared_error

TABLEAU_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#1a55a3", "#a0522d", "#6a3d9a",
]

MARKERS = {
    "L0": ["o", "^", "s"],
    "L1": ["v", "<", ">", "d"],
    "L2": ["p", "*", "h"],
    "L3": ["X", "P"],
    "L4": ["D"],
}

GLOBAL_FIGSIZE = (8, 8)
GLOBAL_DPI = 300
GLOBAL_MARKER_SIZE = 40
GLOBAL_SINGLE_FIGSIZE = (6, 6)

STYLE_CONFIG = {
    "figsize_main": GLOBAL_FIGSIZE,
    "figsize_single": GLOBAL_SINGLE_FIGSIZE,
    "dpi": GLOBAL_DPI,
    "marker_size": GLOBAL_MARKER_SIZE,
    "alpha": 0.7,
    "edgecolors": "w",
    "linewidths": 0.5,
}


def _setup_rcparams():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })
    plt.rcParams["axes.linewidth"] = 2.0


def create_main_comparison_plot(data_sequences, x_min=0, x_max=15, savepath="./image"):
    plt.rcParams["axes.linewidth"] = 2.0
    fig, ax = plt.subplots(
        figsize=STYLE_CONFIG["figsize_main"], dpi=STYLE_CONFIG["dpi"]
    )

    ax.plot([x_min, x_max], [x_min, x_max], "k--", linewidth=1.5, alpha=0.7)

    for config in data_sequences:
        ax.scatter(
            np.array(config["ref_data"]),
            np.array(config["inv_data"]),
            label=config["name"],
            alpha=STYLE_CONFIG["alpha"],
            s=STYLE_CONFIG["marker_size"],
            c=config["color"],
            marker=config["marker"],
            edgecolors=STYLE_CONFIG["edgecolors"],
            linewidths=STYLE_CONFIG["linewidths"],
        )

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([x_min, x_max])
    ax.set_aspect("equal")

    os.makedirs(savepath, exist_ok=True)
    for ext in ["svg", "png"]:
        plt.savefig(
            f"{savepath}/main_comparison.{ext}",
            dpi=STYLE_CONFIG["dpi"], bbox_inches="tight",
            transparent=True,
        )
    plt.close()

    return fig


def create_individual_scatterplot(
    sequence_config, x_min=0, x_max=15, savepath="./image", fig_indx_show=False
):
    plt.rcParams["axes.linewidth"] = 2.0
    fig, ax = plt.subplots(
        figsize=STYLE_CONFIG["figsize_single"], dpi=STYLE_CONFIG["dpi"]
    )
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    refdata = np.array(sequence_config["ref_data"])
    estdata = np.array(sequence_config["inv_data"])
    xylim = [x_min, x_max]

    ax.plot(xylim, xylim, color="black", linestyle="--")

    ax.scatter(
        refdata, estdata,
        alpha=STYLE_CONFIG["alpha"],
        s=STYLE_CONFIG["marker_size"],
        c=sequence_config["color"],
        marker=sequence_config["marker"],
        edgecolors=STYLE_CONFIG["edgecolors"],
        linewidths=STYLE_CONFIG["linewidths"],
    )

    fit = np.polyfit(refdata.flatten(), estdata.flatten(), 1)
    p = np.poly1d(fit)
    est_fit = p(refdata)
    ax.plot(xylim, [p(x) for x in xylim], "r-")

    r2 = r2_score(estdata, est_fit)
    mse = mean_squared_error(refdata, estdata)
    rmse = np.sqrt(mse)
    mbe = np.mean(estdata - refdata)

    if fit[1] >= 0:
        text_func = r"$y = $" + f"{fit[0]:.2f}" + r"$x + $" + f"{np.abs(fit[1]):.2f}\n"
    else:
        text_func = r"$y = $" + f"{fit[0]:.2f}" + r"$x - $" + f"{np.abs(fit[1]):.2f}\n"

    text_content = (
        sequence_config.get("level", "")
        + "\n"
        + text_func
        + r"$R^2$" + f": {r2:.2f}\n"
        + r"$MBE$" + f"   : {mbe:.2f} (s)\n"
        + r"$RMSE$" + f": {rmse:.2f} (s)"
    )

    ax.text(
        0.55, 0.36, text_content,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=1),
    )

    if fig_indx_show and "fig_indx" in sequence_config:
        ax.text(
            0.05, 0.95, f"({sequence_config['fig_indx']})",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
        )

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([x_min, x_max])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_aspect("equal")

    os.makedirs(savepath, exist_ok=True)
    filename = f"{savepath}/scatter_{sequence_config['name']}"
    for ext in ["svg", "png"]:
        plt.savefig(
            f"{filename}.{ext}",
            dpi=STYLE_CONFIG["dpi"], bbox_inches="tight",
            pad_inches=0, transparent=True,
        )
    plt.close()


def generate_all_consistent_plots(
    data_sequences, x_min=0, x_max=15, savepath="./image", fig_indx_show=False
):
    _setup_rcparams()
    os.makedirs(savepath, exist_ok=True)

    create_main_comparison_plot(data_sequences, x_min=x_min, x_max=x_max, savepath=savepath)

    for config in data_sequences:
        xylim = [0, 6]
        create_individual_scatterplot(
            config, x_min=xylim[0], x_max=xylim[1],
            savepath=savepath, fig_indx_show=fig_indx_show,
        )


def create_tt_level_legend(data_sequences, savepath="./image", ncol=2):
    _setup_rcparams()

    labels = [ds.get("level", ds["name"]) for ds in data_sequences]
    markers = [ds["marker"] for ds in data_sequences]
    colors = [ds["color"] for ds in data_sequences]

    fig_legend = plt.figure(figsize=(8, 2))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")

    handles = [
        mlines.Line2D([], [], color=color, marker=marker, linestyle="None",
                       markersize=8, label=label, markeredgewidth=1.5)
        for label, marker, color in zip(labels, markers, colors)
    ]

    legend = ax_legend.legend(
        handles=handles, loc="center", frameon=True, framealpha=0.9,
        edgecolor="gray", fontsize=14, ncol=ncol,
        handlelength=2.0, handletextpad=0.5, columnspacing=1.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_linewidth(1)

    plt.tight_layout()
    for ext in ["png", "svg"]:
        plt.savefig(f"{savepath}.{ext}", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
