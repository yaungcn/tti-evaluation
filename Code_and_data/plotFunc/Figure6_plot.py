import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np
import os
from dataFunc.statisticalProcess import read_correct_ttdata, read_ttdata


def _setup_rcparams(language="en"):
    match language:
        case "cn":
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["SimSun", "Times New Roman"]
            plt.rcParams["font.sans-serif"] = ["SimSun"]
            plt.rcParams["axes.unicode_minus"] = False
        case "en":
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams["mathtext.fontset"] = "stixsans"
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
        }
    )
    plt.rcParams["axes.linewidth"] = 2.0


def _get_legend_labels(language="en"):
    if language == "cn":
        return [
            "程函走时: \n" + r"$t_{eikonal}=\frac{r^2\bar{S_S}}{4\bar{K}}$",
            "案例 1: \n" + r"$K_{surr}=1\times10^{-6}$" + r"$(m/s)$",
            "案例 2: \n" + r"$K_{surr}=1\times10^{-5}$" + r"$(m/s)$",
            "案例 3: \n" + r"$K_{surr}=1\times10^{-4}$" + r"$(m/s)$",
            "案例 4: \n" + r"$K_{surr}=1\times10^{-3}$" + r"$(m/s)$",
            "案例 5: \n" + r"$K_{surr}=1\times10^{-2}$" + r"$(m/s)$",
        ]
    else:
        return [
            "eikonal: \n" + r"$t_{eikonal}=\frac{r^2\bar{S_S}}{4\bar{K}}$",
            "case 1: \n" + r"$K_{surr}=1\times10^{-6}$" + r"$(m/s)$",
            "case 2: \n" + r"$K_{surr}=1\times10^{-5}$" + r"$(m/s)$",
            "case 3: \n" + r"$K_{surr}=1\times10^{-4}$" + r"$(m/s)$",
            "case 4: \n" + r"$K_{surr}=1\times10^{-3}$" + r"$(m/s)$",
            "case 5: \n" + r"$K_{surr}=1\times10^{-2}$" + r"$(m/s)$",
        ]


def _get_line_styles():
    color = ["r", "k", "k", "k", "k", "k"]
    return [
        {"color": color[0], "marker": "+", "linestyle": "-", "linewidth": 1},
        {"color": color[1], "marker": "^", "linestyle": "--", "linewidth": 1},
        {"color": color[2], "marker": "v", "linestyle": "-.", "linewidth": 1},
        {"color": color[3], "marker": "x", "linestyle": "-", "linewidth": 1},
        {"color": color[4], "marker": "s", "linestyle": ":", "linewidth": 1},
        {"color": color[5], "marker": "D", "linestyle": "--", "linewidth": 1},
    ]


def load_figure6_data(
    workpath="./out",
    field_name="dataset_scenarios",
    scenarios=None,
    cases_name=None,
):
    if scenarios is None:
        scenarios = ["3a"]
    if cases_name is None:
        cases_name = [
            "mean_-13.8",
            "mean_-11.5",
            "mean_-9.2",
            "mean_-6.9",
            "mean_-4.6",
        ]

    cases_path = []
    for case_name in cases_name:
        cases_path.append(
            f"{workpath}/{field_name}/{scenarios[0]}/{case_name}/model_export"
        )

    correct_ttdata_cases = []
    ttdata_cases = []
    for path in cases_path:
        case_correct_ttdata = read_correct_ttdata(path)
        case_correct_ttdata = case_correct_ttdata.correct_data[0]
        case_correct_ttdata = np.array(case_correct_ttdata)
        correct_ttdata_cases.append(case_correct_ttdata)

        case_ttdata = read_ttdata(path)
        case_ttdata = case_ttdata.data[0]
        case_ttdata = np.array(case_ttdata)
        ttdata_cases.append(case_ttdata)

    length_path_2 = [
        5**2,
        5**2 + 2.5**2,
        5**2 + 5**2,
        5**2 + 7.5**2,
        5**2 + 10**2,
    ]

    ttdata_eikonal = [l / 4 / np.exp(2.3) for l in length_path_2]

    ttdata = [ttdata_eikonal] + [
        [ttdata_cases[i][0, j] for j in range(5)] for i in range(5)
    ]
    correct_ttdata = [ttdata_eikonal] + [
        [correct_ttdata_cases[i][0, j] for j in range(5)] for i in range(5)
    ]

    return length_path_2, ttdata, correct_ttdata


def _plot_traveltime_lines(
    ax, length_path_2, correct_ttdata, line_styles, legend_labels
):
    for i in range(6):
        ax.plot(
            length_path_2,
            correct_ttdata[i],
            label=legend_labels[i],
            color=line_styles[i]["color"],
            marker=line_styles[i]["marker"],
            linestyle=line_styles[i]["linestyle"],
            linewidth=line_styles[i]["linewidth"],
            markersize=6 if i > 0 else 8,
        )


def _set_axis_labels(ax, language="en"):
    if language == "cn":
        ax.set_ylabel("时间 " + r"$(s)$")
        ax.set_xlabel("路径长度" + r"$^2\ $" + r"$(m^2)$")
    else:
        ax.set_ylabel("Time " + r"$(s)$")
        ax.set_xlabel("Interval Length" + r"$^2\ $" + r"$(m^2)$")


def _add_subfigure_label(ax, label, x=0.02, y=0.98, fontsize=20):
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=fontsize,
    )


def _add_subfigure_label_fig(fig, ax, label, x_inset=0.01, y_inset=0.015, fontsize=20):
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + x_inset,
        bbox.y1 - y_inset,
        f"({label})",
        va="top",
        ha="left",
        fontsize=fontsize,
    )


def _save_figure(fig, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path + ".png", dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(
        save_path + ".svg",
        format="svg",
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=True,
    )


def plot_figure6a(
    length_path_2,
    correct_ttdata,
    line_styles,
    legend_labels,
    language="en",
    save_dir="./figure_table_output/Figure/Figure6",
):
    _setup_rcparams(language)
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_traveltime_lines(
        ax, length_path_2, correct_ttdata, line_styles, legend_labels
    )
    _set_axis_labels(ax, language)
    ax.set_xlim([0, 180])
    ax.set_ylim([0, 14])
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
    )
    ax.grid(True, alpha=0.4, linestyle="-", axis="y")
    _add_subfigure_label(ax, "a")
    _save_figure(fig, f"{save_dir}/Figure6a")
    plt.close()
    return fig


def plot_figure6b(
    length_path_2,
    correct_ttdata,
    line_styles,
    legend_labels,
    language="en",
    save_dir="./figure_table_output/Figure/Figure6",
):
    _setup_rcparams(language)
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_traveltime_lines(
        ax, length_path_2, correct_ttdata, line_styles, legend_labels
    )
    ax.set_xlim([20, 60])
    ax.set_ylim([0.5, 2.0])
    ax.grid(True, alpha=0.4, linestyle="-", axis="y")
    _add_subfigure_label(ax, "b")
    _save_figure(fig, f"{save_dir}/Figure6b")
    plt.close()
    return fig


def plot_figure6c(
    length_path_2,
    correct_ttdata,
    line_styles,
    legend_labels,
    language="en",
    save_dir="./figure_table_output/Figure/Figure6",
):
    _setup_rcparams(language)
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_traveltime_lines(
        ax, length_path_2, correct_ttdata, line_styles, legend_labels
    )
    ax.set_xlim([70, 130])
    ax.set_ylim([1.5, 3.5])
    ax.grid(True, alpha=0.4, linestyle="-", axis="y")
    _add_subfigure_label(ax, "c")
    _save_figure(fig, f"{save_dir}/Figure6c")
    plt.close()
    return fig


def plot_figure6_combined(
    length_path_2,
    correct_ttdata,
    line_styles,
    legend_labels,
    language="en",
    save_dir="./figure_table_output/Figure/Figure6",
):
    _setup_rcparams(language)

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], hspace=0.25, wspace=0.2)

    # -- subfigure (a): full view, spans both columns --
    axa = fig.add_subplot(gs[0, :])
    _plot_traveltime_lines(
        axa, length_path_2, correct_ttdata, line_styles, legend_labels
    )
    _set_axis_labels(axa, language)
    axa.set_xlim([0, 180])
    axa.set_ylim([0, 14])
    axa.legend(
        loc="center right",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
    )
    axa.grid(True, alpha=0.4, linestyle="-", axis="y")

    # red dashed rectangles marking zoom regions
    rect_b = Rectangle(
        (20, 0.5),
        40,
        1.5,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    axa.add_patch(rect_b)

    axa.text(
        22,
        2.2,
        "(b)",
        color="red",
        fontsize=14,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    rect_c = Rectangle(
        (70, 1.5),
        60,
        2.0,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    axa.add_patch(rect_c)

    axa.text(
        72,
        3.7,
        "(c)",
        color="red",
        fontsize=14,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    # -- subfigure (b): zoom 1 --
    axb = fig.add_subplot(gs[1, 0])
    _plot_traveltime_lines(
        axb, length_path_2, correct_ttdata, line_styles, legend_labels
    )
    _set_axis_labels(axb, language)
    axb.set_xlim([20, 60])
    axb.set_ylim([0.5, 2.0])
    axb.grid(True, alpha=0.4, linestyle="-", axis="y")

    # -- subfigure (c): zoom 2 --
    axc = fig.add_subplot(gs[1, 1])
    _plot_traveltime_lines(
        axc, length_path_2, correct_ttdata, line_styles, legend_labels
    )
    _set_axis_labels(axc, language)
    axc.set_xlim([70, 130])
    axc.set_ylim([1.5, 3.5])
    axc.grid(True, alpha=0.4, linestyle="-", axis="y")

    fig.subplots_adjust(right=0.78)

    _add_subfigure_label_fig(fig, axa, "a")
    _add_subfigure_label_fig(fig, axb, "b")
    _add_subfigure_label_fig(fig, axc, "c")

    _save_figure(fig, f"{save_dir}/Figure6_combined")
    plt.close()
    return fig
