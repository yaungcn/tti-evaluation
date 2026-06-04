import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D


def setup_boxplot_style(language="en"):
    match language:
        case "cn":
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["SimSun", "Times New Roman"]
            plt.rcParams["font.sans-serif"] = [
                "SimSun",
                "Arial Unicode MS",
                "DejaVu Sans",
            ]

        case "en":
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams["mathtext.fontset"] = "stixsans"
            plt.rcParams["font.sans-serif"] = [
                "Times New Roman",
                "Arial Unicode MS",
                "DejaVu Sans",
            ]

    plt.rcParams["axes.unicode_minus"] = False
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


_CASE_COLORS = {
    "case1": "#fff8e6",
    "case2": "#ebd4b3",
    "case3": "#cc9e66",
    "case4": "#997a33",
}


def _draw_four_param_boxplot(
    ax1,
    case_data,
    data_label,
    x_positions,
    width,
    space_box,
    ax_lim,
    colors_override=None,
    scatter_data_1a=None,
    scatter_label_prefix=None,
    legend_font=None,
):
    """
    Core boxplot drawing engine for all 2/4 case scenarios.
    Returns dict of axes: {"ax1": ax1, "ax2": ax2, "ax3": ax3, "ax4": ax4}
    """
    median_color = "black"
    spacing = 80
    n_cases = len(case_data)

    case_colors = colors_override if colors_override else _CASE_COLORS

    metric_label = {
        "para1": data_label[0],
        "para2": data_label[1],
        "para3": data_label[2],
        "para4": data_label[3],
    }
    metric_colors = {
        "para1": "#000000",
        "para2": "#000000",
        "para3": "#000000",
        "para4": "#000000",
    }

    ax1_lim = ax_lim["ax1"]
    ax2_lim = ax_lim["ax2"]
    ax3_lim = ax_lim["ax3"]
    ax4_lim = ax_lim["ax4"]
    ticks1 = np.linspace(ax1_lim[0], ax1_lim[1], 6)
    ticks2 = np.linspace(ax2_lim[0], ax2_lim[1], 6)

    if n_cases == 2:
        offsets = [-space_box / 2, +space_box / 2]
    else:
        offsets = [
            -space_box - width,
            -space_box / 2,
            +space_box / 2,
            +space_box + width,
        ]

    # ----- para1 -----
    para1_data = [d[data_label[0]] for d in case_data]
    para1_positions = [x_positions[0] + o for o in offsets[:n_cases]]

    bp = ax1.boxplot(
        para1_data,
        positions=para1_positions,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=median_color, linewidth=2),
        boxprops=dict(linewidth=2),
    )
    for patch, key in zip(bp["boxes"], list(case_colors.keys())[:n_cases]):
        patch.set_facecolor(case_colors[key])
        patch.set_alpha(1)

    ax1.spines["left"].set_position(("outward", spacing))
    ax1.set_ylabel(metric_label["para1"], color=metric_colors["para1"])
    ax1.set_ylim(ax1_lim)
    ax1.set_yticks(ticks1)
    ax1.tick_params(axis="y", labelcolor=metric_colors["para1"])
    ax1.grid(True, alpha=1, axis="y")

    if scatter_data_1a and len(scatter_data_1a[0][data_label[0]]) > 0:
        _draw_scatter_pair(
            ax1,
            para1_positions,
            scatter_data_1a[0][data_label[0]],
            scatter_data_1a[1][data_label[0]],
            scatter_label_prefix,
            x_positions[0],
            space_box,
        )

    # ----- para2 (second left axis) -----
    ax2 = ax1.twinx()
    ax2.spines["left"].set_position(("axes", 0))
    ax2.spines["left"].set_visible(True)
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.set_ticks_position("left")

    para2_data = [d[data_label[1]] for d in case_data]
    para2_positions = [x_positions[1] + o for o in offsets[:n_cases]]

    bp = ax2.boxplot(
        para2_data,
        positions=para2_positions,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=median_color, linewidth=2),
        boxprops=dict(linewidth=2),
    )
    for patch, key in zip(bp["boxes"], list(case_colors.keys())[:n_cases]):
        patch.set_facecolor(case_colors[key])
        patch.set_alpha(1)

    ax2.set_ylabel(metric_label["para2"], color=metric_colors["para2"])
    ax2.set_ylim(ax2_lim)
    ax2.set_yticks(ticks2)
    ax2.tick_params(axis="y", labelcolor=metric_colors["para2"])

    if scatter_data_1a and len(scatter_data_1a[0][data_label[1]]) > 0:
        _draw_scatter_pair(
            ax2,
            para2_positions,
            scatter_data_1a[0][data_label[1]],
            scatter_data_1a[1][data_label[1]],
            scatter_label_prefix,
            x_positions[1],
            space_box,
        )

    # ----- para3 (first right axis) -----
    ax3 = ax1.twinx()
    para3_data = [d[data_label[2]] for d in case_data]
    pearson_positions = [x_positions[2] + o for o in offsets[:n_cases]]

    bp = ax3.boxplot(
        para3_data,
        positions=pearson_positions,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=median_color, linewidth=2),
        boxprops=dict(linewidth=2),
    )
    for patch, key in zip(bp["boxes"], list(case_colors.keys())[:n_cases]):
        patch.set_facecolor(case_colors[key])
        patch.set_alpha(1)

    ax3.spines["right"].set_position(("outward", 0))
    ax3.set_ylabel(metric_label["para3"], color=metric_colors["para3"])
    ax3.set_ylim(ax3_lim)
    ax3.tick_params(axis="y", labelcolor=metric_colors["para3"])

    # ----- para4 (second right axis) -----
    ax4 = ax1.twinx()
    para4_data = [d[data_label[3]] for d in case_data]
    ssim_positions = [x_positions[3] + o for o in offsets[:n_cases]]

    bp = ax4.boxplot(
        para4_data,
        positions=ssim_positions,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=median_color, linewidth=2),
        boxprops=dict(linewidth=2),
    )
    for patch, key in zip(bp["boxes"], list(case_colors.keys())[:n_cases]):
        patch.set_facecolor(case_colors[key])
        patch.set_alpha(1)

    ax4.spines["right"].set_position(("outward", spacing))
    ax4.set_ylabel(metric_label["para4"], color=metric_colors["para4"])
    ax4.set_ylim(ax4_lim)
    ax4.tick_params(axis="y", labelcolor=metric_colors["para4"])

    # ----- X-axis ticks -----
    x_tick_positions = [x_positions[0], x_positions[1], x_positions[2], x_positions[3]]
    x_tick_labels = [
        metric_label["para1"],
        metric_label["para2"],
        metric_label["para3"],
        metric_label["para4"],
    ]
    ax1.set_xticks(x_tick_positions)
    tick_labels = ax1.set_xticklabels(x_tick_labels)
    label_colors = [
        metric_colors["para1"],
        metric_colors["para2"],
        metric_colors["para3"],
        metric_colors["para4"],
    ]
    for label, color in zip(tick_labels, label_colors):
        label.set_color(color)

    # ----- Vertical separator lines -----
    for i in range(3):
        ax1.axvline(
            x=(x_positions[i + 1] - x_positions[i]) / 2 + x_positions[i],
            color="black",
            linestyle="--",
            alpha=0.8,
            linewidth=1,
        )

    ax1.set_axisbelow(True)

    return {"ax1": ax1, "ax2": ax2, "ax3": ax3, "ax4": ax4}


def _draw_scatter_pair(ax, positions, data1_list, data2_list, label_prefix, ref_x, space_box):
    scatter_colors = ["black", "black"]
    scatter_size = 100
    scatter_alpha = 1
    scatter_marker_0 = "X"
    scatter_marker_1 = "^"
    scatter_edgecolor = "black"
    scatter_linewidth = 0.0

    if len(data1_list) > 0:
        ax.scatter(
            positions[0],
            data1_list[0],
            s=scatter_size,
            c=scatter_colors[0],
            marker=scatter_marker_0,
            alpha=scatter_alpha,
            edgecolor=scatter_edgecolor,
            linewidth=scatter_linewidth,
            zorder=10,
            label=(label_prefix if positions[0] == ref_x - space_box / 2 else ""),
        )
    if len(data2_list) > 0:
        ax.scatter(
            positions[1],
            data2_list[0],
            s=scatter_size,
            c=scatter_colors[1],
            marker=scatter_marker_1,
            alpha=scatter_alpha,
            edgecolor=scatter_edgecolor,
            linewidth=scatter_linewidth,
            zorder=10,
            label=(label_prefix if positions[1] == ref_x + space_box / 2 else ""),
        )


def _make_case_patches(case_colors, language="en"):
    match language:
        case "cn":
            case_label = ["案例 1", "案例 2", "案例 3", "案例 4"]
        case "en":
            case_label = ["case 1", "case 2", "case 3", "case 4"]

    patches = []
    for i, key in enumerate(["case1", "case2", "case3", "case4"]):
        patches.append(
            mpatches.Patch(
                facecolor=case_colors[key],
                alpha=1,
                label=case_label[i],
                edgecolor="black",
            )
        )
    return patches


def _make_title(scenario_name, language="en", subplot_label=""):
    match language:
        case "cn":
            title_name = "工况"
        case "en":
            title_name = "scenario"
    prefix = f"({subplot_label}) " if subplot_label else ""
    return f"{prefix}{title_name} {scenario_name}"


# ============================================================
#  Public API
# ============================================================


def boxplot_4case(
    ax,
    case_data,
    data_label,
    scenario_name="2b",
    ax_lim=None,
    language="en",
    title=None,
    case_colors=None,
):
    """
    Draw 4-case boxplot on given ax (scenarios 2b, 2c).
    """
    colors = case_colors if case_colors else _CASE_COLORS
    x_positions = np.array([1, 2.5, 4, 5.5])
    width = 0.15
    space_box = 0.25

    axes = _draw_four_param_boxplot(
        ax,
        case_data,
        data_label,
        x_positions,
        width,
        space_box,
        ax_lim,
        colors_override=colors,
    )

    if title is None:
        match scenario_name:
            case "2b":
                title = _make_title(scenario_name, language, "b")
            case "2c":
                title = _make_title(scenario_name, language, "c")
            case _:
                title = _make_title(scenario_name, language)

    ax.set_title(title, pad=20)
    return axes


def boxplot_2case(
    ax,
    case_data,
    data_label,
    scenario_name="2a",
    ax_lim=None,
    language="en",
    title=None,
    case_colors=None,
):
    """
    Draw 2-case boxplot on given ax (scenario 2a).
    """
    colors = case_colors if case_colors else _CASE_COLORS
    x_positions = np.array([1, 2, 3, 4])
    width = 0.15
    space_box = 0.25

    axes = _draw_four_param_boxplot(
        ax,
        case_data,
        data_label,
        x_positions,
        width,
        space_box,
        ax_lim,
        colors_override=colors,
    )

    if title is None:
        title = _make_title(scenario_name, language, "a")

    ax.set_title(title, pad=20)
    return axes


def boxplot_case_1a2a(
    ax,
    case_data,
    data_label,
    case_data_1a=None,
    scenario_name="1a and 2a",
    ax_lim=None,
    language="en",
    title=None,
    case_colors=None,
):
    """
    Draw combined 1a + 2a boxplot with scatter overlay on given ax.
    """
    colors = case_colors if case_colors else _CASE_COLORS
    x_positions = np.array([1, 2, 3, 4])
    width = 0.15
    space_box = 0.25

    scatter_data = None
    scatter_label = None
    if case_data_1a is not None:
        match language:
            case "cn":
                scatter_label = "工况 1a"
            case "en":
                scatter_label = "scenario 1a"
        scatter_data = case_data_1a

    legend_font = FontProperties(size=20)

    axes = _draw_four_param_boxplot(
        ax,
        case_data,
        data_label,
        x_positions,
        width,
        space_box,
        ax_lim,
        colors_override=colors,
        scatter_data_1a=scatter_data,
        scatter_label_prefix=scatter_label,
        legend_font=legend_font,
    )

    # Scatter legend
    if case_data_1a is not None:
        match language:
            case "cn":
                case_label = "案例"
            case "en":
                case_label = "case"
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor="black",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.0,
                label=f"{scatter_label} {case_label} 1",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="black",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.0,
                label=f"{scatter_label} {case_label} 2",
            ),
        ]
        axes["ax1"].legend(handles=legend_elements, loc="upper left", prop=legend_font)

    if title is None:
        title = _make_title(scenario_name, language, "a")

    ax.set_title(title, pad=20)
    return axes


def create_boxplot_legend(
    ax,
    language="en",
    case_colors=None,
):
    """
    Draw a horizontal case-color legend on given ax.
    """
    colors = case_colors if case_colors else _CASE_COLORS
    legend_font = FontProperties(size=20)

    ax.axis("off")

    match language:
        case "cn":
            case_label = ["案例 1", "案例 2", "案例 3", "案例 4"]
        case "en":
            case_label = ["case 1", "case 2", "case 3", "case 4"]

    patches = []
    for i, key in enumerate(["case1", "case2", "case3", "case4"]):
        patches.append(
            mpatches.Patch(
                facecolor=colors[key],
                alpha=1,
                label=case_label[i],
                edgecolor="black",
            )
        )

    ax.legend(
        handles=patches,
        loc="center",
        ncol=4,
        fontsize=20,
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fancybox=True,
        facecolor="white",
        prop=legend_font,
    )
